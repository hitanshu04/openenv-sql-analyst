# environment/env.py
# Main OpenEnv Environment for SQL Data Analyst
# Inherits from openenv.BaseEnv and implements reset(), step(), state()

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from .models import Action, Observation, Reward
from .db_engine import DatabaseEngine, QueryStatus
from .tasks import (
    Task,
    TASKS,
    get_random_task,
    resolve_answer_domain,
    resolve_ground_truth,
)
from .graders import grade_answer, calculate_final_score

# Try to import openenv.BaseEnv, fallback to a simple base class if not available
try:
    from openenv import BaseEnv
except ImportError:
    # Fallback base class for development/testing
    class BaseEnv:
        """Fallback base class when openenv-core is not installed."""
        pass


# ============================================
# REWARD CONSTANTS (per PRD specification)
# ============================================
REWARD_SUCCESSFUL_QUERY = 0.1      # Successful, error-free SQL query
REWARD_SYNTAX_ERROR = -0.1         # SQLite syntax error
REWARD_DESTRUCTIVE_ACTION = -1.0   # Destructive action detected
REWARD_INFINITE_LOOP = -0.5        # Step count >= 15

# Maximum steps before infinite loop shield activates
MAX_STEPS = 15


@dataclass
class EnvironmentState:
    """
    Internal state of the SQL Analyst environment.
    
    Attributes:
        task: The current task being solved
        ground_truth: Answer derived from the task's reference SQL at reset
        answer_domain: Legitimate candidate answers, used to reject hedging
        seed: Seed used for this episode, echoed back for reproducibility
        episode_id: Optional caller-supplied episode identifier
        step_count: Number of steps taken in current episode
        done: Whether the episode has ended
        last_query_result: Result from the most recent SQL query
        error_message: Error message from the last action
        rewards: List of all rewards received in this episode
        final_score: The final grading score (0.0 to 1.0)
        success: Whether the task was completed successfully
    """
    task: Optional[Task] = None
    ground_truth: Any = None
    answer_domain: Optional[list] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    step_count: int = 0
    done: bool = False
    last_query_result: str = ""
    error_message: str = ""
    rewards: list = field(default_factory=list)
    final_score: float = 0.0
    success: bool = False


class SQLAnalystEnv(BaseEnv):
    """
    SQL Data Analyst Reinforcement Learning Environment.
    
    This environment simulates a Data Analyst workspace where an AI agent
    queries a SQLite database to answer business questions.
    
    Implements the OpenEnv interface:
    - reset(): Initialize a clean episode
    - step(action): Execute an action and return (observation, reward, done, info)
    - state(): Return the current internal state
    
    Reward Shaping (per PRD):
    - +0.1: Successful, error-free SQL query
    - -0.1: SQLite syntax error
    - -1.0: Destructive action detected (done=True)
    - -0.5: Step count >= 15 (infinite loop shield, done=True)
    """
    
    def __init__(self):
        """Initialize the SQL Analyst environment."""
        super().__init__()
        self.db_engine = DatabaseEngine()
        self._state = EnvironmentState()
    
    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> Observation:
        """
        Reset the environment to start a new episode.

        This method:
        1. Initializes a clean in-memory SQLite database
        2. Selects 1 of the 3 tasks (specified, or drawn from a seeded RNG)
        3. Resets step_count to 0
        4. Returns the initial observation

        Args:
            seed: Optional seed making task selection reproducible. Matches the
                OpenEnv reset(seed=..., episode_id=...) interface; without it,
                episodes could not be replayed.
            task_id: Optional specific task to use
            episode_id: Optional caller-supplied episode identifier

        Returns:
            Observation: The initial observation for the episode
        """
        # Initialize clean database
        self.db_engine.initialize()

        self._state.seed = seed
        self._state.episode_id = episode_id

        # Select task
        if task_id:
            for task in TASKS:
                if task.task_id == task_id:
                    self._state.task = task
                    break
            else:
                self._state.task = get_random_task(seed=seed)
        else:
            self._state.task = get_random_task(seed=seed)

        # Derive the correct answer from the task's own reference SQL, against
        # the same data the agent will query. Ground truth cannot drift.
        self._state.ground_truth = resolve_ground_truth(self._state.task, self.db_engine)
        self._state.answer_domain = resolve_answer_domain(self._state.task, self.db_engine)

        # Reset state
        self._state.step_count = 0
        self._state.done = False
        self._state.last_query_result = ""
        self._state.error_message = ""
        self._state.rewards = []
        self._state.final_score = 0.0
        self._state.success = False
        
        # Build initial observation
        return Observation(
            schema_info=self.db_engine.get_schema(),
            current_question=self._state.task.question,
            last_query_result="No queries executed yet.",
            error_message=""
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        This method processes the agent's action and returns:
        - observation: The new state after the action
        - reward: The reward for this action
        - done: Whether the episode has ended
        - info: Additional information
        
        Reward Shaping:
        - +0.1: Successful, error-free SQL query
        - -0.1: SQLite syntax error
        - -1.0: Destructive action detected (done=True)
        - -0.5: Step count >= 15 (done=True)
        
        Args:
            action: The Action to execute
            
        Returns:
            Tuple containing (observation, reward, done, info)
        """
        if self._state.done:
            # Episode already ended
            return self._get_observation(), Reward(value=0.0), True, self._get_info()

        # Increment step count
        self._state.step_count += 1

        # Initialize reward for this step
        reward = 0.0
        self._state.error_message = ""

        # Process action. The action is executed BEFORE the loop shield is
        # considered, so the agent gets all MAX_STEPS actions -- previously the
        # shield fired first and silently discarded the final action, leaving
        # only MAX_STEPS - 1 usable.
        if action.sql_query:
            reward = self._handle_sql_query(action.sql_query)
        elif action.submit_answer:
            reward = self._handle_submit_answer(action.submit_answer)

        # Infinite loop shield: the agent has now used its full step budget
        # without terminating on its own.
        if not self._state.done and self._state.step_count >= MAX_STEPS:
            self._state.done = True
            self._state.error_message = (
                f"Maximum steps ({MAX_STEPS}) reached. Episode terminated."
            )
            reward = REWARD_INFINITE_LOOP

        # Record reward
        self._state.rewards.append(reward)
        
        return self._get_observation(), Reward(value=reward), self._state.done, self._get_info()
    
    def _handle_sql_query(self, query: str) -> float:
        """
        Handle a SQL query action.
        
        Args:
            query: The SQL query to execute
            
        Returns:
            float: The reward for this action
        """
        result, status = self.db_engine.execute_query(query)

        # SQLite's authorizer refused a write/attach/pragma: destructive intent.
        if status == QueryStatus.DENIED:
            self._state.done = True
            self._state.error_message = result
            self._state.last_query_result = ""
            return REWARD_DESTRUCTIVE_ACTION

        if status == QueryStatus.ERROR:
            self._state.error_message = result
            self._state.last_query_result = ""
            return REWARD_SYNTAX_ERROR

        # Successful query
        self._state.last_query_result = result
        self._state.error_message = ""
        return REWARD_SUCCESSFUL_QUERY
    
    def _handle_submit_answer(self, answer: str) -> float:
        """
        Handle a submit answer action.
        
        Args:
            answer: The answer to submit for grading
            
        Returns:
            float: The reward for this action
        """
        # Episode ends when answer is submitted
        self._state.done = True
        
        # Grade the answer against the ground truth derived at reset()
        is_correct, grading_score = grade_answer(
            answer,
            self._state.ground_truth,
            self.db_engine,
            self._state.answer_domain,
        )

        # Calculate final score. grading_score carries the grader's partial
        # credit for near-miss numeric answers, which the score must honour.
        self._state.success = is_correct
        self._state.final_score = calculate_final_score(
            is_correct,
            grading_score,
            self._state.step_count,
            MAX_STEPS
        )
        
        # Reward for submission is based on correctness
        # This is separate from the final_score which considers efficiency
        if is_correct:
            return 1.0  # Full reward for correct answer
        else:
            return 0.0  # No reward for incorrect answer
    
    def _get_observation(self) -> Observation:
        """
        Build the current observation.
        
        Returns:
            Observation: The current state visible to the agent
        """
        return Observation(
            schema_info=self.db_engine.get_schema(),
            current_question=self._state.task.question if self._state.task else "",
            last_query_result=self._state.last_query_result or "No results yet.",
            error_message=self._state.error_message
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Build the info dictionary.
        
        Returns:
            Dict: Additional information about the current state
        """
        return {
            "step_count": self._state.step_count,
            "task_id": self._state.task.task_id if self._state.task else None,
            "task_difficulty": self._state.task.difficulty if self._state.task else None,
            "success": self._state.success,
            "final_score": self._state.final_score,
            "total_reward": sum(self._state.rewards),
            "rewards_history": self._state.rewards.copy()
        }
    
    def state(self) -> Dict[str, Any]:
        """
        Return the current internal state of the environment.
        
        Returns:
            Dict: The full internal state
        """
        return {
            "task_id": self._state.task.task_id if self._state.task else None,
            "task_difficulty": self._state.task.difficulty if self._state.task else None,
            "task_question": self._state.task.question if self._state.task else None,
            "seed": self._state.seed,
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "done": self._state.done,
            "last_query_result": self._state.last_query_result,
            "error_message": self._state.error_message,
            "rewards": self._state.rewards.copy(),
            "total_reward": sum(self._state.rewards),
            "success": self._state.success,
            "final_score": self._state.final_score
        }
    
    def close(self):
        """Clean up resources."""
        if self.db_engine:
            self.db_engine.close()
