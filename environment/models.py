# environment/models.py
# Typed Pydantic models for OpenEnv interface
# Implements Action, Observation, and Reward schemas

from typing import Optional
from pydantic import BaseModel, model_validator


class Action(BaseModel):
    """
    Action model for the SQL Analyst environment.
    
    The agent must provide EXACTLY ONE of:
    - sql_query: Execute a SQL query against the database
    - submit_answer: Submit a final answer for grading
    
    Edge Case Shield: Pydantic model_validator enforces mutual exclusivity.
    """
    sql_query: Optional[str] = None
    submit_answer: Optional[str] = None

    @model_validator(mode='after')
    def validate_exactly_one_action(self) -> 'Action':
        """
        Enforce that the agent provides exactly one of sql_query or submit_answer.
        This prevents ambiguous actions and ensures clean state transitions.
        """
        has_sql = self.sql_query is not None and self.sql_query.strip() != ""
        has_answer = self.submit_answer is not None and self.submit_answer.strip() != ""
        
        if has_sql and has_answer:
            raise ValueError(
                "Invalid action: Provide exactly ONE of 'sql_query' or 'submit_answer', not both."
            )
        
        if not has_sql and not has_answer:
            raise ValueError(
                "Invalid action: Must provide exactly ONE of 'sql_query' or 'submit_answer'."
            )
        
        return self


class Observation(BaseModel):
    """
    Observation model representing the current state visible to the agent.
    
    Fields:
    - schema_info: Database schema information (tables, columns, types)
    - current_question: The task question the agent must answer
    - last_query_result: Result from the most recent SQL query execution
    - error_message: Any error from the last action (empty string if none)
    """
    schema_info: str
    current_question: str
    last_query_result: str
    error_message: str


class Reward(BaseModel):
    """
    Reward model containing a single float value.
    
    Reward shaping:
    - +0.1: per table the task needs, the first time a query reads it
    - -0.01: any other successful query (e.g. SELECT 1, a repeat)
    - -0.1: SQLite syntax error
    - -1.0: Destructive action detected (done=True)
    - -0.5: Step count >= 15 (infinite loop shield, done=True)
    """
    value: float
