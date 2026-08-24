import os
import uvicorn
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from environment.env import SQLAnalystEnv
from environment.models import Action

# Initialize the API and our RL Environment.
# One environment per container, matching OpenEnv's isolation model.
app = FastAPI(title="OpenEnv SQL Analyst")
env = SQLAnalystEnv()


class ResetRequest(BaseModel):
    """Optional reset parameters, mirroring the OpenEnv reset contract."""

    seed: Optional[int] = None
    task_id: Optional[str] = None
    episode_id: Optional[str] = None


@app.get("/")
def health_check():
    """Hackathon requirement: Ping must return 200 OK"""
    return {"status": "ok", "message": "OpenEnv SQL Analyst is live!"}

@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    """
    Hackathon requirement: Must respond to reset()

    Accepts an optional seed so an episode can be reproduced exactly, and an
    optional task_id to pin a specific task.
    """
    request = request or ResetRequest()
    return env.reset(
        seed=request.seed,
        task_id=request.task_id,
        episode_id=request.episode_id,
    )

@app.post("/step")
def step(action: Action):
    """Executes the agent's action and returns the new state"""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

def main():
    print("🚀 Starting OpenEnv Production Server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()