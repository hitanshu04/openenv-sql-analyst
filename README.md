---
title: OpenEnv SQL Analyst
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# SQL Data Analyst RL Environment

> A production-grade, containerized Reinforcement Learning environment for evaluating LLM-powered Data Analysts on real SQL business intelligence tasks.

**OpenEnv Hackathon Submission** | Meta x Scaler

---

## Environment Description and Motivation

This environment simulates a **mission-critical enterprise task**: an AI agent querying a production SQL database to extract business intelligence. In real-world enterprises, data analysts spend countless hours writing SQL queries to answer ad-hoc business questions from stakeholders. This environment provides a standardized benchmark to evaluate whether LLM agents can safely and accurately perform this task autonomously, measuring both **correctness** and **efficiency**.

### Why This Matters

- **Real-World Applicability**: Data analysis is one of the most common knowledge work tasks that LLMs are being deployed for
- **Safety-Critical**: Database access requires strict guardrails to prevent data corruption
- **Measurable Outcomes**: Business questions have definitive correct answers, enabling objective evaluation

### Production-Grade Security

The environment implements security safeguards that mirror real enterprise database access controls:

| Security Layer | Implementation | Purpose |
|----------------|----------------|---------|
| **Mutation Blocker** | Regex-based blocking of `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE` | Prevents data corruption |
| **OOM Protection** | `cursor.fetchmany(50)` instead of `fetchall()` | Prevents memory exhaustion on large result sets |
| **Query Timeout** | 2-second timeout wrapper | Prevents runaway queries from consuming resources |
| **Read-Only Sandbox** | In-memory SQLite (`:memory:` mode) | Isolated execution environment |

---

## Action Space

The agent submits an `Action` object with **exactly one** of two fields:

| Field | Type | Description |
|-------|------|-------------|
| `sql_query` | `Optional[str]` | Execute a SQL query against the database |
| `submit_answer` | `Optional[str]` | Submit a final answer for grading |

**Mutual Exclusivity Enforced**: A Pydantic `@model_validator` ensures the agent provides exactly one of `sql_query` or `submit_answer`. Providing both or neither raises a `ValueError`.

```python
# Example Actions
action_query = Action(sql_query="SELECT COUNT(*) FROM users")
action_submit = Action(submit_answer="15")
```

---

## Observation Space

The agent receives an `Observation` object containing four fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema_info` | `str` | Database schema information (tables, columns, types) |
| `current_question` | `str` | The business question the agent must answer |
| `last_query_result` | `str` | Result from the most recent SQL query (markdown table format) |
| `error_message` | `str` | Any error from the last action (empty string if none) |

---

## Reward Shaping

The environment implements precise partial reward signals to guide learning:

| Event | Reward | Episode Ends? |
|-------|--------|---------------|
| Successful SQL query (no errors) | `+0.1` | No |
| SQLite syntax error | `-0.1` | No |
| Destructive action detected | `-1.0` | **Yes** |
| Step count >= 15 (infinite loop shield) | `-0.5` | **Yes** |
| Correct answer submitted | `+1.0` | **Yes** |
| Incorrect answer submitted | `0.0` | **Yes** |

**Final Score Calculation**: 
- If incorrect: `score = 0.0`
- If correct: `score = 0.7 + (1 - steps/15) * 0.3`
- Score range: `0.0` to `1.0`

---

## Task Descriptions

The environment includes **3 deterministic tasks** of increasing difficulty:

### Easy: User Count
| Attribute | Value |
|-----------|-------|
| **Task ID** | `easy_user_count` |
| **Difficulty** | Easy |
| **Question** | "How many users are registered in the system? Provide the total count as a single number." |
| **Ground Truth** | `15` |
| **SQL Complexity** | Single table `COUNT` query |
| **Reference SQL** | `SELECT COUNT(*) FROM users` |

### Medium: USA Revenue
| Attribute | Value |
|-----------|-------|
| **Task ID** | `medium_usa_revenue` |
| **Difficulty** | Medium |
| **Question** | "What is the total revenue (sum of total_amount) from purchases made by users in the USA? Provide the total as a number (rounded to 2 decimal places if needed)." |
| **Ground Truth** | `2423.87` |
| **SQL Complexity** | Two-table `JOIN` with `SUM` aggregation filtered by country |
| **Reference SQL** | `SELECT ROUND(SUM(p.total_amount), 2) FROM purchases p JOIN users u ON p.user_id = u.user_id WHERE u.country = 'USA'` |

### Hard: Top Spender
| Attribute | Value |
|-----------|-------|
| **Task ID** | `hard_top_spender` |
| **Difficulty** | Hard |
| **Question** | "Who is the top spender (user with highest total purchase amount)? Provide the username of the user who spent the most money in total." |
| **Ground Truth** | `alice` |
| **SQL Complexity** | Complex query with `JOIN`, `GROUP BY`, `ORDER BY`, and `LIMIT` |
| **Reference SQL** | `SELECT u.username FROM users u JOIN purchases p ON u.user_id = p.user_id GROUP BY u.user_id, u.username ORDER BY SUM(p.total_amount) DESC LIMIT 1` |

### Grading System

All graders implement:
- **Type-agnostic normalization**: Whitespace trimming, lowercasing, numeric rounding to 2 decimal places
- **Numeric tolerance**: Answers within 0.01 absolute tolerance are exact matches
- **Partial credit**: Numeric answers within 10% receive 0.5 score
- **SQL evaluation**: If agent submits SQL as answer, it's executed and results compared

---

## Setup and Usage Instructions

### Prerequisites

- Docker installed and running
- Python 3.10+ (for local development)
- (Optional) HuggingFace token for inference with HF-hosted models

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/hitanshu04/openenv-sql-analyst.git
cd openenv_sql_analyst

# Build the Docker image
docker build -t openenv-sql-analyst .

# Run the container
docker run -p 7860:7860 openenv-sql-analyst
```

The server will be available at `http://localhost:7860`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check (returns 200 OK) |
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Execute action, returns (observation, reward, done, info) |
| `/state` | GET | Get current internal state |

### Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server directly
python -m server.app

# Or run validation
chmod +x validate.sh
./validate.sh
```

### Running Inference

```bash
# Set environment variables
export API_KEY="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"  # or LiteLLM proxy endpoint
export MODEL_NAME="gpt-4o-mini"

# Run inference
python inference.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API key for LLM provider (injected by hackathon) | **Required** |
| `API_BASE_URL` | OpenAI-compatible API endpoint (injected by hackathon) | **Required** |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |

### Validation Gates

Run `./validate.sh` before submission. All 4 checks must pass:

| Step | Check | Failure Condition |
|------|-------|-------------------|
| 1/4 | Prerequisites | `docker` or `openenv` CLI not found |
| 2/4 | Docker Build | `Dockerfile` missing or build fails |
| 3/4 | OpenEnv Spec | `openenv validate` fails (yaml/models mismatch) |
| 4/4 | Inference Logs | Missing `[START]`/`[STEP]`/`[END]` tags or invalid score |

---

## Baseline Scores

Expected performance with `gpt-4o-mini`:

| Task | Difficulty | Expected Steps | Expected Score |
|------|------------|----------------|----------------|
| `easy_user_count` | Easy | 2-3 | 0.90 - 1.00 |
| `medium_usa_revenue` | Medium | 3-5 | 0.85 - 0.95 |
| `hard_top_spender` | Hard | 4-7 | 0.75 - 0.90 |

### STDOUT Log Format

The inference script outputs logs in the exact required format:

```
[START] task=<task_id> env=sql_analyst model=<model_name>
[STEP]  step=<n> action=<action_type>=<value> reward=<r.rr> done=<bool> error=<msg>
[END]   success=<bool> steps=<n> score=<s.ss> rewards=<r1>,<r2>,...
```

**Example Output**:
```
[START] task=easy_user_count env=sql_analyst model=gpt-4o-mini
[STEP]  step=1 action=sql_query=SELECT COUNT(*) FROM users reward=0.10 done=false error=null
[STEP]  step=2 action=submit_answer=15 reward=1.00 done=true error=null
[END]   success=true steps=2 score=0.96 rewards=0.10,1.00
```

---

## Project Architecture

```
openenv_sql_analyst/
├── openenv.yaml          # OpenEnv specification (name, schemas, endpoints)
├── Dockerfile            # Container config (python:3.10-slim, port 7860)
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Python project configuration
├── validate.sh           # Pre-submission validation (4 gates)
├── inference.py          # Baseline LLM agent implementation
├── data/
│   └── mock_data.sql     # SQLite mock database (3 tables, ~50 rows)
├── environment/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Pydantic schemas (Action, Observation, Reward)
│   ├── db_engine.py      # SQLite engine with security safeguards
│   ├── tasks.py          # Task definitions (Easy, Medium, Hard)
│   ├── graders.py        # Deterministic grading system
│   └── env.py            # Main SQLAnalystEnv class (reset, step, state)
└── server/
    └── app.py            # FastAPI server (/reset, /step, /state endpoints)
```

---

## Technical Specifications

| Specification | Value |
|---------------|-------|
| Python Version | 3.10 |
| Container Base | `python:3.10-slim` |
| Container Port | 7860 |
| vCPU Limit | 2 |
| Memory Limit | 8 GB |
| Max Runtime | 20 minutes |
| Max Steps per Episode | 15 |
| Query Timeout | 2 seconds |
| Max Fetch Rows | 50 |
| Database | SQLite (in-memory) |

---

## Database Schema

The mock database contains 3 tables:

### users
| Column | Type | Constraints |
|--------|------|-------------|
| user_id | INTEGER | PRIMARY KEY |
| username | TEXT | NOT NULL |
| email | TEXT | NOT NULL |
| country | TEXT | NOT NULL |
| created_at | TEXT | NOT NULL |

### products
| Column | Type | Constraints |
|--------|------|-------------|
| product_id | INTEGER | PRIMARY KEY |
| product_name | TEXT | NOT NULL |
| category | TEXT | NOT NULL |
| price | REAL | NOT NULL |
| stock | INTEGER | NOT NULL |

### purchases
| Column | Type | Constraints |
|--------|------|-------------|
| purchase_id | INTEGER | PRIMARY KEY |
| user_id | INTEGER | NOT NULL, FOREIGN KEY |
| product_id | INTEGER | NOT NULL, FOREIGN KEY |
| quantity | INTEGER | NOT NULL |
| purchase_date | TEXT | NOT NULL |
| total_amount | REAL | NOT NULL |

---

## License

MIT License

---

## Acknowledgments

Built for the **Meta x Scaler OpenEnv Hackathon** - advancing the frontier of LLM agent evaluation through standardized, production-grade reinforcement learning environments.
