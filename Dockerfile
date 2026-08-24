# OpenEnv SQL Analyst Environment
# Base: python:3.10-slim for minimal memory footprint (<8GB RAM limit)

FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies first so edits to application code do not
# invalidate this layer. No build toolchain is required: every dependency
# publishes a pure-Python or manylinux wheel, so gcc is not installed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (see .dockerignore for what is excluded)
COPY . .

# Run as an unprivileged user. This container executes agent-supplied SQL, so
# it should hold no more privilege than the task actually requires.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the OpenEnv serving port
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Serve the FastAPI app directly. The previous CMD shelled out to `uv run`,
# which re-resolved the project's dependencies on every container start.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
