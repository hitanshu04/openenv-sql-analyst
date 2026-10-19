"""Model backends. Each chat turns a message history into (reply text, usage)."""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAI, RateLimitError

from inference import extract_response_text

ROOT = Path(__file__).resolve().parent.parent
ENV = {**dotenv_values(ROOT / ".env"), **os.environ}

MAX_REPLY_TOKENS = 2048  # room for reasoning models; v1's 500 can truncate them


class QuotaExhausted(Exception):
    """A free tier ran out. Stop this model; a later run resumes it."""


class InfraError(Exception):
    """The call failed for reasons that are not the model's answer."""


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    provider: str  # a key of OPENAI_COMPATIBLE, or "claude-cli"
    model_id: str
    served_via: str


OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN"),
}

# Every model here is reachable at no cost (19 Oct 2026). IDs were taken from
# each provider's live model list, not guessed.
LINEUP = [
    ModelSpec("claude-sonnet-5-5", "claude-cli", "claude-sonnet-5-5",
              "Claude Code CLI on a claude.ai subscription, tools disabled"),
    ModelSpec("claude-haiku-4-5", "claude-cli", "haiku",
              "Claude Code CLI on a claude.ai subscription, tools disabled"),
    # 3.8 Flash's free tier allows 20 requests/day, so a full run takes days.
    ModelSpec("gemini-3.8-flash", "gemini", "gemini-3.8-flash", "Google AI Studio free tier"),
    ModelSpec("gemini-3.5-flash", "gemini", "gemini-3.5-flash", "Google AI Studio free tier"),
    ModelSpec("gpt-oss-120b", "groq", "openai/gpt-oss-120b", "Groq free tier"),
    ModelSpec("gpt-oss-20b", "groq", "openai/gpt-oss-20b", "Groq free tier"),
    ModelSpec("qwen3.8-27b", "groq", "qwen/qwen3.8-27b", "Groq free tier"),
]

QUOTA_WORDS = ("per day", "tokens per day", "requests per day", "exceeded your current quota",
               "usage limit", "limit reached", "quota")


def _is_quota(text):
    return any(word in text.lower() for word in QUOTA_WORDS)


def _failed_generation(error):
    """The model's own text, when a provider rejects it as a malformed tool call."""
    body = error.body if isinstance(error.body, dict) else {}
    body = body.get("error", body)
    if body.get("code") == "tool_use_failed" and body.get("failed_generation") is not None:
        return str(body["failed_generation"])
    return None


class OpenAICompatibleChat:
    """Stateless: the full history is sent every turn, at temperature 0."""

    def __init__(self, spec, client=None, retries=6):
        if client is None:
            base_url, key_name = OPENAI_COMPATIBLE[spec.provider]
            if not ENV.get(key_name):
                raise InfraError(f"{key_name} is not set in .env")
            client = OpenAI(base_url=base_url, api_key=ENV[key_name], max_retries=0)
        self.client = client
        self.spec = spec
        self.retries = retries

    def reply(self, messages):
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=MAX_REPLY_TOKENS,
                )
                usage = response.usage
                return extract_response_text(response.choices[0].message), {
                    "served_model": response.model,
                    "tokens_in": getattr(usage, "prompt_tokens", 0) or 0,
                    "tokens_out": getattr(usage, "completion_tokens", 0) or 0,
                }
            except RateLimitError as exc:
                # Per-minute limits clear quickly; daily ones don't. Back off,
                # and give up on this model if it keeps failing.
                if "per day" in str(exc).lower() or attempt == self.retries - 1:
                    raise QuotaExhausted(str(exc)[:300]) from exc
                time.sleep(min(60, 5 * 2 ** attempt))
            except BadRequestError as exc:
                # Groq rejects a generation that calls a tool we never offered
                # and returns its text. That is the model's malformed answer,
                # so it goes back as the reply and becomes a parse error.
                text = _failed_generation(exc)
                if text is None:
                    raise InfraError(str(exc)[:300]) from exc
                return text, {"served_model": self.spec.model_id, "tokens_in": 0, "tokens_out": 0}
            except (APIConnectionError, APIStatusError) as exc:
                if attempt == self.retries - 1:
                    raise InfraError(str(exc)[:300]) from exc
                time.sleep(min(60, 5 * 2 ** attempt))
        raise InfraError("unreachable")


class ClaudeCliChat:
    """
    Claude through the Claude Code CLI in headless mode: no tools, safe mode
    (no CLAUDE.md, skills, hooks or MCP), our system prompt instead of Claude
    Code's, run from an empty directory. The conversation continues natively
    with --resume, so only the newest user message is sent each turn.
    """

    def __init__(self, spec, workdir):
        self.spec = spec
        self.workdir = workdir
        self.session_id = None
        self.sent = 0

    def reply(self, messages):
        system = messages[0]["content"]
        new = [m["content"] for m in messages[max(self.sent, 1):] if m["role"] == "user"]
        command = ["claude", "-p", "--safe-mode", "--tools", "", "--model", self.spec.model_id,
                   "--system-prompt", system, "--output-format", "json"]
        if self.session_id:
            command += ["--resume", self.session_id]
        command.append("\n\n".join(new))

        try:
            done = subprocess.run(command, cwd=self.workdir, capture_output=True,
                                  text=True, timeout=300)
        except subprocess.TimeoutExpired as exc:
            raise InfraError("claude CLI timed out") from exc
        output = done.stdout.strip()
        try:
            data = json.loads(output[output.index("{"):])
        except ValueError as exc:
            raise InfraError(f"unreadable CLI output: {output[:200]} {done.stderr[:200]}") from exc
        if data.get("is_error"):
            text = str(data.get("result", ""))
            raise (QuotaExhausted if _is_quota(text) else InfraError)(text[:300])

        self.session_id = data.get("session_id")
        self.sent = len(messages)
        # The CLI also uses a small model for housekeeping; count only the agent model.
        wants_haiku = "haiku" in self.spec.model_id
        usage = {k: v for k, v in (data.get("modelUsage") or {}).items()
                 if ("haiku" in k) == wants_haiku} or data.get("modelUsage") or {}
        return str(data.get("result", "")), {
            "served_model": ",".join(usage),
            "tokens_in": sum(v.get("inputTokens", 0) + v.get("cacheReadInputTokens", 0)
                             + v.get("cacheCreationInputTokens", 0) for v in usage.values()),
            "tokens_out": sum(v.get("outputTokens", 0) for v in usage.values()),
        }


def new_chat(spec, workdir=None):
    """A fresh conversation with this model, for one episode."""
    if spec.provider == "claude-cli":
        return ClaudeCliChat(spec, workdir or tempfile.mkdtemp(prefix="sqlgym-claude-"))
    return OpenAICompatibleChat(spec)
