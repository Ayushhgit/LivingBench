"""
GitHub-based task generator.

Generates evaluation tasks from:
- GitHub issues (understanding, summarization, solution proposal)
- Pull request diffs (code review, bug detection)

These tasks test real-world software engineering understanding and
cannot be easily memorized due to temporal novelty.
"""

from __future__ import annotations

import re
import hashlib
from typing import Iterator, Any
from datetime import datetime, timedelta

import httpx

from livingbench.core.types import (
    Task,
    TaskSource,
    TaskFormat,
    Skill,
    DifficultyLevel,
)
from livingbench.tasks.base import TaskGeneratorBase
from livingbench.core.registry import task_generators


@task_generators.register("github_issue")
class GitHubTaskGenerator(TaskGeneratorBase):
    """
    Generate tasks from GitHub issues and PRs.

    Task types:
    1. Issue Understanding: Summarize the issue, identify the root cause
    2. Solution Proposal: Given an issue, propose a fix approach
    3. Code Review: Given a PR diff, identify bugs or improvements
    4. Issue Categorization: Classify issue type, severity, components
    """

    def __init__(
        self,
        repos: list[str] | None = None,
        api_token: str | None = None,
        cache_dir: str = "data/cache/github",
    ):
        self.repos = repos or [
            "python/cpython",
            "pytorch/pytorch",
            "huggingface/transformers",
        ]
        self.api_token = api_token
        self.cache_dir = cache_dir
        self.base_url = "https://api.github.com"

    @property
    def source(self) -> TaskSource:
        return TaskSource.GITHUB_ISSUE

    def generate(self, n_tasks: int, **kwargs) -> Iterator[Task]:
        """Generate tasks from GitHub issues."""
        # For offline/demo mode, generate synthetic GitHub-like tasks
        if not self.api_token:
            yield from self._generate_synthetic_github_tasks(n_tasks)
            return

        # Real API mode
        tasks_per_repo = n_tasks // len(self.repos) + 1

        for repo in self.repos:
            issues = self._fetch_issues(repo, tasks_per_repo * 2)

            for issue in issues[:tasks_per_repo]:
                task = self._issue_to_task(issue, repo)
                if task and self.validate_task(task):
                    yield task

    def _fetch_issues(self, repo: str, limit: int) -> list[dict]:
        """Fetch recent issues from a repository."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.api_token:
            headers["Authorization"] = f"token {self.api_token}"

        try:
            # Get issues from last 30 days for freshness
            since = (datetime.utcnow() - timedelta(days=30)).isoformat()

            response = httpx.get(
                f"{self.base_url}/repos/{repo}/issues",
                headers=headers,
                params={
                    "state": "all",
                    "sort": "created",
                    "direction": "desc",
                    "per_page": min(limit, 100),
                    "since": since,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return []

    def _issue_to_task(self, issue: dict, repo: str) -> Task | None:
        """Convert a GitHub issue to an evaluation task."""
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        labels = [l["name"] for l in issue.get("labels", [])]

        if len(body) < 100:  # Skip very short issues
            return None

        # Clean up the issue body
        body = self._clean_markdown(body)

        # Determine task type based on labels and content
        if any(l in ["bug", "defect", "error"] for l in labels):
            return self._create_bug_analysis_task(title, body, repo, issue)
        elif any(l in ["enhancement", "feature"] for l in labels):
            return self._create_feature_analysis_task(title, body, repo, issue)
        else:
            return self._create_understanding_task(title, body, repo, issue)

    def _create_bug_analysis_task(
        self, title: str, body: str, repo: str, issue: dict
    ) -> Task:
        """Create a bug analysis task."""
        prompt = f"""Analyze this bug report from the {repo} repository:

Title: {title}

Description:
{body[:2000]}

Tasks:
1. Summarize the bug in 2-3 sentences
2. Identify the likely root cause category (logic error, resource leak, race condition, etc.)
3. Suggest what area of the codebase is likely affected
4. Rate the severity (critical, high, medium, low)

Provide a structured analysis."""

        return Task(
            source=TaskSource.GITHUB_ISSUE,
            source_metadata={
                "repo": repo,
                "issue_number": issue.get("number"),
                "issue_url": issue.get("html_url"),
                "task_type": "bug_analysis",
            },
            prompt=prompt,
            context=body,
            format=TaskFormat.STRUCTURED_JSON,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.CODE_UNDERSTANDING,
                Skill.CAUSAL_INFERENCE,
            ],
            difficulty=DifficultyLevel.MEDIUM,
        )

    def _create_feature_analysis_task(
        self, title: str, body: str, repo: str, issue: dict
    ) -> Task:
        """Create a feature request analysis task."""
        prompt = f"""Analyze this feature request from the {repo} repository:

Title: {title}

Description:
{body[:2000]}

Tasks:
1. Summarize the requested feature in 2-3 sentences
2. Identify the main use case and user benefit
3. List 3 potential implementation challenges
4. Suggest related features or alternatives that might exist

Provide a structured analysis."""

        return Task(
            source=TaskSource.GITHUB_ISSUE,
            source_metadata={
                "repo": repo,
                "issue_number": issue.get("number"),
                "task_type": "feature_analysis",
            },
            prompt=prompt,
            context=body,
            format=TaskFormat.STRUCTURED_JSON,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.MULTI_STEP_PLANNING,
                Skill.KNOWLEDGE_INTEGRATION,
            ],
            difficulty=DifficultyLevel.MEDIUM,
        )

    def _create_understanding_task(
        self, title: str, body: str, repo: str, issue: dict
    ) -> Task:
        """Create a general understanding task."""
        prompt = f"""Read this GitHub issue from the {repo} repository:

Title: {title}

Description:
{body[:2000]}

Answer the following:
1. What is the main topic/concern raised in this issue?
2. Is this a bug report, feature request, question, or discussion?
3. What technical knowledge would be needed to address this issue?
4. What additional information would be helpful to have?

Provide clear, structured answers."""

        return Task(
            source=TaskSource.GITHUB_ISSUE,
            source_metadata={
                "repo": repo,
                "issue_number": issue.get("number"),
                "task_type": "understanding",
            },
            prompt=prompt,
            context=body,
            format=TaskFormat.FREE_TEXT,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.KNOWLEDGE_INTEGRATION,
            ],
            difficulty=DifficultyLevel.EASY,
        )

    def _generate_synthetic_github_tasks(self, n_tasks: int) -> Iterator[Task]:
        """Generate synthetic GitHub-like tasks for demo/offline mode."""
        import random

        rng = random.Random(42)

        issue_templates = [
            {
                "title": "Memory leak in dataset loader when using multiple workers",
                "body": """When using DataLoader with num_workers > 0, memory usage continuously increases over epochs.

**Environment:**
- Python 3.10
- PyTorch 2.0
- Ubuntu 22.04, 64GB RAM

**To Reproduce:**
```python
loader = DataLoader(dataset, batch_size=32, num_workers=4)
for epoch in range(100):
    for batch in loader:
        model(batch)  # Memory grows each epoch
```

**Expected behavior:**
Memory should stay constant across epochs.

**Actual behavior:**
Memory increases by ~100MB per epoch until OOM.

I've tried setting pin_memory=False and persistent_workers=False but the issue persists.""",
                "type": "bug",
                "skills": [Skill.CODE_UNDERSTANDING, Skill.CAUSAL_INFERENCE],
                "difficulty": DifficultyLevel.HARD,
            },
            {
                "title": "Add support for async context managers in middleware",
                "body": """Currently, the middleware system only supports synchronous context managers. This limits use cases where async setup/teardown is needed.

**Use Case:**
I need to acquire a database connection from an async pool before handling requests:

```python
class DatabaseMiddleware:
    async def __aenter__(self):
        self.conn = await pool.acquire()  # Not currently possible
        return self

    async def __aexit__(self, *args):
        await self.conn.release()
```

**Proposal:**
Extend the middleware protocol to support `__aenter__`/`__aexit__` in addition to `__enter__`/`__exit__`.

This would require modifying the request handling pipeline to be fully async-aware.""",
                "type": "feature",
                "skills": [Skill.CODE_UNDERSTANDING, Skill.MULTI_STEP_PLANNING],
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "title": "Gradient checkpointing causes NaN with mixed precision",
                "body": """Using gradient checkpointing together with AMP (automatic mixed precision) causes NaN gradients.

**Minimal reproduction:**
```python
model = TransformerModel()
model.gradient_checkpointing_enable()

with torch.cuda.amp.autocast():
    output = model(input_ids)
    loss = output.loss

scaler.scale(loss).backward()  # Gradients become NaN
```

**Analysis:**
The issue seems to occur in the attention computation during the recomputation phase. The softmax output becomes all zeros in fp16, leading to division by zero in the backward pass.

**Possible fix:**
Apply autocast only to specific operations that are numerically stable in fp16, or force fp32 for the attention softmax during checkpointing.""",
                "type": "bug",
                "skills": [
                    Skill.CODE_UNDERSTANDING,
                    Skill.MATHEMATICAL_REASONING,
                    Skill.CAUSAL_INFERENCE,
                ],
                "difficulty": DifficultyLevel.VERY_HARD,
            },
        ]

        for i in range(n_tasks):
            template = rng.choice(issue_templates)

            if template["type"] == "bug":
                prompt = f"""Analyze this bug report:

Title: {template['title']}

Description:
{template['body']}

Tasks:
1. Summarize the bug in 2-3 sentences
2. Identify the likely root cause
3. Suggest a debugging approach
4. Rate the severity (critical, high, medium, low)

Provide structured analysis."""
            else:
                prompt = f"""Analyze this feature request:

Title: {template['title']}

Description:
{template['body']}

Tasks:
1. Summarize the requested feature
2. Identify the main use case
3. List implementation challenges
4. Suggest alternatives if any

Provide structured analysis."""

            yield Task(
                source=TaskSource.GITHUB_ISSUE,
                source_metadata={
                    "synthetic": True,
                    "template_index": i % len(issue_templates),
                },
                prompt=prompt,
                context=template["body"],
                format=TaskFormat.FREE_TEXT,
                required_skills=template["skills"],
                difficulty=template["difficulty"],
            )

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Clean markdown formatting for clarity."""
        # Remove image links
        text = re.sub(r"!\[.*?\]\(.*?\)", "[image]", text)
        # Simplify code blocks
        text = re.sub(r"```(\w+)?\n", "```\n", text)
        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        return text.strip()
