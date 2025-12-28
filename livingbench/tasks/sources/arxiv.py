"""
ArXiv-based task generator.

Generates evaluation tasks from scientific paper abstracts:
- Comprehension questions
- Method extraction
- Contribution summarization
- Related work identification

These tasks test technical reading comprehension and
scientific reasoning abilities.
"""

from __future__ import annotations

import re
import random
from typing import Iterator
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


@task_generators.register("arxiv")
class ArxivTaskGenerator(TaskGeneratorBase):
    """
    Generate tasks from ArXiv paper abstracts.

    Task types:
    1. Comprehension: Answer questions about the abstract
    2. Method Extraction: Identify key techniques/methods
    3. Contribution Summary: What's new in this paper?
    4. Limitation Inference: What might be limitations?
    """

    def __init__(
        self,
        categories: list[str] | None = None,
        cache_dir: str = "data/cache/arxiv",
        seed: int = 42,
    ):
        self.categories = categories or ["cs.CL", "cs.LG", "cs.AI"]
        self.cache_dir = cache_dir
        self.rng = random.Random(seed)
        self.base_url = "http://export.arxiv.org/api/query"

    @property
    def source(self) -> TaskSource:
        return TaskSource.ARXIV_ABSTRACT

    def generate(self, n_tasks: int, **kwargs) -> Iterator[Task]:
        """Generate tasks from ArXiv abstracts."""
        use_api = kwargs.get("use_api", False)

        if use_api:
            yield from self._generate_from_api(n_tasks)
        else:
            yield from self._generate_synthetic_arxiv_tasks(n_tasks)

    def _generate_from_api(self, n_tasks: int) -> Iterator[Task]:
        """Fetch real papers from ArXiv API."""
        papers_per_category = n_tasks // len(self.categories) + 1

        for category in self.categories:
            papers = self._fetch_papers(category, papers_per_category * 2)

            for paper in papers[:papers_per_category]:
                task = self._paper_to_task(paper)
                if task and self.validate_task(task):
                    yield task

    def _fetch_papers(self, category: str, limit: int) -> list[dict]:
        """Fetch recent papers from a category."""
        try:
            response = httpx.get(
                self.base_url,
                params={
                    "search_query": f"cat:{category}",
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": min(limit, 100),
                },
                timeout=30,
            )
            response.raise_for_status()

            # Parse Atom feed
            return self._parse_arxiv_response(response.text)
        except Exception:
            return []

    def _parse_arxiv_response(self, xml_text: str) -> list[dict]:
        """Parse ArXiv Atom feed response."""
        papers = []

        # Simple regex-based parsing (for robustness without xml dependencies)
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for entry in entries:
            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            abstract_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            id_match = re.search(r"<id>(.*?)</id>", entry)

            if title_match and abstract_match:
                papers.append({
                    "title": self._clean_text(title_match.group(1)),
                    "abstract": self._clean_text(abstract_match.group(1)),
                    "id": id_match.group(1) if id_match else None,
                })

        return papers

    def _paper_to_task(self, paper: dict) -> Task | None:
        """Convert paper abstract to evaluation task."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if len(abstract) < 200:
            return None

        # Randomly select task type
        task_type = self.rng.choice([
            "comprehension",
            "method_extraction",
            "contribution",
            "limitation",
        ])

        if task_type == "comprehension":
            return self._create_comprehension_task(title, abstract, paper)
        elif task_type == "method_extraction":
            return self._create_method_task(title, abstract, paper)
        elif task_type == "contribution":
            return self._create_contribution_task(title, abstract, paper)
        else:
            return self._create_limitation_task(title, abstract, paper)

    def _create_comprehension_task(
        self, title: str, abstract: str, paper: dict
    ) -> Task:
        """Create reading comprehension task."""
        questions = [
            "What is the main problem or challenge addressed in this paper?",
            "What approach or method does this paper propose?",
            "What type of evaluation or experiments are mentioned?",
            "What domain or field does this research belong to?",
        ]

        selected_q = self.rng.choice(questions)

        prompt = f"""Read the following research paper abstract:

Title: {title}

Abstract:
{abstract}

Question: {selected_q}

Provide a concise answer (2-3 sentences) based only on information in the abstract."""

        return Task(
            source=TaskSource.ARXIV_ABSTRACT,
            source_metadata={"paper_id": paper.get("id"), "task_type": "comprehension"},
            prompt=prompt,
            context=abstract,
            format=TaskFormat.FREE_TEXT,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.KNOWLEDGE_INTEGRATION,
            ],
            difficulty=DifficultyLevel.MEDIUM,
        )

    def _create_method_task(self, title: str, abstract: str, paper: dict) -> Task:
        """Create method extraction task."""
        prompt = f"""Analyze this research paper abstract:

Title: {title}

Abstract:
{abstract}

Extract and list:
1. The main technical approach/method (1-2 sentences)
2. Key algorithmic or architectural components mentioned
3. Any baselines or comparisons referenced
4. Datasets or benchmarks mentioned (if any)

Be specific and use terminology from the abstract."""

        return Task(
            source=TaskSource.ARXIV_ABSTRACT,
            source_metadata={"paper_id": paper.get("id"), "task_type": "method"},
            prompt=prompt,
            context=abstract,
            format=TaskFormat.STRUCTURED_JSON,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.KNOWLEDGE_INTEGRATION,
                Skill.STRUCTURED_OUTPUT,
            ],
            difficulty=DifficultyLevel.HARD,
        )

    def _create_contribution_task(self, title: str, abstract: str, paper: dict) -> Task:
        """Create contribution summary task."""
        prompt = f"""Read this research paper abstract:

Title: {title}

Abstract:
{abstract}

Identify and explain the main contributions of this paper:
1. What is novel about this work? (distinguish from prior work)
2. What is the key technical insight?
3. What are the claimed results or improvements?

Focus on what makes this paper different from existing work."""

        return Task(
            source=TaskSource.ARXIV_ABSTRACT,
            source_metadata={"paper_id": paper.get("id"), "task_type": "contribution"},
            prompt=prompt,
            context=abstract,
            format=TaskFormat.FREE_TEXT,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.ANALOGICAL_REASONING,
                Skill.KNOWLEDGE_INTEGRATION,
            ],
            difficulty=DifficultyLevel.HARD,
        )

    def _create_limitation_task(self, title: str, abstract: str, paper: dict) -> Task:
        """Create limitation inference task."""
        prompt = f"""Read this research paper abstract:

Title: {title}

Abstract:
{abstract}

Based on the abstract, infer potential limitations of this work:
1. What scenarios might this approach not work well for?
2. What assumptions might limit generalizability?
3. What additional experiments would strengthen the claims?

Note: The abstract may not explicitly state limitations - use your understanding to infer reasonable concerns."""

        return Task(
            source=TaskSource.ARXIV_ABSTRACT,
            source_metadata={"paper_id": paper.get("id"), "task_type": "limitation"},
            prompt=prompt,
            context=abstract,
            format=TaskFormat.FREE_TEXT,
            required_skills=[
                Skill.READING_COMPREHENSION,
                Skill.COUNTERFACTUAL_REASONING,
                Skill.CAUSAL_INFERENCE,
            ],
            difficulty=DifficultyLevel.VERY_HARD,
        )

    def _generate_synthetic_arxiv_tasks(self, n_tasks: int) -> Iterator[Task]:
        """Generate synthetic ArXiv-like tasks for demo mode."""
        abstracts = [
            {
                "title": "Scaling Laws for Neural Language Models",
                "abstract": """We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.""",
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "title": "Constitutional AI: Harmlessness from AI Feedback",
                "abstract": """As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a set of principles, or a 'constitution', which is used by the AI system to reason about and correct its own behavior. We find that a model trained in this way can be instructed to critique its own responses and amend them according to constitutional principles, allowing for rapid self-improvement. We show that this 'Constitutional AI' (CAI) approach can train less harmful AI systems using a fraction of the human feedback that would otherwise be required. We discuss the robustness of this technique and the potential role it can play in developing AI systems that remain beneficial as they become more capable.""",
                "difficulty": DifficultyLevel.HARD,
            },
            {
                "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                "abstract": """We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain-of-thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.""",
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "abstract": """Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.""",
                "difficulty": DifficultyLevel.HARD,
            },
            {
                "title": "Training Compute-Optimal Large Language Models",
                "abstract": """We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4x more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks.""",
                "difficulty": DifficultyLevel.HARD,
            },
        ]

        task_types = ["comprehension", "method", "contribution", "limitation"]

        for i in range(n_tasks):
            paper = self.rng.choice(abstracts)
            task_type = self.rng.choice(task_types)

            if task_type == "comprehension":
                task = self._create_comprehension_task(
                    paper["title"], paper["abstract"], {"id": f"synthetic_{i}"}
                )
            elif task_type == "method":
                task = self._create_method_task(
                    paper["title"], paper["abstract"], {"id": f"synthetic_{i}"}
                )
            elif task_type == "contribution":
                task = self._create_contribution_task(
                    paper["title"], paper["abstract"], {"id": f"synthetic_{i}"}
                )
            else:
                task = self._create_limitation_task(
                    paper["title"], paper["abstract"], {"id": f"synthetic_{i}"}
                )

            # Adjust difficulty based on paper
            task = Task(
                **{**task.model_dump(), "difficulty": paper["difficulty"]}
            )

            yield task

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()
