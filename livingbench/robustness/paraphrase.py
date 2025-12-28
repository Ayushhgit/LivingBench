"""
Paraphrase Generation for Robustness Testing.

Generates semantically equivalent variants of tasks to test:
- Robustness to surface form changes
- Reliance on specific phrasings
- Memorization of exact prompts

A robust model should give consistent answers across paraphrases.
"""

from __future__ import annotations

import random
import re
from typing import Iterator
from uuid import uuid4

from livingbench.core.types import Task, TaskSource


class ParaphraseGenerator:
    """
    Generate paraphrased variants of tasks.

    Strategies:
    1. Synonym substitution
    2. Sentence reordering
    3. Voice change (active/passive)
    4. Question reformulation
    5. Instruction rephrasing

    The key constraint: semantic meaning must be preserved.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Synonym mappings (simplified)
        self.synonyms = {
            "calculate": ["compute", "determine", "find", "work out"],
            "what": ["which", "what exactly"],
            "explain": ["describe", "elaborate on", "clarify"],
            "analyze": ["examine", "investigate", "study"],
            "find": ["locate", "identify", "discover"],
            "show": ["demonstrate", "illustrate", "display"],
            "give": ["provide", "supply", "offer"],
            "use": ["utilize", "employ", "apply"],
            "make": ["create", "produce", "generate"],
            "help": ["assist", "aid", "support"],
            "first": ["initially", "to begin with", "firstly"],
            "then": ["next", "subsequently", "after that"],
            "also": ["additionally", "furthermore", "moreover"],
            "however": ["nevertheless", "nonetheless", "but"],
            "because": ["since", "as", "due to the fact that"],
            "important": ["significant", "crucial", "essential"],
            "problem": ["issue", "challenge", "difficulty"],
            "answer": ["response", "solution", "reply"],
            "question": ["query", "inquiry"],
            "result": ["outcome", "consequence"],
        }

        # Question reformulation templates
        self.question_templates = [
            ("What is", ["Determine", "Find", "Calculate", "Identify"]),
            ("How do", ["In what way do", "By what means do"]),
            ("Why does", ["What causes", "What explains why"]),
            ("Can you", ["Would you be able to", "Is it possible to"]),
            ("Please", ["Kindly", "Would you", "I'd like you to"]),
        ]

    def generate_paraphrases(
        self,
        task: Task,
        n_paraphrases: int = 3,
    ) -> list[Task]:
        """
        Generate n paraphrased variants of a task.

        Args:
            task: Original task
            n_paraphrases: Number of variants to generate

        Returns:
            List of paraphrased task variants
        """
        paraphrases = []
        strategies = [
            self._synonym_substitution,
            self._question_reformulation,
            self._instruction_variation,
            self._sentence_reorder,
        ]

        for i in range(n_paraphrases):
            # Select strategy (different strategy each time if possible)
            strategy = strategies[i % len(strategies)]

            # Generate paraphrased prompt
            new_prompt = strategy(task.prompt)

            # Create variant task
            variant = self._create_variant(task, new_prompt, f"paraphrase_{i}")
            paraphrases.append(variant)

        return paraphrases

    def _synonym_substitution(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        new_words = []

        for word in words:
            lower_word = word.lower().strip(".,!?;:")
            if lower_word in self.synonyms and self.rng.random() < 0.3:
                replacement = self.rng.choice(self.synonyms[lower_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                if word[-1] in ".,!?;:":
                    replacement += word[-1]
                new_words.append(replacement)
            else:
                new_words.append(word)

        return " ".join(new_words)

    def _question_reformulation(self, text: str) -> str:
        """Reformulate questions."""
        result = text

        for pattern, replacements in self.question_templates:
            if pattern.lower() in text.lower():
                replacement = self.rng.choice(replacements)
                result = re.sub(
                    pattern,
                    replacement,
                    result,
                    count=1,
                    flags=re.IGNORECASE,
                )
                break

        return result

    def _instruction_variation(self, text: str) -> str:
        """Vary instruction phrasing."""
        variations = [
            (r"Answer the following:", "Respond to this:"),
            (r"Consider the following", "Look at the following"),
            (r"Given that", "Knowing that"),
            (r"Based on", "Using"),
            (r"In order to", "To"),
            (r"You need to", "Your task is to"),
            (r"Make sure to", "Ensure that you"),
        ]

        result = text
        for pattern, replacement in variations:
            if re.search(pattern, text, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                break

        return result

    def _sentence_reorder(self, text: str) -> str:
        """Reorder sentences where semantically safe."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) < 3:
            return text

        # Find context/instruction boundary
        # Usually first sentence is context, rest is instruction or vice versa
        # Only reorder if safe (no dependencies)

        # Simple heuristic: if first sentence contains "Given" or "Consider",
        # it's context and should stay first
        first_is_context = any(
            kw in sentences[0].lower()
            for kw in ["given", "consider", "suppose", "assume", "let"]
        )

        if first_is_context and len(sentences) >= 3:
            # Keep first sentence, shuffle middle sentences
            middle = sentences[1:-1]
            self.rng.shuffle(middle)
            return " ".join([sentences[0]] + middle + [sentences[-1]])

        return text

    def _create_variant(
        self,
        original: Task,
        new_prompt: str,
        variant_type: str,
    ) -> Task:
        """Create a variant task from original."""
        return Task(
            id=uuid4(),
            version=original.version,
            source=original.source,
            source_metadata={
                **original.source_metadata,
                "variant_of": str(original.id),
            },
            prompt=new_prompt,
            context=original.context,
            format=original.format,
            choices=original.choices,
            reference_answer=original.reference_answer,
            reference_explanation=original.reference_explanation,
            available_tools=original.available_tools,
            expected_tool_calls=original.expected_tool_calls,
            required_skills=original.required_skills,
            difficulty=original.difficulty,
            is_variant=True,
            parent_task_id=original.id,
            variant_type=variant_type,
        )


class ParaphraseConsistencyChecker:
    """
    Check consistency of responses across paraphrases.

    A consistent model should give semantically equivalent answers
    to paraphrased questions.
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, require exact match. If False, use fuzzy matching.
        """
        self.strict = strict

    def check_consistency(
        self,
        original_answer: str,
        paraphrase_answers: list[str],
    ) -> dict:
        """
        Check if answers are consistent across paraphrases.

        Returns:
            Dictionary with consistency metrics
        """
        if not paraphrase_answers:
            return {"consistency": 1.0, "n_variants": 0}

        if self.strict:
            matches = sum(
                1 for ans in paraphrase_answers
                if self._normalize(ans) == self._normalize(original_answer)
            )
        else:
            matches = sum(
                1 for ans in paraphrase_answers
                if self._fuzzy_match(original_answer, ans)
            )

        consistency = matches / len(paraphrase_answers)

        return {
            "consistency": consistency,
            "n_variants": len(paraphrase_answers),
            "n_matches": matches,
            "inconsistent_answers": [
                ans for ans in paraphrase_answers
                if not self._fuzzy_match(original_answer, ans)
            ],
        }

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, remove punctuation, normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def _fuzzy_match(self, text1: str, text2: str) -> bool:
        """Check if two texts are semantically similar."""
        norm1 = self._normalize(text1)
        norm2 = self._normalize(text2)

        # Exact match after normalization
        if norm1 == norm2:
            return True

        # Check for numeric equivalence
        nums1 = re.findall(r'\d+\.?\d*', norm1)
        nums2 = re.findall(r'\d+\.?\d*', norm2)
        if nums1 and nums2 and nums1 == nums2:
            return True

        # Word overlap check (Jaccard similarity > 0.8)
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if words1 and words2:
            jaccard = len(words1 & words2) / len(words1 | words2)
            if jaccard > 0.8:
                return True

        return False
