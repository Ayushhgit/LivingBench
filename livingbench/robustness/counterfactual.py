"""
Counterfactual Variant Generation.

Generates tasks with modified premises that should change the answer.
This tests whether models:

1. Actually reason from premises (vs. pattern matching)
2. Are sensitive to relevant changes
3. Ignore irrelevant distractors

A good model should:
- Change answer when relevant premises change
- Keep answer when irrelevant details change
"""

from __future__ import annotations

import random
import re
from uuid import uuid4

from livingbench.core.types import Task, Skill


class CounterfactualGenerator:
    """
    Generate counterfactual variants of tasks.

    Types of counterfactuals:
    1. Premise negation: Negate key premises
    2. Value perturbation: Change numbers/quantities
    3. Entity substitution: Replace key entities
    4. Irrelevant addition: Add irrelevant information (should not change answer)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_counterfactuals(
        self,
        task: Task,
        n_counterfactuals: int = 2,
    ) -> list[tuple[Task, str]]:
        """
        Generate counterfactual variants.

        Args:
            task: Original task
            n_counterfactuals: Number of variants to generate

        Returns:
            List of (variant_task, expected_change) tuples
            expected_change is "should_change" or "should_not_change"
        """
        counterfactuals = []

        # Try different strategies
        strategies = [
            (self._negate_premise, "should_change"),
            (self._perturb_values, "should_change"),
            (self._add_irrelevant, "should_not_change"),
        ]

        for i in range(n_counterfactuals):
            strategy_fn, expected_change = strategies[i % len(strategies)]
            result = strategy_fn(task)

            if result:
                new_prompt, new_answer = result
                variant = self._create_variant(task, new_prompt, new_answer)
                counterfactuals.append((variant, expected_change))

        return counterfactuals

    def _negate_premise(self, task: Task) -> tuple[str, str | None] | None:
        """Negate a key premise in the task."""
        prompt = task.prompt

        # Patterns for negation
        negation_patterns = [
            (r"\bAll (\w+) are (\w+)", r"No \1 are \2"),
            (r"\bNo (\w+) are (\w+)", r"All \1 are \2"),
            (r"\bis true\b", "is false"),
            (r"\bis false\b", "is true"),
            (r"\bcan\b", "cannot"),
            (r"\bcannot\b", "can"),
            (r"\bwill\b", "will not"),
            (r"\bwill not\b", "will"),
            (r"\bmust\b", "must not"),
            (r"\balways\b", "never"),
            (r"\bnever\b", "always"),
        ]

        for pattern, replacement in negation_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                new_prompt = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
                # Answer likely changes but we can't determine exact new answer
                # without deeper semantic understanding
                return new_prompt, None

        return None

    def _perturb_values(self, task: Task) -> tuple[str, str | None] | None:
        """Perturb numerical values in the task."""
        prompt = task.prompt

        # Find numbers in the prompt
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', prompt)

        if not numbers:
            return None

        # Select a number to perturb
        target = self.rng.choice(numbers)
        target_val = float(target)

        # Perturb by a meaningful amount
        if target_val == 0:
            new_val = self.rng.choice([1, 5, 10])
        elif target_val < 10:
            new_val = target_val + self.rng.choice([-2, -1, 1, 2])
        else:
            # Percentage change
            factor = self.rng.choice([0.5, 0.75, 1.25, 1.5, 2.0])
            new_val = target_val * factor

        # Format appropriately
        if '.' in target:
            new_str = f"{new_val:.2f}"
        else:
            new_str = str(int(new_val))

        new_prompt = prompt.replace(target, new_str, 1)

        return new_prompt, None

    def _add_irrelevant(self, task: Task) -> tuple[str, str | None] | None:
        """Add irrelevant information that shouldn't affect the answer."""
        prompt = task.prompt

        irrelevant_additions = [
            "Note: This problem was inspired by a conversation last Tuesday.",
            "(The numbers in this problem are all prime, which is interesting but not relevant to the solution.)",
            "Fun fact: This type of problem was first studied in the 18th century.",
            "By the way, the problem setter's favorite color is blue.",
            "Interestingly, all the names used in this problem start with vowels.",
        ]

        addition = self.rng.choice(irrelevant_additions)

        # Find a good place to insert
        sentences = prompt.split('. ')
        if len(sentences) > 1:
            # Insert after first sentence
            insert_pos = 1
            sentences.insert(insert_pos, addition)
            new_prompt = '. '.join(sentences)
        else:
            # Prepend
            new_prompt = addition + " " + prompt

        # Answer should NOT change
        return new_prompt, task.reference_answer

    def _create_variant(
        self,
        original: Task,
        new_prompt: str,
        new_answer: str | None,
    ) -> Task:
        """Create a counterfactual variant task."""
        return Task(
            id=uuid4(),
            version=original.version,
            source=original.source,
            source_metadata={
                **original.source_metadata,
                "counterfactual_of": str(original.id),
            },
            prompt=new_prompt,
            context=original.context,
            format=original.format,
            choices=original.choices,
            reference_answer=new_answer,
            reference_explanation=None,
            available_tools=original.available_tools,
            expected_tool_calls=original.expected_tool_calls,
            required_skills=original.required_skills + [Skill.COUNTERFACTUAL_REASONING],
            difficulty=original.difficulty,
            is_variant=True,
            parent_task_id=original.id,
            variant_type="counterfactual",
        )


class CounterfactualSensitivityChecker:
    """
    Check model sensitivity to counterfactual changes.

    Measures:
    - Does model change answer when it should? (sensitivity)
    - Does model keep answer when it shouldn't change? (stability)
    """

    def check_sensitivity(
        self,
        original_answer: str,
        counterfactual_answers: list[tuple[str, str]],  # (answer, expected_change)
    ) -> dict:
        """
        Check counterfactual sensitivity.

        Args:
            original_answer: Answer to original task
            counterfactual_answers: List of (answer, "should_change"|"should_not_change")

        Returns:
            Sensitivity metrics
        """
        should_change = []
        should_not_change = []

        for answer, expected in counterfactual_answers:
            did_change = self._answers_differ(original_answer, answer)

            if expected == "should_change":
                should_change.append(did_change)
            else:
                should_not_change.append(not did_change)

        sensitivity = sum(should_change) / len(should_change) if should_change else 1.0
        stability = sum(should_not_change) / len(should_not_change) if should_not_change else 1.0

        return {
            "sensitivity": sensitivity,  # Rate of correct changes
            "stability": stability,  # Rate of correct non-changes
            "n_should_change": len(should_change),
            "n_should_not_change": len(should_not_change),
        }

    def _answers_differ(self, ans1: str, ans2: str) -> bool:
        """Check if two answers are meaningfully different."""
        # Normalize
        norm1 = ans1.lower().strip()
        norm2 = ans2.lower().strip()

        if norm1 == norm2:
            return False

        # Check numeric equivalence
        nums1 = re.findall(r'\d+\.?\d*', norm1)
        nums2 = re.findall(r'\d+\.?\d*', norm2)

        if nums1 and nums2 and nums1 == nums2:
            return False

        return True
