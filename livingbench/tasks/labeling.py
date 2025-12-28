"""
Automatic task labeling for skills and difficulty.

Provides:
- Skill detection from task content
- Difficulty estimation
- Validation of human-assigned labels

These labelers use heuristics and can optionally be augmented
with LLM-based classification for higher accuracy.
"""

from __future__ import annotations

import re
from typing import Any

from livingbench.core.types import Task, Skill, DifficultyLevel


class SkillLabeler:
    """
    Automatically detect required skills from task content.

    Uses keyword matching and structural analysis to identify
    which cognitive skills are needed to complete a task.
    """

    # Keyword patterns for each skill
    SKILL_PATTERNS: dict[Skill, list[str]] = {
        Skill.LOGICAL_DEDUCTION: [
            r"\bif\b.*\bthen\b",
            r"\btherefore\b",
            r"\bimplies\b",
            r"\bconclud",
            r"\bfollow[s]?\sfrom\b",
            r"\ball\s+\w+\s+are\b",
            r"\bno\s+\w+\s+are\b",
        ],
        Skill.MATHEMATICAL_REASONING: [
            r"\bcalculat",
            r"\bcompute\b",
            r"\bsolve\b",
            r"\bequation\b",
            r"\bformula\b",
            r"\d+\s*[\+\-\*\/\^\%]\s*\d+",
            r"\bprobability\b",
            r"\bstatistic",
        ],
        Skill.CAUSAL_INFERENCE: [
            r"\bbecause\b",
            r"\bcaused?\s+by\b",
            r"\bresult[s]?\s+in\b",
            r"\bwhy\b",
            r"\beffect\b",
            r"\bconsequen",
        ],
        Skill.COUNTERFACTUAL_REASONING: [
            r"\bif\s+.*\bhad\b",
            r"\bwould\s+have\b",
            r"\bwhat\s+if\b",
            r"\binstead\b",
            r"\balternativ",
            r"\bhypothetical",
        ],
        Skill.READING_COMPREHENSION: [
            r"\bread\b.*\b(text|passage|abstract)\b",
            r"\baccording\s+to\b",
            r"\bbased\s+on\b.*\b(text|passage|context)\b",
            r"\bsummariz",
            r"\bextract\b",
        ],
        Skill.CODE_GENERATION: [
            r"\bwrite\b.*\b(code|function|script|program)\b",
            r"\bimplement\b",
            r"\bcreate\s+a\s+(function|class|method)\b",
            r"\bpython\b",
            r"\bjavascript\b",
        ],
        Skill.CODE_UNDERSTANDING: [
            r"\bwhat\s+does\s+this\s+code\b",
            r"\bexplain\b.*\bcode\b",
            r"\bdebug\b",
            r"\bbug\b",
            r"\bfix\b.*\b(code|error|issue)\b",
            r"```\w*\n",  # Code block
        ],
        Skill.MULTI_STEP_PLANNING: [
            r"\bstep[s]?\b",
            r"\bfirst\b.*\bthen\b",
            r"\bplan\b",
            r"\bsequence\b",
            r"\border\b.*\boperations\b",
        ],
        Skill.TOOL_SELECTION: [
            r"\btool[s]?\b",
            r"\bfunction[s]?\s+available\b",
            r"\buse\s+(the|a)\s+\w+\s+to\b",
            r"\bapi\b",
        ],
        Skill.TOOL_COMPOSITION: [
            r"\bmultiple\s+tool[s]?\b",
            r"\bcombine\b",
            r"\bchain\b.*\b(tool|function)\b",
            r"\bsequence\s+of\s+(tool|function)\b",
        ],
        Skill.UNCERTAINTY_CALIBRATION: [
            r"\bconfiden",
            r"\buncertain",
            r"\bprobab",
            r"\bhow\s+sure\b",
            r"\bmight\s+be\b",
        ],
        Skill.ABSTENTION: [
            r"\bif\s+you\s+(don't|cannot|can't)\s+know\b",
            r"\bonly\s+if\b.*\bsure\b",
            r"\brefuse\b",
            r"\bdecline\b",
        ],
    }

    def label(self, task: Task) -> list[Skill]:
        """
        Detect skills required for a task.

        Returns list of skills in order of confidence.
        """
        text = f"{task.prompt} {task.context or ''}"
        text = text.lower()

        detected_skills: dict[Skill, int] = {}

        for skill, patterns in self.SKILL_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches)

            if score > 0:
                detected_skills[skill] = score

        # Sort by score and return
        sorted_skills = sorted(
            detected_skills.keys(),
            key=lambda s: detected_skills[s],
            reverse=True
        )

        # Always include at least reading comprehension as baseline
        if not sorted_skills:
            sorted_skills = [Skill.READING_COMPREHENSION]

        return sorted_skills

    def validate_labels(self, task: Task) -> dict[str, Any]:
        """
        Validate task's assigned skill labels against detected skills.

        Returns a report of agreement/disagreement.
        """
        detected = set(self.label(task))
        assigned = set(task.required_skills)

        return {
            "assigned_skills": [s.value for s in assigned],
            "detected_skills": [s.value for s in detected],
            "agreement": len(detected & assigned),
            "missing_in_assigned": [s.value for s in detected - assigned],
            "extra_in_assigned": [s.value for s in assigned - detected],
            "jaccard_similarity": len(detected & assigned) / len(detected | assigned)
            if detected | assigned else 1.0,
        }


class DifficultyEstimator:
    """
    Estimate task difficulty based on structural features.

    Features considered:
    - Text length and complexity
    - Number of steps required
    - Presence of adversarial elements
    - Domain specificity
    """

    def estimate(self, task: Task) -> DifficultyLevel:
        """
        Estimate difficulty level for a task.

        Returns estimated DifficultyLevel.
        """
        score = 0.0
        reasons = []

        text = f"{task.prompt} {task.context or ''}"

        # Length complexity
        word_count = len(text.split())
        if word_count > 500:
            score += 2.0
            reasons.append("long_text")
        elif word_count > 200:
            score += 1.0
            reasons.append("medium_text")

        # Multi-step indicators
        step_patterns = [
            r"\bfirst\b.*\bthen\b.*\bfinally\b",
            r"\bstep\s*[123456789]",
            r"\b1\.\s.*\b2\.\s.*\b3\.\s",
        ]
        for pattern in step_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                score += 1.5
                reasons.append("multi_step")
                break

        # Technical complexity
        technical_terms = [
            r"\balgorithm\b", r"\brecursion\b", r"\bO\([^)]+\)",
            r"\bderivative\b", r"\bintegral\b", r"\bmatrix\b",
            r"\basymptotic\b", r"\bNP-hard\b",
        ]
        tech_count = sum(
            1 for p in technical_terms
            if re.search(p, text, re.IGNORECASE)
        )
        score += tech_count * 0.5
        if tech_count > 0:
            reasons.append("technical_terms")

        # Negation and exceptions (adds complexity)
        negation_patterns = [
            r"\bexcept\b", r"\bunless\b", r"\bhowever\b",
            r"\bbut\s+not\b", r"\bnot\s+all\b",
        ]
        neg_count = sum(
            1 for p in negation_patterns
            if re.search(p, text, re.IGNORECASE)
        )
        score += neg_count * 0.5
        if neg_count > 0:
            reasons.append("negation_complexity")

        # Skill count
        n_skills = len(task.required_skills)
        if n_skills >= 4:
            score += 2.0
            reasons.append("many_skills")
        elif n_skills >= 2:
            score += 1.0
            reasons.append("multiple_skills")

        # Code presence (usually harder)
        if "```" in text:
            score += 1.0
            reasons.append("code_present")

        # Map score to difficulty
        if score < 1.5:
            return DifficultyLevel.EASY
        elif score < 3.0:
            return DifficultyLevel.MEDIUM
        elif score < 5.0:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.VERY_HARD

    def validate_difficulty(self, task: Task) -> dict[str, Any]:
        """
        Validate task's assigned difficulty against estimated difficulty.
        """
        estimated = self.estimate(task)
        assigned = task.difficulty

        difficulty_order = [
            DifficultyLevel.TRIVIAL,
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
            DifficultyLevel.VERY_HARD,
            DifficultyLevel.ADVERSARIAL,
        ]

        assigned_idx = difficulty_order.index(assigned)
        estimated_idx = difficulty_order.index(estimated)
        difference = abs(assigned_idx - estimated_idx)

        return {
            "assigned": assigned.value,
            "estimated": estimated.value,
            "match": assigned == estimated,
            "difference": difference,
            "assessment": "accurate" if difference <= 1 else "needs_review",
        }


class TaskValidator:
    """
    Comprehensive task validation.

    Checks:
    - Required fields are present
    - Labels are reasonable
    - No obvious issues
    """

    def __init__(self):
        self.skill_labeler = SkillLabeler()
        self.difficulty_estimator = DifficultyEstimator()

    def validate(self, task: Task) -> dict[str, Any]:
        """
        Perform comprehensive validation on a task.
        """
        issues = []
        warnings = []

        # Basic checks
        if not task.prompt or len(task.prompt.strip()) < 10:
            issues.append("prompt_too_short")

        if not task.required_skills:
            issues.append("no_skills_assigned")

        # Skill validation
        skill_report = self.skill_labeler.validate_labels(task)
        if skill_report["jaccard_similarity"] < 0.3:
            warnings.append("skill_labels_may_be_inaccurate")

        # Difficulty validation
        diff_report = self.difficulty_estimator.validate_difficulty(task)
        if diff_report["difference"] > 1:
            warnings.append("difficulty_may_be_miscalibrated")

        # Format-specific checks
        if task.format == "multiple_choice" and not task.choices:
            issues.append("multiple_choice_missing_choices")

        if task.format == "tool_calls" and not task.available_tools:
            issues.append("tool_task_missing_tools")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "skill_validation": skill_report,
            "difficulty_validation": diff_report,
        }
