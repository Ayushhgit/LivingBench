"""
Synthetic task generators for controlled evaluation.

These generators create tasks with known ground truth, enabling:
- Precise measurement of specific capabilities
- Controlled difficulty scaling
- Reproducible evaluation
- No external dependencies

The synthetic tasks are designed to:
1. Test specific cognitive skills in isolation
2. Scale difficulty systematically
3. Resist memorization (parameterized generation)
4. Have unambiguous ground truth
"""

from __future__ import annotations

import random
from typing import Iterator
from itertools import product

from livingbench.core.types import (
    Task,
    TaskSource,
    TaskFormat,
    Skill,
    DifficultyLevel,
)
from livingbench.tasks.base import TaskGeneratorBase
from livingbench.core.registry import task_generators


@task_generators.register("synthetic_reasoning")
class SyntheticReasoningGenerator(TaskGeneratorBase):
    """
    Generate synthetic logical reasoning tasks.

    Task types include:
    - Syllogistic reasoning (All A are B, All B are C, ...)
    - Constraint satisfaction (scheduling, seating, etc.)
    - Propositional logic (if-then, and, or, not)
    - Relational reasoning (taller than, older than, etc.)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._template_bank = self._build_template_bank()

    @property
    def source(self) -> TaskSource:
        return TaskSource.SYNTHETIC_REASONING

    def generate(self, n_tasks: int, **kwargs) -> Iterator[Task]:
        """Generate n reasoning tasks with varying types and difficulties."""
        difficulty_weights = kwargs.get("difficulty_weights", {
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.3,
            DifficultyLevel.VERY_HARD: 0.1,
        })

        difficulties = list(difficulty_weights.keys())
        weights = list(difficulty_weights.values())

        task_types = ["syllogism", "constraint", "propositional", "relational"]

        for _ in range(n_tasks):
            difficulty = self.rng.choices(difficulties, weights=weights)[0]
            task_type = self.rng.choice(task_types)

            if task_type == "syllogism":
                task = self._generate_syllogism(difficulty)
            elif task_type == "constraint":
                task = self._generate_constraint_task(difficulty)
            elif task_type == "propositional":
                task = self._generate_propositional(difficulty)
            else:
                task = self._generate_relational(difficulty)

            if self.validate_task(task):
                yield task

    def _generate_syllogism(self, difficulty: DifficultyLevel) -> Task:
        """Generate syllogistic reasoning task."""
        # Define entities and properties
        entities = ["professors", "scientists", "artists", "engineers", "doctors"]
        properties = ["creative", "logical", "curious", "methodical", "patient"]

        self.rng.shuffle(entities)
        self.rng.shuffle(properties)

        # Difficulty determines chain length and negations
        if difficulty == DifficultyLevel.EASY:
            chain_length = 2
            use_negation = False
        elif difficulty == DifficultyLevel.MEDIUM:
            chain_length = 3
            use_negation = False
        elif difficulty == DifficultyLevel.HARD:
            chain_length = 3
            use_negation = True
        else:
            chain_length = 4
            use_negation = True

        # Build syllogistic chain
        premises = []
        for i in range(chain_length):
            if use_negation and i == chain_length - 2:
                premises.append(f"No {entities[i]} are {properties[i]}.")
            else:
                premises.append(f"All {entities[i]} are {properties[i]}.")

            if i < chain_length - 1:
                premises.append(f"All things that are {properties[i]} are {entities[i+1]}.")

        # Generate question
        first_entity = entities[0]
        last_property = properties[chain_length - 1]

        # Compute answer
        if use_negation:
            answer = "No" if chain_length >= 3 else "Yes"
            explanation = "The chain contains a negative premise that breaks the positive inference."
        else:
            answer = "Yes"
            explanation = f"Through transitive inference: {' -> '.join(entities[:chain_length])}"

        prompt = f"""Consider the following premises:

{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(premises))}

Question: Are all {first_entity} {last_property}?

Answer with only 'Yes' or 'No', then explain your reasoning."""

        return Task(
            source=TaskSource.SYNTHETIC_REASONING,
            source_metadata={"task_type": "syllogism", "chain_length": chain_length},
            prompt=prompt,
            format=TaskFormat.FREE_TEXT,
            reference_answer=answer,
            reference_explanation=explanation,
            required_skills=[Skill.LOGICAL_DEDUCTION, Skill.MULTI_STEP_PLANNING],
            difficulty=difficulty,
        )

    def _generate_constraint_task(self, difficulty: DifficultyLevel) -> Task:
        """Generate constraint satisfaction task (scheduling/arrangement)."""
        names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
        positions = ["first", "second", "third", "fourth", "fifth", "sixth"]

        # Difficulty determines number of people and constraints
        if difficulty == DifficultyLevel.EASY:
            n_people = 3
            n_constraints = 2
        elif difficulty == DifficultyLevel.MEDIUM:
            n_people = 4
            n_constraints = 3
        elif difficulty == DifficultyLevel.HARD:
            n_people = 5
            n_constraints = 5
        else:
            n_people = 6
            n_constraints = 7

        selected_names = names[:n_people]
        selected_positions = positions[:n_people]

        # Generate a valid arrangement first
        arrangement = list(selected_names)
        self.rng.shuffle(arrangement)

        # Generate constraints that are satisfied by this arrangement
        constraints = []
        constraint_types = [
            "before", "after", "adjacent", "not_adjacent", "position"
        ]

        for _ in range(n_constraints):
            c_type = self.rng.choice(constraint_types)
            p1, p2 = self.rng.sample(range(n_people), 2)

            if c_type == "before" and p1 < p2:
                constraints.append(f"{arrangement[p1]} must be before {arrangement[p2]}.")
            elif c_type == "after" and p1 > p2:
                constraints.append(f"{arrangement[p1]} must be after {arrangement[p2]}.")
            elif c_type == "adjacent" and abs(p1 - p2) == 1:
                constraints.append(f"{arrangement[p1]} must be adjacent to {arrangement[p2]}.")
            elif c_type == "not_adjacent" and abs(p1 - p2) > 1:
                constraints.append(f"{arrangement[p1]} must not be adjacent to {arrangement[p2]}.")
            elif c_type == "position":
                constraints.append(f"{arrangement[p1]} must be in the {selected_positions[p1]} position.")

        # Ensure we have enough constraints
        while len(constraints) < n_constraints:
            p = self.rng.randrange(n_people)
            constraints.append(f"{arrangement[p]} must be in the {selected_positions[p]} position.")

        constraints = constraints[:n_constraints]

        # Ask about a specific position
        query_pos = self.rng.randrange(n_people)
        question = f"Who is in the {selected_positions[query_pos]} position?"
        answer = arrangement[query_pos]

        prompt = f"""Arrange {n_people} people in a line based on these constraints:

{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(constraints))}

{question}"""

        return Task(
            source=TaskSource.SYNTHETIC_REASONING,
            source_metadata={"task_type": "constraint", "n_people": n_people},
            prompt=prompt,
            format=TaskFormat.FREE_TEXT,
            reference_answer=answer,
            reference_explanation=f"Valid arrangement: {' -> '.join(arrangement)}",
            required_skills=[
                Skill.LOGICAL_DEDUCTION,
                Skill.MULTI_STEP_PLANNING,
                Skill.COUNTERFACTUAL_REASONING,
            ],
            difficulty=difficulty,
        )

    def _generate_propositional(self, difficulty: DifficultyLevel) -> Task:
        """Generate propositional logic task."""
        variables = ["P", "Q", "R", "S", "T"]
        meanings = [
            "it is raining",
            "the ground is wet",
            "the game is cancelled",
            "people stay home",
            "traffic is light",
        ]

        if difficulty == DifficultyLevel.EASY:
            n_vars = 2
            n_premises = 2
        elif difficulty == DifficultyLevel.MEDIUM:
            n_vars = 3
            n_premises = 3
        elif difficulty == DifficultyLevel.HARD:
            n_vars = 4
            n_premises = 4
        else:
            n_vars = 5
            n_premises = 5

        selected_vars = variables[:n_vars]
        selected_meanings = meanings[:n_vars]

        # Create variable meanings
        var_map = {v: m for v, m in zip(selected_vars, selected_meanings)}

        # Generate truth assignment
        truth = {v: self.rng.choice([True, False]) for v in selected_vars}

        # Generate premises that are true under this assignment
        premises = []
        for i in range(n_premises):
            v1, v2 = self.rng.sample(selected_vars, 2)

            # Create implications that are true
            if truth[v1] and truth[v2]:
                premises.append(f"If {var_map[v1]}, then {var_map[v2]}.")
            elif not truth[v1]:
                premises.append(f"If {var_map[v1]}, then {var_map[v2]}.")
            elif truth[v1] and not truth[v2]:
                # This would make "if v1 then v2" false, so use negation
                premises.append(f"If {var_map[v1]}, then it is not the case that {var_map[v2]}.")

        # Add a fact
        true_vars = [v for v, t in truth.items() if t]
        if true_vars:
            fact_var = self.rng.choice(true_vars)
            premises.insert(0, f"It is the case that {var_map[fact_var]}.")

        # Query
        query_var = self.rng.choice(selected_vars)
        answer = "True" if truth[query_var] else "False"

        prompt = f"""Given the following statements:

{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(premises))}

Is it true that {var_map[query_var]}?

Answer with 'True' or 'False', then explain your reasoning."""

        return Task(
            source=TaskSource.SYNTHETIC_REASONING,
            source_metadata={"task_type": "propositional", "n_vars": n_vars},
            prompt=prompt,
            format=TaskFormat.FREE_TEXT,
            reference_answer=answer,
            reference_explanation=f"Truth values: {truth}",
            required_skills=[
                Skill.LOGICAL_DEDUCTION,
                Skill.CAUSAL_INFERENCE,
            ],
            difficulty=difficulty,
        )

    def _generate_relational(self, difficulty: DifficultyLevel) -> Task:
        """Generate relational reasoning task."""
        names = ["Alex", "Blake", "Casey", "Drew", "Ellis", "Finley"]
        relations = [
            ("taller than", "shorter than", "height"),
            ("older than", "younger than", "age"),
            ("faster than", "slower than", "speed"),
            ("heavier than", "lighter than", "weight"),
        ]

        relation_pair = self.rng.choice(relations)
        rel_pos, rel_neg, attr = relation_pair

        if difficulty == DifficultyLevel.EASY:
            n_people = 3
        elif difficulty == DifficultyLevel.MEDIUM:
            n_people = 4
        elif difficulty == DifficultyLevel.HARD:
            n_people = 5
        else:
            n_people = 6

        selected = names[:n_people]

        # Create true ordering
        ordering = list(selected)
        self.rng.shuffle(ordering)

        # Generate statements based on ordering
        statements = []
        for i in range(n_people - 1):
            # Add direct comparisons
            if self.rng.random() < 0.5:
                statements.append(f"{ordering[i]} is {rel_pos} {ordering[i+1]}.")
            else:
                statements.append(f"{ordering[i+1]} is {rel_neg} {ordering[i]}.")

        # Add some transitive statements for harder difficulties
        if difficulty in [DifficultyLevel.HARD, DifficultyLevel.VERY_HARD]:
            for i in range(n_people - 2):
                if self.rng.random() < 0.3:
                    statements.append(f"{ordering[i]} is {rel_pos} {ordering[i+2]}.")

        self.rng.shuffle(statements)

        # Query
        query_type = self.rng.choice(["compare", "extreme"])

        if query_type == "compare":
            p1, p2 = self.rng.sample(range(n_people), 2)
            question = f"Is {ordering[p1]} {rel_pos} {ordering[p2]}?"
            answer = "Yes" if p1 < p2 else "No"
        else:
            extreme = self.rng.choice(["most", "least"])
            if extreme == "most":
                question = f"Who has the greatest {attr}?"
                answer = ordering[0]
            else:
                question = f"Who has the least {attr}?"
                answer = ordering[-1]

        prompt = f"""Consider the following facts about {n_people} people:

{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(statements))}

{question}"""

        return Task(
            source=TaskSource.SYNTHETIC_REASONING,
            source_metadata={"task_type": "relational", "relation": attr},
            prompt=prompt,
            format=TaskFormat.FREE_TEXT,
            reference_answer=answer,
            reference_explanation=f"Ordering from highest to lowest: {' > '.join(ordering)}",
            required_skills=[
                Skill.LOGICAL_DEDUCTION,
                Skill.MULTI_STEP_PLANNING,
            ],
            difficulty=difficulty,
        )

    def _build_template_bank(self) -> dict:
        """Build template bank for task generation."""
        return {
            "syllogism": {},
            "constraint": {},
            "propositional": {},
            "relational": {},
        }


@task_generators.register("synthetic_math")
class SyntheticMathGenerator(TaskGeneratorBase):
    """
    Generate synthetic mathematical reasoning tasks.

    Task types include:
    - Arithmetic word problems
    - Algebraic equations
    - Probability reasoning
    - Geometric reasoning
    - Number theory
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @property
    def source(self) -> TaskSource:
        return TaskSource.SYNTHETIC_MATH

    def generate(self, n_tasks: int, **kwargs) -> Iterator[Task]:
        """Generate n math tasks."""
        task_types = ["arithmetic", "algebra", "probability", "number_theory"]

        difficulty_weights = kwargs.get("difficulty_weights", {
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.3,
            DifficultyLevel.VERY_HARD: 0.1,
        })

        difficulties = list(difficulty_weights.keys())
        weights = list(difficulty_weights.values())

        for _ in range(n_tasks):
            difficulty = self.rng.choices(difficulties, weights=weights)[0]
            task_type = self.rng.choice(task_types)

            if task_type == "arithmetic":
                task = self._generate_arithmetic(difficulty)
            elif task_type == "algebra":
                task = self._generate_algebra(difficulty)
            elif task_type == "probability":
                task = self._generate_probability(difficulty)
            else:
                task = self._generate_number_theory(difficulty)

            if self.validate_task(task):
                yield task

    def _generate_arithmetic(self, difficulty: DifficultyLevel) -> Task:
        """Generate arithmetic word problem."""
        contexts = [
            ("apples", "bought", "gave away", "has"),
            ("books", "received", "lent", "owns"),
            ("dollars", "earned", "spent", "saved"),
            ("stickers", "collected", "traded", "has"),
        ]

        item, action1, action2, final = self.rng.choice(contexts)
        names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan"]
        name = self.rng.choice(names)

        if difficulty == DifficultyLevel.EASY:
            # Simple two-step problem
            start = self.rng.randint(10, 50)
            add = self.rng.randint(5, 20)
            sub = self.rng.randint(1, min(add, start + add - 1))
            answer = start + add - sub

            prompt = f"""{name} had {start} {item}. They {action1} {add} more {item}, then {action2} {sub} {item}.

How many {item} does {name} have now?

Provide only the numerical answer."""

        elif difficulty == DifficultyLevel.MEDIUM:
            # Multi-step with multiplication
            groups = self.rng.randint(3, 7)
            per_group = self.rng.randint(4, 12)
            subtract = self.rng.randint(1, groups * per_group // 2)
            answer = groups * per_group - subtract

            prompt = f"""{name} has {groups} boxes of {item}, with {per_group} {item} in each box. After giving away {subtract} {item}, how many {item} does {name} have?

Provide only the numerical answer."""

        elif difficulty == DifficultyLevel.HARD:
            # Fractions and ratios
            total = self.rng.randint(24, 120)
            # Ensure divisibility
            total = total - (total % 12)
            fraction1 = self.rng.choice([2, 3, 4, 6])
            fraction2 = self.rng.choice([2, 3, 4])

            first_part = total // fraction1
            remaining = total - first_part
            second_part = remaining // fraction2

            answer = total - first_part - second_part

            prompt = f"""{name} had {total} {item}. They gave 1/{fraction1} to their friend, then donated 1/{fraction2} of the remaining {item} to charity.

How many {item} does {name} have left?

Provide only the numerical answer."""

        else:
            # Complex multi-step with percentages
            initial = self.rng.randint(100, 500)
            percent1 = self.rng.choice([10, 15, 20, 25])
            add_amount = self.rng.randint(10, 50)
            percent2 = self.rng.choice([10, 20, 25, 50])

            after_first = initial * (100 - percent1) // 100
            after_add = after_first + add_amount
            final_answer = after_add * (100 - percent2) // 100
            answer = final_answer

            prompt = f"""{name} started with {initial} {item}. First, they lost {percent1}% of their {item}. Then they found {add_amount} more {item}. Finally, they gave away {percent2}% of what they had.

How many {item} does {name} have now?

Provide only the numerical answer."""

        return Task(
            source=TaskSource.SYNTHETIC_MATH,
            source_metadata={"task_type": "arithmetic", "difficulty": difficulty.value},
            prompt=prompt,
            format=TaskFormat.NUMERIC,
            reference_answer=str(answer),
            required_skills=[Skill.MATHEMATICAL_REASONING, Skill.READING_COMPREHENSION],
            difficulty=difficulty,
        )

    def _generate_algebra(self, difficulty: DifficultyLevel) -> Task:
        """Generate algebraic reasoning task."""
        if difficulty == DifficultyLevel.EASY:
            # Linear equation
            a = self.rng.randint(2, 10)
            x = self.rng.randint(1, 20)
            b = self.rng.randint(1, 50)
            result = a * x + b

            prompt = f"""Solve for x:
{a}x + {b} = {result}

Provide only the numerical answer."""
            answer = str(x)

        elif difficulty == DifficultyLevel.MEDIUM:
            # System of equations (simple)
            x = self.rng.randint(1, 10)
            y = self.rng.randint(1, 10)

            a1, b1 = self.rng.randint(1, 5), self.rng.randint(1, 5)
            a2, b2 = self.rng.randint(1, 5), self.rng.randint(1, 5)

            c1 = a1 * x + b1 * y
            c2 = a2 * x + b2 * y

            prompt = f"""Solve the system of equations:
{a1}x + {b1}y = {c1}
{a2}x + {b2}y = {c2}

What is x + y?

Provide only the numerical answer."""
            answer = str(x + y)

        elif difficulty == DifficultyLevel.HARD:
            # Quadratic
            r1 = self.rng.randint(-10, 10)
            r2 = self.rng.randint(-10, 10)

            # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
            b = -(r1 + r2)
            c = r1 * r2

            prompt = f"""Find all solutions to:
x² {"+" if b >= 0 else "-"} {abs(b)}x {"+" if c >= 0 else "-"} {abs(c)} = 0

Provide the sum of all solutions."""
            answer = str(r1 + r2)

        else:
            # Word problem requiring algebraic setup
            rate1 = self.rng.randint(40, 80)
            rate2 = self.rng.randint(50, 90)
            head_start = self.rng.randint(10, 30)

            # Meeting time: rate1 * t = rate2 * (t - head_start_time_equivalent)
            # When does B catch up if A starts ahead by distance?
            distance_head_start = head_start
            if rate2 > rate1:
                time_to_catch = distance_head_start / (rate2 - rate1)
                meeting_distance = rate2 * time_to_catch
                answer = str(int(meeting_distance))

                prompt = f"""Train A travels at {rate1} km/h. Train B travels at {rate2} km/h on a parallel track. Train A has a {head_start} km head start.

How far will Train B have traveled when it catches up to Train A?

Provide only the numerical answer (in km, rounded to nearest integer)."""
            else:
                prompt = f"""If x² + y² = {r1**2 + r2**2} and xy = {r1 * r2}, what is (x + y)²?

Provide only the numerical answer."""
                answer = str((r1 + r2) ** 2)

        return Task(
            source=TaskSource.SYNTHETIC_MATH,
            source_metadata={"task_type": "algebra"},
            prompt=prompt,
            format=TaskFormat.NUMERIC,
            reference_answer=answer,
            required_skills=[Skill.MATHEMATICAL_REASONING, Skill.LOGICAL_DEDUCTION],
            difficulty=difficulty,
        )

    def _generate_probability(self, difficulty: DifficultyLevel) -> Task:
        """Generate probability reasoning task."""
        if difficulty == DifficultyLevel.EASY:
            # Simple probability
            total = self.rng.randint(10, 50)
            favorable = self.rng.randint(1, total - 1)

            prompt = f"""A bag contains {total} marbles. {favorable} of them are red.

What is the probability of drawing a red marble? Express as a decimal rounded to 2 places."""

            answer = f"{favorable / total:.2f}"

        elif difficulty == DifficultyLevel.MEDIUM:
            # Two events
            total = self.rng.randint(20, 52)
            type_a = self.rng.randint(5, total // 2)
            type_b = self.rng.randint(5, total - type_a)

            # P(A or B) for disjoint events
            prob = (type_a + type_b) / total

            prompt = f"""A deck has {total} cards: {type_a} are hearts and {type_b} are spades (no overlap).

What is the probability of drawing a heart or a spade? Express as a decimal rounded to 2 places."""

            answer = f"{prob:.2f}"

        elif difficulty == DifficultyLevel.HARD:
            # Conditional probability
            total = 100
            has_feature = self.rng.randint(20, 50)
            has_feature_positive = int(has_feature * self.rng.uniform(0.6, 0.9))
            no_feature = total - has_feature
            no_feature_positive = int(no_feature * self.rng.uniform(0.1, 0.3))

            total_positive = has_feature_positive + no_feature_positive
            prob = has_feature_positive / total_positive

            prompt = f"""In a study of {total} patients:
- {has_feature} have condition X, of which {has_feature_positive} tested positive
- {no_feature} don't have condition X, of which {no_feature_positive} tested positive

Given a positive test result, what's the probability the patient has condition X?
Express as a decimal rounded to 2 places."""

            answer = f"{prob:.2f}"

        else:
            # Complex conditional / Bayes
            p_a = self.rng.uniform(0.1, 0.5)
            p_b_given_a = self.rng.uniform(0.6, 0.95)
            p_b_given_not_a = self.rng.uniform(0.05, 0.3)

            p_b = p_a * p_b_given_a + (1 - p_a) * p_b_given_not_a
            p_a_given_b = (p_b_given_a * p_a) / p_b

            prompt = f"""Given:
- P(Disease) = {p_a:.2f}
- P(Positive | Disease) = {p_b_given_a:.2f}
- P(Positive | No Disease) = {p_b_given_not_a:.2f}

Using Bayes' theorem, calculate P(Disease | Positive).
Express as a decimal rounded to 2 places."""

            answer = f"{p_a_given_b:.2f}"

        return Task(
            source=TaskSource.SYNTHETIC_MATH,
            source_metadata={"task_type": "probability"},
            prompt=prompt,
            format=TaskFormat.NUMERIC,
            reference_answer=answer,
            required_skills=[
                Skill.MATHEMATICAL_REASONING,
                Skill.CAUSAL_INFERENCE,
            ],
            difficulty=difficulty,
        )

    def _generate_number_theory(self, difficulty: DifficultyLevel) -> Task:
        """Generate number theory task."""
        if difficulty == DifficultyLevel.EASY:
            # GCD/LCM
            a = self.rng.randint(10, 50)
            b = self.rng.randint(10, 50)

            import math
            gcd = math.gcd(a, b)

            prompt = f"""What is the greatest common divisor (GCD) of {a} and {b}?

Provide only the numerical answer."""
            answer = str(gcd)

        elif difficulty == DifficultyLevel.MEDIUM:
            # Divisibility
            base = self.rng.randint(100, 999)
            multiplier = self.rng.randint(7, 13)
            product = base * multiplier

            prompt = f"""What is the smallest prime factor of {product}?

Provide only the numerical answer."""

            # Find smallest prime factor
            n = product
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    answer = str(i)
                    break
            else:
                answer = str(n)

        elif difficulty == DifficultyLevel.HARD:
            # Modular arithmetic
            base = self.rng.randint(2, 10)
            exp = self.rng.randint(10, 30)
            mod = self.rng.randint(7, 17)

            result = pow(base, exp, mod)

            prompt = f"""What is {base}^{exp} mod {mod}?

Provide only the numerical answer."""
            answer = str(result)

        else:
            # Harder number theory
            n = self.rng.randint(50, 200)

            # Count primes up to n
            def count_primes(n):
                if n < 2:
                    return 0
                sieve = [True] * (n + 1)
                sieve[0] = sieve[1] = False
                for i in range(2, int(n**0.5) + 1):
                    if sieve[i]:
                        for j in range(i*i, n + 1, i):
                            sieve[j] = False
                return sum(sieve)

            prime_count = count_primes(n)

            prompt = f"""How many prime numbers are less than or equal to {n}?

Provide only the numerical answer."""
            answer = str(prime_count)

        return Task(
            source=TaskSource.SYNTHETIC_MATH,
            source_metadata={"task_type": "number_theory"},
            prompt=prompt,
            format=TaskFormat.NUMERIC,
            reference_answer=answer,
            required_skills=[Skill.MATHEMATICAL_REASONING],
            difficulty=difficulty,
        )
