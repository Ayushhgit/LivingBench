"""
Tool use task generator.

Generates scenarios requiring appropriate tool selection and composition:
- Search tasks
- Calculator/computation tasks
- Code execution tasks
- Multi-tool composition tasks

These tasks evaluate:
- Tool selection accuracy
- Argument construction
- Result interpretation
- Error recovery
"""

from __future__ import annotations

import random
import json
from typing import Iterator

from livingbench.core.types import (
    Task,
    TaskSource,
    TaskFormat,
    Skill,
    DifficultyLevel,
    ToolDefinition,
)
from livingbench.tasks.base import TaskGeneratorBase
from livingbench.core.registry import task_generators


# Define available tools for evaluation
AVAILABLE_TOOLS = [
    ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations. Supports basic arithmetic, exponents, roots, trigonometry.",
        parameters={
            "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
        },
        required_params=["expression"],
        returns="number",
    ),
    ToolDefinition(
        name="web_search",
        description="Search the web for current information. Returns top search results.",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results to return", "default": 5},
        },
        required_params=["query"],
        returns="list of search results",
    ),
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a location.",
        parameters={
            "location": {"type": "string", "description": "City name or coordinates"},
            "units": {"type": "string", "description": "celsius or fahrenheit", "default": "celsius"},
        },
        required_params=["location"],
        returns="weather information",
    ),
    ToolDefinition(
        name="run_code",
        description="Execute Python code in a sandboxed environment.",
        parameters={
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Max execution time in seconds", "default": 30},
        },
        required_params=["code"],
        returns="code output or error",
    ),
    ToolDefinition(
        name="read_file",
        description="Read contents of a file.",
        parameters={
            "path": {"type": "string", "description": "File path to read"},
            "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
        },
        required_params=["path"],
        returns="file contents",
    ),
    ToolDefinition(
        name="write_file",
        description="Write contents to a file.",
        parameters={
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        required_params=["path", "content"],
        returns="success status",
    ),
    ToolDefinition(
        name="http_request",
        description="Make an HTTP request to a URL.",
        parameters={
            "url": {"type": "string", "description": "URL to request"},
            "method": {"type": "string", "description": "HTTP method", "default": "GET"},
            "body": {"type": "object", "description": "Request body for POST/PUT"},
        },
        required_params=["url"],
        returns="HTTP response",
    ),
    ToolDefinition(
        name="database_query",
        description="Execute a SQL query on the database.",
        parameters={
            "query": {"type": "string", "description": "SQL query to execute"},
            "params": {"type": "array", "description": "Query parameters"},
        },
        required_params=["query"],
        returns="query results",
    ),
]


@task_generators.register("tool_use")
class ToolUseTaskGenerator(TaskGeneratorBase):
    """
    Generate tool use evaluation tasks.

    Tests various aspects of tool use:
    1. Tool selection: Choosing the right tool for the task
    2. Argument construction: Providing correct parameters
    3. Multi-step tool use: Composing multiple tools
    4. Error handling: Recovering from tool failures
    5. Tool avoidance: Knowing when NOT to use tools
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tools = AVAILABLE_TOOLS

    @property
    def source(self) -> TaskSource:
        return TaskSource.TOOL_USE_SCENARIO

    def generate(self, n_tasks: int, **kwargs) -> Iterator[Task]:
        """Generate tool use tasks."""
        task_types = [
            "single_tool",
            "multi_tool",
            "tool_selection",
            "no_tool_needed",
            "error_recovery",
        ]

        difficulty_weights = kwargs.get("difficulty_weights", {
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.3,
            DifficultyLevel.VERY_HARD: 0.1,
        })

        for _ in range(n_tasks):
            task_type = self.rng.choice(task_types)
            difficulty = self.rng.choices(
                list(difficulty_weights.keys()),
                weights=list(difficulty_weights.values())
            )[0]

            if task_type == "single_tool":
                task = self._generate_single_tool_task(difficulty)
            elif task_type == "multi_tool":
                task = self._generate_multi_tool_task(difficulty)
            elif task_type == "tool_selection":
                task = self._generate_tool_selection_task(difficulty)
            elif task_type == "no_tool_needed":
                task = self._generate_no_tool_task(difficulty)
            else:
                task = self._generate_error_recovery_task(difficulty)

            if self.validate_task(task):
                yield task

    def _generate_single_tool_task(self, difficulty: DifficultyLevel) -> Task:
        """Generate task requiring single tool use."""
        scenarios = [
            {
                "prompt": "What is the result of 17.5 * 23.8 + 156.2 / 4?",
                "tool": "calculator",
                "expected_call": {"name": "calculator", "arguments": {"expression": "17.5 * 23.8 + 156.2 / 4"}},
                "answer": "455.8",
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "prompt": "Calculate the compound interest on $10,000 at 5% annual rate for 10 years, compounded annually.",
                "tool": "calculator",
                "expected_call": {"name": "calculator", "arguments": {"expression": "10000 * (1 + 0.05) ** 10"}},
                "answer": "16288.95",
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "prompt": "Write a Python function that finds all prime numbers up to 100 and execute it to show the result.",
                "tool": "run_code",
                "expected_call": {
                    "name": "run_code",
                    "arguments": {
                        "code": "def primes(n):\n    sieve = [True] * (n+1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5)+1):\n        if sieve[i]:\n            for j in range(i*i, n+1, i):\n                sieve[j] = False\n    return [i for i in range(n+1) if sieve[i]]\n\nprint(primes(100))"
                    }
                },
                "answer": "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]",
                "difficulty": DifficultyLevel.HARD,
            },
        ]

        # Filter by difficulty
        valid_scenarios = [s for s in scenarios if s["difficulty"] == difficulty]
        if not valid_scenarios:
            valid_scenarios = scenarios

        scenario = self.rng.choice(valid_scenarios)

        # Build tool list for this task
        tool_objs = [t for t in self.tools if t.name == scenario["tool"]]
        # Add some distractors
        distractors = [t for t in self.tools if t.name != scenario["tool"]]
        tool_objs.extend(self.rng.sample(distractors, min(2, len(distractors))))

        prompt = f"""You have access to the following tools:

{self._format_tools(tool_objs)}

Task: {scenario['prompt']}

Use the appropriate tool to complete this task. Respond with a JSON object containing:
- "tool_call": the tool name and arguments
- "reasoning": why you chose this tool"""

        return Task(
            source=TaskSource.TOOL_USE_SCENARIO,
            source_metadata={"task_type": "single_tool", "target_tool": scenario["tool"]},
            prompt=prompt,
            format=TaskFormat.TOOL_CALLS,
            reference_answer=json.dumps(scenario["expected_call"]),
            available_tools=tool_objs,
            expected_tool_calls=[scenario["expected_call"]],
            required_skills=[Skill.TOOL_SELECTION, Skill.INSTRUCTION_FOLLOWING],
            difficulty=difficulty,
        )

    def _generate_multi_tool_task(self, difficulty: DifficultyLevel) -> Task:
        """Generate task requiring multiple tools."""
        scenarios = [
            {
                "prompt": """I need to analyze weather patterns. First, get the current weather for London,
then calculate the difference between today's temperature (assume 15°C from the result) and the average annual temperature of 11°C.""",
                "tools": ["get_weather", "calculator"],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"location": "London"}},
                    {"name": "calculator", "arguments": {"expression": "15 - 11"}},
                ],
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "prompt": """Write a Python script that generates 10 random numbers between 1 and 100,
then use the calculator to compute their mean and standard deviation.""",
                "tools": ["run_code", "calculator"],
                "expected_calls": [
                    {"name": "run_code", "arguments": {"code": "import random; nums = [random.randint(1, 100) for _ in range(10)]; print(nums)"}},
                    {"name": "calculator", "arguments": {"expression": "mean calculation"}},
                    {"name": "calculator", "arguments": {"expression": "std calculation"}},
                ],
                "difficulty": DifficultyLevel.HARD,
            },
            {
                "prompt": """Search for the current population of Tokyo, then calculate what percentage
of Japan's population (125 million) lives in Tokyo.""",
                "tools": ["web_search", "calculator"],
                "expected_calls": [
                    {"name": "web_search", "arguments": {"query": "Tokyo current population"}},
                    {"name": "calculator", "arguments": {"expression": "(tokyo_pop / 125000000) * 100"}},
                ],
                "difficulty": DifficultyLevel.MEDIUM,
            },
        ]

        valid_scenarios = [s for s in scenarios if s["difficulty"] == difficulty]
        if not valid_scenarios:
            valid_scenarios = scenarios

        scenario = self.rng.choice(valid_scenarios)

        # Get required tools
        tool_objs = [t for t in self.tools if t.name in scenario["tools"]]
        # Add distractors
        distractors = [t for t in self.tools if t.name not in scenario["tools"]]
        tool_objs.extend(self.rng.sample(distractors, min(2, len(distractors))))

        prompt = f"""You have access to the following tools:

{self._format_tools(tool_objs)}

Task: {scenario['prompt']}

This task requires multiple tool calls in sequence. Respond with a JSON object containing:
- "tool_calls": array of tool calls in order
- "reasoning": explain your approach"""

        return Task(
            source=TaskSource.TOOL_USE_SCENARIO,
            source_metadata={"task_type": "multi_tool", "target_tools": scenario["tools"]},
            prompt=prompt,
            format=TaskFormat.TOOL_CALLS,
            reference_answer=json.dumps(scenario["expected_calls"]),
            available_tools=tool_objs,
            expected_tool_calls=scenario["expected_calls"],
            required_skills=[
                Skill.TOOL_SELECTION,
                Skill.TOOL_COMPOSITION,
                Skill.MULTI_STEP_PLANNING,
            ],
            difficulty=difficulty,
        )

    def _generate_tool_selection_task(self, difficulty: DifficultyLevel) -> Task:
        """Generate task testing tool selection ability."""
        scenarios = [
            {
                "prompt": "What is 2 + 2?",
                "correct_approach": "no_tool",  # Too simple, should answer directly
                "incorrect_tools": ["calculator"],
                "reasoning": "This is trivial arithmetic that doesn't require a tool",
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "prompt": "Look up when the Eiffel Tower was built",
                "correct_approach": "web_search",
                "incorrect_tools": ["calculator", "run_code"],
                "reasoning": "This requires factual lookup, not computation",
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "prompt": "Calculate the factorial of 20",
                "correct_approach": "calculator_or_code",
                "incorrect_tools": ["web_search", "get_weather"],
                "reasoning": "This requires computation, either via calculator or code",
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "prompt": "Generate all permutations of the letters A, B, C, D",
                "correct_approach": "run_code",
                "incorrect_tools": ["calculator", "web_search"],
                "reasoning": "This requires algorithmic generation, best done with code",
                "difficulty": DifficultyLevel.HARD,
            },
        ]

        valid_scenarios = [s for s in scenarios if s["difficulty"] == difficulty]
        if not valid_scenarios:
            valid_scenarios = scenarios

        scenario = self.rng.choice(valid_scenarios)

        prompt = f"""You have access to the following tools:

{self._format_tools(self.tools)}

Task: {scenario['prompt']}

Analyze which tool (if any) is most appropriate for this task. Respond with JSON:
- "selected_tool": tool name or "none" if no tool needed
- "reasoning": why this is the best choice
- "alternatives_considered": other tools you considered and why they're less suitable"""

        return Task(
            source=TaskSource.TOOL_USE_SCENARIO,
            source_metadata={"task_type": "tool_selection"},
            prompt=prompt,
            format=TaskFormat.STRUCTURED_JSON,
            reference_answer=json.dumps({
                "correct_approach": scenario["correct_approach"],
                "reasoning": scenario["reasoning"],
            }),
            available_tools=self.tools,
            required_skills=[Skill.TOOL_SELECTION, Skill.UNCERTAINTY_CALIBRATION],
            difficulty=difficulty,
        )

    def _generate_no_tool_task(self, difficulty: DifficultyLevel) -> Task:
        """Generate task where tool use is NOT needed."""
        scenarios = [
            {
                "prompt": "What color is the sky on a clear day?",
                "answer": "Blue",
                "trap_tools": ["web_search"],
                "reasoning": "Common knowledge, no lookup needed",
            },
            {
                "prompt": "Translate 'Hello' to Spanish",
                "answer": "Hola",
                "trap_tools": ["web_search"],
                "reasoning": "Basic translation within model capabilities",
            },
            {
                "prompt": "What is the capital of France?",
                "answer": "Paris",
                "trap_tools": ["web_search"],
                "reasoning": "Basic factual knowledge, no lookup needed",
            },
            {
                "prompt": "Write a haiku about programming",
                "answer": None,  # Creative task
                "trap_tools": ["web_search", "run_code"],
                "reasoning": "Creative task, no tool can help",
            },
        ]

        scenario = self.rng.choice(scenarios)

        # Include trap tools to see if model uses them unnecessarily
        trap_tool_objs = [t for t in self.tools if t.name in scenario["trap_tools"]]
        other_tools = [t for t in self.tools if t.name not in scenario["trap_tools"]]
        all_tools = trap_tool_objs + self.rng.sample(other_tools, min(2, len(other_tools)))

        prompt = f"""You have access to the following tools:

{self._format_tools(all_tools)}

Task: {scenario['prompt']}

Important: Only use tools if they are truly necessary. Answer directly if you can.

Respond with JSON:
- "answer": your answer
- "tool_used": tool name if you used one, or "none"
- "reasoning": why you did or didn't use a tool"""

        return Task(
            source=TaskSource.TOOL_USE_SCENARIO,
            source_metadata={
                "task_type": "no_tool_needed",
                "trap_tools": scenario["trap_tools"],
            },
            prompt=prompt,
            format=TaskFormat.STRUCTURED_JSON,
            reference_answer=json.dumps({
                "answer": scenario["answer"],
                "tool_used": "none",
                "reasoning": scenario["reasoning"],
            }),
            available_tools=all_tools,
            expected_tool_calls=[],  # No tools should be called
            required_skills=[Skill.ABSTENTION, Skill.UNCERTAINTY_CALIBRATION],
            difficulty=DifficultyLevel.MEDIUM,
        )

    def _generate_error_recovery_task(self, difficulty: DifficultyLevel) -> Task:
        """Generate task testing error recovery from tool failures."""
        scenarios = [
            {
                "prompt": """Use the calculator to evaluate: sqrt(-1)
Note: The calculator will return an error for this. How should you handle it?""",
                "expected_behavior": "recognize_complex_number",
                "reasoning": "Should explain that sqrt(-1) = i (imaginary unit)",
                "difficulty": DifficultyLevel.MEDIUM,
            },
            {
                "prompt": """Fetch the weather for "InvalidCityName12345"
The tool will return an error. What should you do?""",
                "expected_behavior": "ask_for_clarification",
                "reasoning": "Should ask user to provide a valid city name",
                "difficulty": DifficultyLevel.EASY,
            },
            {
                "prompt": """Execute this Python code: print(undefined_variable)
The code will fail. How do you handle this?""",
                "expected_behavior": "explain_and_fix",
                "reasoning": "Should explain the NameError and suggest defining the variable",
                "difficulty": DifficultyLevel.MEDIUM,
            },
        ]

        scenario = self.rng.choice(scenarios)

        prompt = f"""You have access to the following tools:

{self._format_tools(self.tools)}

Task: {scenario['prompt']}

Respond with JSON:
- "initial_attempt": what you tried
- "error_encountered": what went wrong
- "recovery_action": how you handled the error
- "final_response": your response to the user"""

        return Task(
            source=TaskSource.TOOL_USE_SCENARIO,
            source_metadata={"task_type": "error_recovery"},
            prompt=prompt,
            format=TaskFormat.STRUCTURED_JSON,
            reference_answer=json.dumps({
                "expected_behavior": scenario["expected_behavior"],
                "reasoning": scenario["reasoning"],
            }),
            available_tools=self.tools,
            required_skills=[Skill.ERROR_RECOVERY, Skill.SELF_CORRECTION],
            difficulty=scenario["difficulty"],
        )

    def _format_tools(self, tools: list[ToolDefinition]) -> str:
        """Format tools for display in prompt."""
        lines = []
        for tool in tools:
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in tool.parameters.items()
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)
