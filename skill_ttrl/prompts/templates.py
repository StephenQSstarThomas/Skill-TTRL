"""
Prompt templates for Skill TTRL.

Defines the system prompt and formatting for injecting skill bank context
and instructing the model to perform skill operations.
"""

# System prompt that instructs the model on the skill ops format
SKILL_TTRL_SYSTEM_PROMPT = """\
You are an expert problem solver. For each problem, you may perform skill \
operations before solving it. You have access to a skill bank of previously \
learned strategies.

## Available Skill Operations

You may include a <skill_ops> block before your solution with any combination of:

1. **Generate**: Create a new skill/strategy relevant to this problem.
   <generate>Description of the new skill...</generate>

2. **Retrieve**: Request relevant skills from the skill bank.
   <retrieve query="search terms">Retrieved skills will appear here</retrieve>

3. **Evolve**: Improve an existing skill from the bank.
   <evolve base="#skill_id">Improved version of the skill...</evolve>

You decide which operations (if any) to perform based on the problem. \
Then solve the problem using the skills you've gathered.

## Output Format

<skill_ops>
  (your skill operations here, or leave empty if none needed)
</skill_ops>

<solution>
Your step-by-step reasoning and solution process.
</solution>

<answer>
Your final answer.
</answer>"""


# Template for math/reasoning tasks
REASONING_PROMPT_TEMPLATE = """\
{system_prompt}

{skill_bank_context}

## Problem

{problem}

{suffix_prompt}"""


# Template for agent tasks (ALFWorld, WebShop)
AGENT_PROMPT_TEMPLATE = """\
{system_prompt}

{skill_bank_context}

## Task

{task_description}

## Current State

{current_state}

## Available Actions

{available_actions}

Decide which skill operations to perform (if any), then choose your action."""


# Simpler template when skill bank is empty (cold start)
COLD_START_TEMPLATE = """\
{system_prompt}

The skill bank is currently empty. As you solve this problem, think about what \
general mathematical strategies or techniques you are using that could be \
applied to similar problems.

## Problem

{problem}

{suffix_prompt}"""


# Template for the suffix that instructs answer formatting
MATH_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within "
    "\\boxed{}."
)

CODE_SUFFIX = (
    "\nPlease provide your solution as a Python function."
)
