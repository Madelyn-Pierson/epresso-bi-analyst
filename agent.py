"""Assistant: a dispatcher plus a router.
The router (rules now, a language model later) only CHOOSES a tool and its
arguments. Our code RUNS the tool. Nothing here executes model-written code."""
from __future__ import annotations

import analytics
from analytics import ToolResult

# The ONLY things the assistant may do. Add your own tools here.
TOOLS = {
    'get_total': {
        'fn': analytics.get_total,
        'params': {'value_col', 'filters'},
        'description': 'Sum a numeric column, optionally filtered.',
    },
    'group_total': {
        'fn': analytics.group_total,
        'params': {'value_col', 'group_col', 'filters'},
        'description': 'Sum a numeric column by group (store, channel, year, ...).',
    },
    'data_quality_report': {
        'fn': analytics.data_quality_report,
        'params': set(),
        'description': 'Report duplicates and missing values in the loaded data.',
    },
}

UNSUPPORTED = ('I can only answer questions that my approved analytical functions can '
               'calculate from the loaded data: ' + ', '.join(TOOLS) + '. '
               'I cannot answer that one.')

# For when you add a language model: keep these rules in the system prompt.
SYSTEM_PROMPT = """You are an analytical assistant for E-Presso managers.
Rules:
- Only state numbers that appear in a tool result. Never estimate or recall figures.
- Always call a tool before making a numerical claim.
- State the scope (which filters/period) of every answer.
- Separate Findings (from tool results), Interpretation, and Next steps.
- Revenue is not profit. Association is not causation.
- If the data cannot support a conclusion, say so and name the missing information.
- Never ask for, run, or write code. Never modify data."""


def run_tool(name: str, args: dict, df, log: list) -> ToolResult:
    """The dispatcher: validate the request, run the function, log the call."""
    if name not in TOOLS:
        result = analytics._error(name, f'Unknown tool: {name}')
    else:
        unexpected = set(args) - TOOLS[name]['params']
        if unexpected:
            result = analytics._error(name, f'Unexpected argument(s): {sorted(unexpected)}')
        else:
            result = TOOLS[name]['fn'](df, **args)
    log.append({'tool': name, 'arguments': args, 'result': result.to_dict()})
    return result


def rules_router(question: str, df):
    """Option B router. TODO: add rules for the tools YOU build.
    Returns (tool_name, arguments) or None if the question is unsupported."""
    q = question.lower()
    if any(w in q for w in ('missing', 'duplicate', 'quality', 'blank')):
        return 'data_quality_report', {}
    return None


def llm_router(question: str, df):
    """Option A router. TODO: after the rules-based version works, ask a model that
    supports function calling to choose from TOOLS. Return (name, args) or None."""
    raise NotImplementedError('Add the language-model router after your tools are tested.')


def format_findings(result: ToolResult) -> str:
    if result.error:
        return f'The request could not be completed: {result.error}'
    return f'{result.definition}: {result.value}  (rows used: {result.rows_used})'


def answer_question(question: str, df, mode: str = 'rules') -> dict:
    log: list = []
    routed = rules_router(question, df) if mode == 'rules' else llm_router(question, df)
    if routed is None:
        return {'scope': 'n/a', 'findings': UNSUPPORTED, 'interpretation': '',
                'next_steps': '', 'tool_log': log}
    name, args = routed
    result = run_tool(name, args, df, log)
    return {
        'scope': str(result.scope),
        'findings': format_findings(result),
        'interpretation': ' '.join(result.caveats),
        'next_steps': '',
        'tool_log': log,
    }
