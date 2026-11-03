"""Assistant: a dispatcher plus a router.
The router (rules now, a language model later) only CHOOSES a tool and its
arguments. Our code RUNS the tool. Nothing here executes model-written code."""
from __future__ import annotations

import re

import analytics
from analytics import ToolResult

# The ONLY things the assistant may do. Add your own tools here.
TOOLS = {
    'get_total': {
        'fn': analytics.get_total,
        'params': {'value_col', 'filters'},
        'description': 'Sum a numeric column (revenue), optionally filtered.',
    },
    'group_total': {
        'fn': analytics.group_total,
        'params': {'value_col', 'group_col', 'filters'},
        'description': 'Sum revenue by group (store type, category, market, ...).',
    },
    'top_n': {
        'fn': analytics.top_n,
        'params': {'value_col', 'group_col', 'n', 'filters'},
        'description': 'The n groups with the highest revenue.',
    },
    'revenue_share': {
        'fn': analytics.revenue_share,
        'params': {'value_col', 'group_col', 'filters'},
        'description': 'Each group\'s revenue as a percent of the total.',
    },
    'monthly_revenue': {
        'fn': analytics.monthly_revenue,
        'params': {'date_col', 'value_col', 'filters'},
        'description': 'Revenue per calendar month.',
    },
    'yoy_growth': {
        'fn': analytics.yoy_growth,
        'params': {'date_col', 'value_col', 'filters'},
        'description': 'Yearly revenue and percent change versus the prior year.',
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


def _mentioned_column(q: str, df, exclude=()):
    """The longest column name that appears in the question (spaces/case ignored)."""
    squash = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
    q_flat = squash(q)
    hits = [c for c in df.columns if c not in exclude and len(squash(c)) >= 3 and squash(c) in q_flat]
    return max(hits, key=lambda c: len(squash(c))) if hits else None


def rules_router(question: str, df, value_col: str | None = None, date_col: str | None = None):
    """Option B router: simple keyword rules. Extend these for the tools YOU build.
    Returns (tool_name, arguments) or None if the question is unsupported."""
    q = question.lower()
    if any(w in q for w in ('missing', 'duplicate', 'quality', 'blank')):
        return 'data_quality_report', {}
    value_col = value_col or ('Revenue' if 'Revenue' in df.columns else None)
    if value_col is None:
        return None
    if date_col and any(w in q for w in ('growth', 'yoy', 'year over year', 'year-over-year', 'per year', 'by year')):
        return 'yoy_growth', {'date_col': date_col, 'value_col': value_col}
    if date_col and any(w in q for w in ('month', 'trend')):
        return 'monthly_revenue', {'date_col': date_col, 'value_col': value_col}
    group = _mentioned_column(q, df, exclude={value_col, date_col})
    if group:
        if any(w in q for w in ('share', 'percent', '%', 'proportion')):
            return 'revenue_share', {'value_col': value_col, 'group_col': group}
        if any(w in q for w in ('top', 'best', 'highest', 'largest')):
            found = re.search(r'\b(\d+)\b', q)
            return 'top_n', {'value_col': value_col, 'group_col': group, 'n': int(found.group(1)) if found else 5}
        return 'group_total', {'value_col': value_col, 'group_col': group}
    if any(w in q for w in ('total', 'revenue', 'sales')):
        return 'get_total', {'value_col': value_col}
    return None


def llm_router(question: str, df, value_col=None, date_col=None):
    """Option A router. TODO: after the rules-based version works, ask a model that
    supports function calling to choose from TOOLS. Return (name, args) or None."""
    raise NotImplementedError('Add the language-model router after your tools are tested.')


def format_findings(result: ToolResult) -> str:
    if result.error:
        return f'The request could not be completed: {result.error}'
    return f'{result.definition}: {result.value}  (rows used: {result.rows_used})'


def answer_question(question: str, df, mode: str = 'rules',
                    value_col: str | None = None, date_col: str | None = None) -> dict:
    log: list = []
    router = rules_router if mode == 'rules' else llm_router
    routed = router(question, df, value_col, date_col)
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