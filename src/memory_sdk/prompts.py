"""Default prompts for the four consolidation stages.

These ship as working defaults and are overridable per stage via
``MemoryConfig.prompts``. What is NOT overridable is the machine-checked
contract around them: extraction and reflection must return a JSON array (the
tolerant parser in ``consolidation.py`` handles prose wrappers and trailing
commas), and core-memory condensation output is validated for length and an
over-compression floor in code — a prompt override cannot relax those guards.
"""

from __future__ import annotations

DEFAULT_EXTRACT_PROMPT = (
    "You extract durable memories from a short run of consecutive exchanges between a "
    "user and an AI assistant. "
    "Return ONLY a JSON array, no prose, no code fences. Each element is an object:\n"
    '  {"text": "<one atomic fact, third person, self-contained>",\n'
    '   "subject": "<the main entity the fact is about, e.g. the user\'s name or \'user\'>",\n'
    '   "importance": <integer 1-10>,\n'
    '   "scope": "user" | "relationship" | "world",\n'
    '   "entities": ["<other named entities mentioned>"]}\n'
    "Record only lasting facts: names, preferences, relationships, recurring people, "
    "plans, and notable events. Ignore greetings, small talk, and the assistant's own "
    "conversational filler. Use scope 'user' for facts about the user, "
    "'relationship' for facts about the user<->assistant bond, 'world' otherwise. "
    "If nothing is worth remembering, return []."
)

DEFAULT_SUMMARY_PROMPT = (
    "You maintain ONE rolling summary of what an AI assistant knows about its user. "
    "You are given the previous summary (possibly empty) and the facts learned since "
    "it was written. Rewrite the summary so it carries forward everything from the "
    "previous summary that is still true, folds in the new facts, and drops only what "
    "the new facts contradict. 3-5 sentences of plain prose, no list, no preamble."
)

DEFAULT_REFLECT_PROMPT = (
    "From the following facts, infer up to 3 higher-level, non-obvious insights about the "
    "user or the relationship (patterns, inferred preferences, emotional tendencies). "
    "Return ONLY a JSON array of short strings. If nothing can be inferred, return []."
)

DEFAULT_CORE_CONDENSE_PROMPT = (
    "You compress a pinned 'core memory' block about a user. Rewrite the lines below "
    "in the same '- ' bullet format, tighter: merge lines that say the same thing and "
    "drop pure restatements. Preserve every DISTINCT durable fact — names, ages, job, "
    "location, relationships, hard preferences and boundaries. Never invent anything "
    "that is not in the input. Return ONLY the bullet lines, no preamble, no code fences."
)

# NOTE: the test suite's scripted LLM fakes route on substrings of the system
# prompt — "summar" and "insight" select the summarize/reflect stages. The
# condense prompt must contain neither word, or condensation calls get
# mis-routed in tests. If you override prompts in your own app this constraint
# does not apply to you; it only pins the defaults.
