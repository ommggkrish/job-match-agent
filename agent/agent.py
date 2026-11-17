"""The agent loop.

1. Send the job ad + tool list to the LLM.
2. If the LLM asks for tools, run them and send the results back. Repeat.
3. When the LLM stops asking for tools (or MAX_AGENT_STEPS is reached),
   ask it for a ReportDraft in JSON mode.
4. Code - not the LLM - applies the guardrail and computes the score.
"""

import json
import re
from dataclasses import dataclass, field

from agent import config
from agent.models import ConstraintCheck, MatchReport, ReportDraft
from agent.tools import TOOL_SCHEMAS, ToolBox, normalize_skill

SYSTEM_PROMPT = f"""You are a careful career assistant for a working-student job search.
Your job: decide how well ONE job ad fits the candidate's CV and rules.

Work with the tools, step by step:
1. If the user gave a URL, call fetch_job_ad first.
2. Call extract_requirements.
3. Then call check_constraints (with the extracted location, work_mode, min_hours_per_week and
   german_level) AND compare_skills (with the extracted skills) together in the same turn -
   they do not depend on each other. compare_skills already returns the CV skill list;
   call read_cv only if you really need the full CV text.
When you have all results, reply with the single word DONE.

Rules: be honest. Never claim the candidate has a skill that is not in the CV.
CV versions you can recommend: {", ".join(config.CV_VERSIONS)}."""

FINAL_PROMPT = """Write the match report as JSON.
- matching_skills: ONLY skills from this CV skill list: {cv_skills}
- missing_skills: skills the job needs that are not in the list
- cv_version: the CV version that best fits the job
- extra_red_flags: serious problems not already covered by the constraint check (can be empty)
- tips: 2-4 short, concrete, honest tips. Never tell the candidate to claim skills they lack.

TOOL RESULTS:
{tool_results}

JOB AD (start):
{job_ad}"""


@dataclass
class AgentResult:
    report: MatchReport
    trace: list[dict] = field(default_factory=list)  # every tool call, for the UI and debugging
    steps: int = 0
    hit_step_limit: bool = False


def looks_like_url(text: str) -> bool:
    return bool(re.fullmatch(r"https?://\S+", text.strip()))


def run_agent(job_input: str, llm, toolbox: ToolBox | None = None, max_steps: int = config.MAX_AGENT_STEPS) -> AgentResult:
    toolbox = toolbox or ToolBox(llm)
    job_input = job_input.strip()

    if looks_like_url(job_input):
        first_message = f"Job ad URL: {job_input}"
    else:
        toolbox.job_text = job_input
        first_message = f"Here is the job ad text:\n\n{job_input[:12_000]}"

    messages: list[dict] = [{"role": "user", "text": first_message}]
    trace: list[dict] = []
    hit_limit = True

    # ---- the loop: the model decides which tool to call next
    step = 0
    for step in range(1, max_steps + 1):
        turn = llm.next_turn(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
        messages.append({"role": "model", "text": turn.text, "tool_calls": turn.tool_calls, "raw": turn.raw})
        if not turn.tool_calls:
            hit_limit = False
            break
        for call in turn.tool_calls:
            result = toolbox.call(call.name, call.args)
            trace.append({"step": step, "tool": call.name, "args": call.args, "result": result})
            messages.append({"role": "tool", "name": call.name, "id": call.id, "result": result})

    # ---- final answer in JSON mode, validated by Pydantic
    prompt = FINAL_PROMPT.format(
        cv_skills=", ".join(toolbox.cv_skills),
        tool_results=json.dumps(toolbox.results, ensure_ascii=False, indent=1)[:20_000],
        job_ad=toolbox.job_text[:4000],
    )
    draft = llm.structured(prompt, ReportDraft, system=SYSTEM_PROMPT)
    report = build_report(draft, toolbox)
    return AgentResult(report=report, trace=trace, steps=step, hit_step_limit=hit_limit)


# ------------------------------------------------------------- guardrail + score

def apply_skill_guardrail(claimed: list[str], cv_skills: list[str]) -> tuple[list[str], list[str]]:
    """Keep only skills that really are in the CV. Returns (kept, removed)."""
    cv_by_key = {normalize_skill(s): s for s in cv_skills}
    kept, removed = [], []
    for skill in claimed:
        real = cv_by_key.get(normalize_skill(skill))
        if real is None:
            removed.append(skill)
        elif real not in kept:
            kept.append(real)
    return kept, removed


def compute_score(n_matching: int, n_missing: int, n_red_flags: int) -> int:
    """Share of required skills the CV covers, minus 30 points per red flag."""
    total = n_matching + n_missing
    skill_score = round(100 * n_matching / total) if total else 50
    return max(0, min(100, skill_score - 30 * n_red_flags))


def verdict_for(score: int, has_red_flags: bool) -> str:
    if has_red_flags:
        return "not a fit" if score < 30 else "weak fit"
    if score >= 70:
        return "strong fit"
    if score >= 45:
        return "possible fit"
    if score >= 25:
        return "weak fit"
    return "not a fit"


def build_report(draft: ReportDraft, toolbox: ToolBox) -> MatchReport:
    requirements = toolbox.results.get("extract_requirements") or {}
    constraints_raw = toolbox.results.get("check_constraints")
    constraints = ConstraintCheck(**constraints_raw) if constraints_raw else None

    matching, removed = apply_skill_guardrail(draft.matching_skills, toolbox.cv_skills)
    # An invented "match" is really a missing skill.
    missing = list(dict.fromkeys(draft.missing_skills + removed))
    missing = [s for s in missing if normalize_skill(s) not in {normalize_skill(m) for m in matching}]

    red_flags = list(constraints.red_flags) if constraints else []
    red_flags += [f for f in draft.extra_red_flags if f not in red_flags]
    warnings = list(constraints.warnings) if constraints else ["Constraint check did not run."]

    score = compute_score(len(matching), len(missing), len(red_flags))
    return MatchReport(
        job_title=requirements.get("job_title") or "Unknown job",
        company=requirements.get("company"),
        score=score,
        verdict=verdict_for(score, bool(red_flags)),
        summary=draft.summary,
        matching_skills=matching,
        missing_skills=missing,
        cv_version=draft.cv_version,
        red_flags=red_flags,
        warnings=warnings,
        tips=draft.tips,
        constraints=constraints,
        guardrail_removed=removed,
    )
