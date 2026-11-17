from agent.agent import apply_skill_guardrail, compute_score, run_agent, verdict_for
from agent.llm import ModelTurn
from agent.models import ReportDraft, Requirements
from agent.tools import ToolBox
from tests.fake_llm import FakeLLM, call

AD = "Working student Machine Learning (m/f/d), Munich, hybrid, 16-20 h/week. Python, PyTorch, Kubernetes."

REQ = Requirements(
    job_title="Working Student Machine Learning",
    company="Example GmbH",
    location="Munich",
    work_mode="hybrid",
    min_hours_per_week=16,
    german_level="plus",
    skills=["Python", "PyTorch", "Kubernetes"],
)


def make_draft(matching, missing=("Kubernetes",)):
    return ReportDraft(
        summary="Good fit.",
        matching_skills=list(matching),
        missing_skills=list(missing),
        cv_version="01-ai-ml-engineering",
        extra_red_flags=[],
        tips=["Mention your PyTorch project."],
    )


def happy_path_turns():
    """The tool calls a well-behaved model would make, one turn each."""
    return [
        ModelTurn(tool_calls=[call("extract_requirements")]),
        ModelTurn(
            tool_calls=[
                call("check_constraints", location="Munich", work_mode="hybrid", min_hours_per_week=16, german_level="plus"),
                call("compare_skills", required_skills=["Python", "PyTorch", "Kubernetes"]),
            ]
        ),
        ModelTurn(text="DONE"),
    ]


def test_agent_runs_tools_and_builds_report():
    llm = FakeLLM(happy_path_turns(), {"Requirements": REQ, "ReportDraft": make_draft(["Python", "PyTorch"])})
    result = run_agent(AD, llm)

    assert [t["tool"] for t in result.trace] == ["extract_requirements", "check_constraints", "compare_skills"]
    assert result.steps == 3 and not result.hit_step_limit
    report = result.report
    assert report.job_title == "Working Student Machine Learning"
    assert report.matching_skills == ["Python", "PyTorch"]
    assert report.missing_skills == ["Kubernetes"]
    assert report.red_flags == []
    assert report.score == 67  # 2 of 3 skills, no red flags
    assert report.verdict == "possible fit"


def test_tool_results_are_sent_back_to_the_model():
    llm = FakeLLM(happy_path_turns(), {"Requirements": REQ, "ReportDraft": make_draft(["Python"])})
    run_agent(AD, llm)
    second_request = llm.next_turn_calls[1]
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["result"]["job_title"] == REQ.job_title


def test_guardrail_removes_invented_skills():
    """The model claims Kubernetes and Rust, but neither is in the CV."""
    draft = make_draft(["Python", "Kubernetes", "Rust"], missing=())
    llm = FakeLLM(happy_path_turns(), {"Requirements": REQ, "ReportDraft": draft})
    report = run_agent(AD, llm).report

    cv_skills = ToolBox(llm).cv_skills
    assert all(skill in cv_skills for skill in report.matching_skills)
    assert report.matching_skills == ["Python"]
    assert report.guardrail_removed == ["Kubernetes", "Rust"]
    assert "Kubernetes" in report.missing_skills  # an invented match counts as missing


def test_agent_stops_at_step_limit():
    looping = [ModelTurn(tool_calls=[call("read_cv")]) for _ in range(20)]
    llm = FakeLLM(looping, {"ReportDraft": make_draft(["Python"])})
    result = run_agent(AD, llm, max_steps=3)
    assert result.hit_step_limit
    assert len(llm.next_turn_calls) == 3
    assert result.report.warnings == ["Constraint check did not run."]


def test_url_input_is_fetched_by_the_tool():
    page = "<p>" + "Working student data analyst in Regensburg. " * 10 + "</p>"

    class Resp:
        text = page

        def raise_for_status(self):
            pass

    llm = FakeLLM(
        [ModelTurn(tool_calls=[call("fetch_job_ad", url="https://jobs.example.com/1")]), ModelTurn(text="DONE")],
        {"ReportDraft": make_draft(["SQL"])},
    )
    toolbox = ToolBox(llm, get=lambda *a, **k: Resp())
    result = run_agent("https://jobs.example.com/1", llm, toolbox=toolbox)
    assert toolbox.job_text.startswith("Working student data analyst")
    assert result.trace[0]["result"]["chars"] > 200


def test_score_and_verdict_rules():
    assert compute_score(3, 0, 0) == 100
    assert compute_score(3, 1, 1) == 45
    assert compute_score(0, 0, 0) == 50
    assert compute_score(0, 5, 2) == 0
    assert verdict_for(90, has_red_flags=True) == "weak fit"
    assert verdict_for(20, has_red_flags=True) == "not a fit"
    assert verdict_for(70, has_red_flags=False) == "strong fit"


def test_guardrail_uses_cv_spelling():
    kept, removed = apply_skill_guardrail(["sklearn", "python", "Excel"], ["Python", "scikit-learn"])
    assert kept == ["scikit-learn", "Python"]
    assert removed == ["Excel"]
