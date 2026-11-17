import pytest

from agent.models import Requirements
from agent.tools import (
    SAMPLE_CV,
    ToolBox,
    check_constraints,
    compare_skills,
    fetch_job_ad,
    html_to_text,
    parse_cv_skills,
    read_cv,
)
from tests.fake_llm import FakeLLM

# ------------------------------------------------------------ check_constraints


def test_munich_hybrid_20h_english_is_ok():
    result = check_constraints("Munich", "hybrid", 20, "none")
    assert result.location_ok and result.hours_ok and result.language_ok
    assert result.red_flags == []


def test_remote_is_ok_even_if_company_is_far_away():
    assert check_constraints("Hamburg", "remote", 20, "none").location_ok


def test_onsite_berlin_is_red_flag():
    result = check_constraints("Berlin", "on-site", 20, "none")
    assert not result.location_ok
    assert len(result.red_flags) == 1


def test_german_city_spelling_muenchen_is_ok():
    assert check_constraints("München, Bayern", "on-site", 15, "none").location_ok


def test_more_than_20_hours_is_red_flag():
    result = check_constraints("Regensburg", "hybrid", 40, "none")
    assert not result.hours_ok
    assert "40" in result.red_flags[0]


def test_fluent_german_is_red_flag_but_plus_is_fine():
    assert not check_constraints("Landshut", "hybrid", 20, "fluent").language_ok
    assert check_constraints("Landshut", "hybrid", 20, "plus").language_ok


def test_unknown_values_give_warnings_not_red_flags():
    result = check_constraints(None, "unknown", None, "basic")
    assert result.red_flags == []
    assert len(result.warnings) == 3


# --------------------------------------------------------------- compare_skills


def test_compare_skills_handles_spelling_and_aliases():
    result = compare_skills(["python", "Sklearn", "Scikit-Learn", "Kubernetes"], ["Python", "scikit-learn"])
    assert result.matching == ["Python", "scikit-learn"]
    assert result.missing == ["Kubernetes"]


# ---------------------------------------------------------------------- read_cv


def test_sample_cv_skills_are_parsed():
    skills = parse_cv_skills(SAMPLE_CV.read_text(encoding="utf-8"))
    assert "Python" in skills
    assert "PyTorch" in skills  # "(basics)" is removed
    assert "Programming" not in skills  # category names are not skills
    assert "German" not in skills  # languages section is not the skills section


def test_read_cv_falls_back_to_sample(tmp_path):
    cv = tmp_path / "cv.md"
    cv.write_text("## Skills\n- Python, Go\n", encoding="utf-8")
    assert parse_cv_skills(read_cv(cv)) == ["Python", "Go"]


# ----------------------------------------------------------------- fetch_job_ad


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_html_to_text_drops_scripts_and_tags():
    html = "<html><head><title>x</title></head><body><script>var a=1</script><h1>ML Intern</h1><p>Python</p></body></html>"
    assert html_to_text(html) == "ML Intern Python"


def test_fetch_job_ad_returns_text():
    page = "<p>" + "We are hiring a working student. " * 20 + "</p>"
    text = fetch_job_ad("https://example.com/job", get=lambda *a, **k: FakeResponse(page))
    assert text.startswith("We are hiring")


def test_fetch_job_ad_rejects_non_http_urls():
    with pytest.raises(ValueError):
        fetch_job_ad("file:///etc/passwd")


def test_fetch_job_ad_rejects_empty_javascript_pages():
    with pytest.raises(ValueError, match="JavaScript"):
        fetch_job_ad("https://example.com", get=lambda *a, **k: FakeResponse("<div id=app></div>"))


# ---------------------------------------------------------------------- ToolBox


def test_toolbox_returns_errors_instead_of_crashing():
    box = ToolBox(FakeLLM([]))
    assert "Unknown tool" in box.call("delete_everything", {})["error"]
    assert "No job ad loaded" in box.call("extract_requirements", {})["error"]


def test_toolbox_extract_requirements_uses_llm():
    req = Requirements(job_title="ML Intern", work_mode="hybrid", german_level="none", skills=["Python"])
    box = ToolBox(FakeLLM([], {"Requirements": req}))
    box.job_text = "We need Python."
    assert box.call("extract_requirements", {})["job_title"] == "ML Intern"


def test_toolbox_accepts_float_hours_from_json():
    box = ToolBox(FakeLLM([]))
    result = box.call("check_constraints", {"work_mode": "remote", "german_level": "none", "min_hours_per_week": 20.0})
    assert result["hours_ok"] is True
