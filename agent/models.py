"""Pydantic models. Everything the LLM returns is validated against one of these."""

from typing import Literal

from pydantic import BaseModel, Field

WorkMode = Literal["remote", "hybrid", "on-site", "unknown"]
GermanLevel = Literal["none", "plus", "basic", "fluent"]
CvVersion = Literal[
    "01-ai-ml-engineering",
    "02-mlops-devops",
    "03-data-analytics",
    "04-software-development",
]
Verdict = Literal["strong fit", "possible fit", "weak fit", "not a fit"]


class Requirements(BaseModel):
    """What a job ad asks for. Produced by the extract_requirements tool."""

    job_title: str
    company: str | None = None
    location: str | None = Field(None, description="City or region of the office, as written in the ad")
    work_mode: WorkMode = Field(description="remote = fully remote, hybrid = some office days")
    min_hours_per_week: int | None = Field(
        None,
        description="Lowest weekly hours the employer accepts. '16-20 h' -> 16, full-time -> 40, not stated -> null",
    )
    german_level: GermanLevel = Field(
        description=(
            "none = German not mentioned; plus = German is nice to have; "
            "basic = basic German (A2-B1) required; fluent = good/very good/fluent/B2 or higher required"
        )
    )
    skills: list[str] = Field(description="Technical skills and tools the ad asks for, short names like 'Python'")


class ConstraintCheck(BaseModel):
    """Result of the check_constraints tool (plain Python rules, no LLM)."""

    location_ok: bool
    hours_ok: bool
    language_ok: bool
    red_flags: list[str] = []
    warnings: list[str] = []


class SkillComparison(BaseModel):
    """Result of the compare_skills tool."""

    matching: list[str]
    missing: list[str]
    cv_skills: list[str]


class ReportDraft(BaseModel):
    """What the LLM writes at the end. Code then checks it and turns it into a MatchReport."""

    summary: str = Field(description="Two or three honest sentences about the fit")
    matching_skills: list[str] = Field(description="Skills from the CV skill list that the job needs")
    missing_skills: list[str] = Field(description="Skills the job needs that are not in the CV")
    cv_version: CvVersion
    extra_red_flags: list[str] = Field(description="Other serious problems, e.g. 'Master students only'")
    tips: list[str] = Field(description="Concrete, honest application tips. Never suggest claiming skills the CV lacks")


class MatchReport(BaseModel):
    """The final, validated report shown to the user."""

    job_title: str
    company: str | None
    score: int = Field(ge=0, le=100)
    verdict: Verdict
    summary: str
    matching_skills: list[str]
    missing_skills: list[str]
    cv_version: CvVersion
    red_flags: list[str]
    warnings: list[str]
    tips: list[str]
    constraints: ConstraintCheck | None
    guardrail_removed: list[str] = Field(
        default=[], description="Skills the LLM claimed as matching that are NOT in the CV (removed by code)"
    )
