"""The 5 tools the agent can call.

Each tool is a plain Python function, so it can be unit-tested without an LLM.
`ToolBox` connects them to the agent: it runs a tool by name and remembers results.
`TOOL_SCHEMAS` describes the tools to the LLM (name, purpose, arguments as JSON schema).
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

import requests

from agent import config
from agent.models import ConstraintCheck, Requirements, SkillComparison

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CV = DATA_DIR / "my_cv.md"  # gitignored, never committed
SAMPLE_CV = DATA_DIR / "sample_cv.md"  # fictional CV that ships with the repo

MAX_AD_CHARS = 15_000

# Different spellings of the same skill. Keys and values are already normalised.
SKILL_ALIASES = {
    "sklearn": "scikitlearn",
    "postgres": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "golang": "go",
    "py": "python",
}


# ---------------------------------------------------------------- fetch_job_ad

class _TextExtractor(HTMLParser):
    """Collects visible text from HTML and skips <script>, <style> etc."""

    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def fetch_job_ad(url: str, get: Callable = requests.get) -> str:
    """Download a job page and return its readable text."""
    if not re.match(r"^https?://", url):
        raise ValueError("URL must start with http:// or https://")
    response = get(url, timeout=15, headers={"User-Agent": "job-match-agent/1.0"})
    response.raise_for_status()
    text = html_to_text(response.text)
    if len(text) < 200:
        raise ValueError("The page has almost no text (it may need JavaScript). Paste the ad text instead.")
    return text[:MAX_AD_CHARS]


# -------------------------------------------------------- extract_requirements

def extract_requirements(text: str, llm) -> Requirements:
    """Ask the LLM to turn the ad text into a validated Requirements object."""
    prompt = (
        "Extract the requirements from this job ad. Only use information that is in the ad.\n\n"
        f"JOB AD:\n{text[:MAX_AD_CHARS]}"
    )
    return llm.structured(prompt, Requirements)


# --------------------------------------------------------------------- read_cv

def read_cv(path: Path | None = None) -> str:
    """Load the CV. Uses data/my_cv.md if it exists, otherwise the fictional sample CV."""
    if path is None:
        path = REAL_CV if REAL_CV.exists() else SAMPLE_CV
    return Path(path).read_text(encoding="utf-8")


def parse_cv_skills(cv_text: str) -> list[str]:
    """Read the skill list from the '## Skills' section of the CV markdown.

    Lines look like '- Machine learning: Python, scikit-learn, PyTorch (basics)'.
    """
    match = re.search(r"^##\s*Skills\s*$(.*?)(?=^##\s|\Z)", cv_text, re.M | re.S | re.I)
    if not match:
        return []
    skills: list[str] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        line = line.lstrip("- ").split(":", 1)[-1]  # drop the category name
        for item in line.split(","):
            item = re.sub(r"\(.*?\)", "", item).strip()  # drop notes like "(basics)"
            if item and item not in skills:
                skills.append(item)
    return skills


# ----------------------------------------------------------- check_constraints

def check_constraints(
    location: str | None,
    work_mode: str,
    min_hours_per_week: int | None,
    german_level: str,
) -> ConstraintCheck:
    """Apply my job-search rules (see config.py). Pure Python, no LLM."""
    red_flags: list[str] = []
    warnings: list[str] = []

    # Location: fully remote is fine, otherwise the office must be in the allowed area.
    place = (location or "").lower()
    if work_mode == "remote":
        location_ok = True
    elif any(town in place for town in config.ALLOWED_LOCATIONS):
        location_ok = True
    elif not place:
        location_ok = True
        warnings.append("Location not stated in the ad – ask before applying.")
    else:
        location_ok = False
        red_flags.append(f"Office in {location} ({work_mode}) is outside the Deggendorf–Munich–Regensburg area.")

    # Hours: max 20 per week for a non-EU student.
    if min_hours_per_week is None:
        hours_ok = True
        warnings.append("Weekly hours not stated – confirm max 20 h/week is possible.")
    elif min_hours_per_week > config.MAX_HOURS_PER_WEEK:
        hours_ok = False
        red_flags.append(
            f"Needs at least {min_hours_per_week} h/week; the limit is {config.MAX_HOURS_PER_WEEK} h/week."
        )
    else:
        hours_ok = True

    # Language: English is fine, fluent German is not realistic yet.
    if german_level == "fluent":
        language_ok = False
        red_flags.append("Good or fluent German is required.")
    else:
        language_ok = True
        if german_level == "basic":
            warnings.append("Basic German is required – be honest about your level in the cover letter.")

    return ConstraintCheck(
        location_ok=location_ok,
        hours_ok=hours_ok,
        language_ok=language_ok,
        red_flags=red_flags,
        warnings=warnings,
    )


# --------------------------------------------------------------- compare_skills

def normalize_skill(skill: str) -> str:
    """'Scikit-Learn ' -> 'scikitlearn', 'k8s' -> 'kubernetes'."""
    key = re.sub(r"[^a-z0-9+#]", "", skill.lower())
    return SKILL_ALIASES.get(key, key)


def compare_skills(required_skills: list[str], cv_skills: list[str]) -> SkillComparison:
    """Split the required skills into matching (in the CV) and missing."""
    cv_by_key = {normalize_skill(s): s for s in cv_skills}
    matching, missing = [], []
    for skill in required_skills:
        key = normalize_skill(skill)
        if key in cv_by_key:
            if cv_by_key[key] not in matching:
                matching.append(cv_by_key[key])  # use the CV's spelling
        elif skill not in missing:
            missing.append(skill)
    return SkillComparison(matching=matching, missing=missing, cv_skills=cv_skills)


# --------------------------------------------------------------- tool registry

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "fetch_job_ad",
        "description": "Download a job ad from a URL. Only call this if the user gave a link.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The job ad URL"}},
            "required": ["url"],
        },
    },
    {
        "name": "extract_requirements",
        "description": (
            "Extract structured requirements (title, location, work mode, hours, German level, skills) "
            "from the current job ad. The ad text is already loaded, so no arguments are needed."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_cv",
        "description": "Read the candidate's CV (markdown) and the list of skills it contains.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "check_constraints",
        "description": "Check the job against the candidate's rules for location, weekly hours and German level.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Office location from the requirements"},
                "work_mode": {"type": "string", "enum": ["remote", "hybrid", "on-site", "unknown"]},
                "min_hours_per_week": {"type": "integer", "description": "Minimum weekly hours, omit if unknown"},
                "german_level": {"type": "string", "enum": ["none", "plus", "basic", "fluent"]},
            },
            "required": ["work_mode", "german_level"],
        },
    },
    {
        "name": "compare_skills",
        "description": "Compare the skills the job needs with the skills in the CV.",
        "parameters": {
            "type": "object",
            "properties": {
                "required_skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["required_skills"],
        },
    },
]


class ToolBox:
    """Runs tools by name for the agent and keeps the latest result of each tool.

    The job ad text is stored here, so the LLM does not have to copy thousands of
    characters into a tool argument (slow, expensive and error-prone).
    """

    def __init__(self, llm, cv_path: Path | None = None, get: Callable = requests.get):
        self.llm = llm
        self.get = get
        self.cv_text = read_cv(cv_path)
        self.cv_skills = parse_cv_skills(self.cv_text)
        self.job_text = ""
        self.results: dict[str, Any] = {}

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool. Errors are returned to the LLM instead of crashing the agent."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            result = handler(**args)
        except Exception as exc:  # the model sees the error and can try something else
            return {"error": f"{type(exc).__name__}: {exc}"}
        self.results[name] = result
        return result

    def _tool_fetch_job_ad(self, url: str) -> dict:
        self.job_text = fetch_job_ad(url, get=self.get)
        return {"chars": len(self.job_text), "text": self.job_text[:6000]}

    def _tool_extract_requirements(self) -> dict:
        if not self.job_text:
            raise ValueError("No job ad loaded yet. Call fetch_job_ad first.")
        return extract_requirements(self.job_text, self.llm).model_dump()

    def _tool_read_cv(self) -> dict:
        return {"cv": self.cv_text, "skills": self.cv_skills}

    def _tool_check_constraints(
        self, work_mode: str, german_level: str, location: str | None = None, min_hours_per_week: int | None = None
    ) -> dict:
        hours = int(min_hours_per_week) if min_hours_per_week is not None else None  # JSON may send 20.0
        return check_constraints(location, work_mode, hours, german_level).model_dump()

    def _tool_compare_skills(self, required_skills: list[str]) -> dict:
        return compare_skills(list(required_skills), self.cv_skills).model_dump()
