"""My job-search rules. Change these to use the agent for a different person."""

# On-site / hybrid is only OK in these places (Deggendorf – Munich – Regensburg area).
# Fully remote jobs inside Germany are always OK.
ALLOWED_LOCATIONS = [
    "deggendorf",
    "munich",
    "münchen",
    "muenchen",
    "regensburg",
    "landshut",
    "dingolfing",
    "straubing",
    "plattling",
    "freising",
    "erding",
    "neutraubling",
]

# Non-EU students may work at most 20 hours per week during the semester.
MAX_HOURS_PER_WEEK = 20

# The CV versions the agent can recommend.
CV_VERSIONS = [
    "01-ai-ml-engineering",
    "02-mlops-devops",
    "03-data-analytics",
    "04-software-development",
]

# Safety limit for the agent loop: stop after this many model turns.
MAX_AGENT_STEPS = 8
