"""Streamlit UI: paste a job ad (or a link) and get a match report."""

import os

import streamlit as st
from dotenv import load_dotenv

from agent.agent import looks_like_url, run_agent
from agent.llm import GeminiClient

load_dotenv()

st.set_page_config(page_title="Job Match Agent", page_icon="🎯", layout="wide")
st.title("🎯 Job Match Agent")
st.caption(
    "An AI agent with tool calling: it reads a job ad, checks it against a CV and my job-search rules "
    "(location, max 20 h/week, German level) and returns a validated match report."
)


def get_api_key() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]  # Streamlit Community Cloud
    except Exception:
        return os.getenv("GEMINI_API_KEY")  # local .env


job_input = st.text_area("Job ad text or link", height=250, placeholder="Paste the job ad here, or a https:// link")

if st.button("Analyse", type="primary", disabled=not job_input.strip()):
    api_key = get_api_key()
    if not api_key:
        st.error("No GEMINI_API_KEY found. Add it to .env (local) or to the app secrets (cloud).")
        st.stop()

    label = "Fetching the page and analysing…" if looks_like_url(job_input) else "Analysing…"
    with st.spinner(label):
        try:
            result = run_agent(job_input, GeminiClient(api_key=api_key))
        except Exception as exc:
            st.error(f"The agent failed: {exc}")
            st.stop()

    report = result.report
    st.subheader(f"{report.job_title}" + (f" – {report.company}" if report.company else ""))

    col1, col2, col3 = st.columns(3)
    col1.metric("Match score", f"{report.score}/100")
    col2.metric("Verdict", report.verdict)
    col3.metric("Use CV version", report.cv_version)

    st.write(report.summary)

    for flag in report.red_flags:
        st.error(f"🚩 {flag}")
    for warning in report.warnings:
        st.warning(warning)

    left, right = st.columns(2)
    with left:
        st.markdown("**✅ Matching skills**")
        st.write(", ".join(report.matching_skills) or "–")
    with right:
        st.markdown("**❌ Missing skills**")
        st.write(", ".join(report.missing_skills) or "–")

    if report.guardrail_removed:
        st.info(
            "Guardrail: the model claimed these skills as matching, but they are not in the CV, "
            f"so they were removed: {', '.join(report.guardrail_removed)}"
        )

    st.markdown("**💡 Tips**")
    for tip in report.tips:
        st.markdown(f"- {tip}")

    with st.expander(f"Agent trace – {len(result.trace)} tool calls in {result.steps} steps"):
        if result.hit_step_limit:
            st.warning("The agent hit the step limit.")
        for item in result.trace:
            st.markdown(f"**Step {item['step']}: `{item['tool']}`**  args: `{item['args']}`")
            st.json(item["result"], expanded=False)

    with st.expander("Raw report (JSON)"):
        st.json(report.model_dump())
