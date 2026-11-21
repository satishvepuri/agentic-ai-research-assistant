"""Streamlit interface for the faster, one-pass local research assistant."""
import streamlit as st
from research_workflow import MODEL, run_research

st.set_page_config(page_title="Local AI Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Agentic AI Research Assistant")
st.caption(f"Local model: {MODEL} · LangGraph 5-stage workflow · DDGS search · $0 API fees")
st.info("Reports use search snippets, not independently verified full webpages. Review source links.")

# A form submits only on deliberate clicks, rather than other UI rerenders.
with st.form("research_form"):
    topic = st.text_input("What would you like to research?", value="Recent developments in agentic AI")
    submitted = st.form_submit_button("Research", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Enter a research question first.")
    elif st.session_state.get("research_running", False):
        st.warning("A research request is already running. Don't click again.")
    else:
        st.session_state["research_running"] = True
        status = st.empty()
        try:
            status.info("Starting research...")
            result = run_research(topic, progress=lambda message: status.info(message))
            st.session_state["research_result"] = result
            st.session_state["research_topic"] = topic
            status.success("Research finished.")
        except Exception as exc:
            status.error(f"Research failed: {type(exc).__name__}: {exc}")
        finally:
            st.session_state["research_running"] = False

if st.session_state.get("research_result"):
    result = st.session_state["research_result"]
    st.success(f"Report: {len(result.get('sources', []))} sources · 1 research pass")
    st.markdown(result["report"])
    st.download_button("Download report", data=result["report"], file_name="research_report.md", mime="text/markdown")
