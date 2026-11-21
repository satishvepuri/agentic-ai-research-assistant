"""Command-line entry point for the local research assistant."""
import argparse
from pathlib import Path

from research_workflow import run_research


def main() -> None:
    parser = argparse.ArgumentParser(description="Free local LangGraph Research Assistant")
    parser.add_argument("topic", nargs="?", default="Recent developments in agentic AI", help="Your research question")
    args = parser.parse_args()
    try:
        result = run_research(args.topic)
    except Exception as exc:
        print(f"\nResearch failed: {type(exc).__name__}: {exc}")
        print("Check that Ollama is open, qwen3:4b is installed, and your internet connection works.")
        raise SystemExit(1)
    report_path = Path(__file__).resolve().parent / "research_report.md"
    report_path.write_text(result["report"], encoding="utf-8")
    print("\n" + result["report"])
    print(f"\nSaved to: {report_path}")
    print(f"Sources: {len(result.get('sources', []))}; Search rounds: {result.get('searches_done', 0)}")


if __name__ == "__main__":
    main()
