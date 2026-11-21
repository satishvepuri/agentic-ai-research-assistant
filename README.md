
# Agentic AI Research Assistant

A local research assistant built with Python, LangGraph, Ollama, and Streamlit.

The application takes a research question, retrieves relevant web search results, organizes the evidence, and generates a structured report with source references.

I developed this project to explore workflow orchestration with LangGraph and the practical challenges of running language models locally.

## Tech Stack

- Python
- LangGraph
- LangChain Ollama
- Qwen3 4B
- DuckDuckGo Search (DDGS)
- Streamlit

## How It Works

The application uses a five-stage workflow:

1. **Planning** - Generates search queries from the research topic.
2. **Research** - Retrieves search results and removes duplicate URLs.
3. **Analysis** - Organizes the retrieved evidence.
4. **Review** - Checks source coverage and records search limitations.
5. **Report Generation** - Uses a local language model to generate a structured Markdown report.

LangGraph manages the execution order and shared workflow state.

The planning, analysis, and review stages use deterministic Python logic. The report generation stage uses Ollama.

## Project Structure

    agentic-ai-research-assistant/
    ├── app.py
    ├── main.py
    ├── research_workflow.py
    ├── requirements.txt
    ├── tests/
    │   └── test_workflow.py
    └── README.md

## Installation

### Prerequisites

- Python 3.10 or later
- Ollama installed locally
- Internet connection for web searches

### 1. Clone the repository

    git clone https://github.com/satishvepuri/agentic-ai-research-assistant.git
    cd agentic-ai-research-assistant

### 2. Create a virtual environment

    python -m venv venv

### 3. Install dependencies (Windows)

    .\venv\Scripts\python.exe -m pip install -r requirements.txt

### 4. Download the model

    ollama pull qwen3:4b

### 5. Run the application

    .\venv\Scripts\python.exe -m streamlit run app.py

Open the local URL displayed in the terminal.

## Testing

Run the workflow tests with:

    .\venv\Scripts\python.exe -m unittest discover -s tests -v

The tests use mock model and search responses, so they do not require a live Ollama model or web search.

## Design Decisions

**Local inference**

Ollama was selected to avoid dependency on paid model APIs and to experiment with local inference.

**Bounded execution**

The workflow performs one research pass instead of repeatedly searching and generating responses. This limits unnecessary model calls.

**Source transparency**

Generated reports include the links retrieved during web search. The application does not independently verify the full contents of those pages.

## Known Limitations

- Research quality depends on search-result snippets.
- Local model inference can be slow on CPU-only systems.
- Search requests may occasionally fail or return limited results.
- Reports may fall back to a structured summary when the model output does not match the expected format.
- The workflow is predefined rather than a fully autonomous multi-agent system.

## Future Improvements

- Retrieve and analyze full webpage content.
- Add source relevance scoring.
- Improve report generation reliability.
- Benchmark smaller local models.
- Add automated evaluation for report quality.
