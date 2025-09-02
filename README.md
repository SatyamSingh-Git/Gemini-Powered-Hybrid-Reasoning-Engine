# Gemini‑Powered Hybrid Reasoning Engine
Harness Google’s Gemini models with a hybrid reasoning pipeline that blends multi‑step CoT, tool‑use, retrieval, and structured orchestration — all in a simple, production‑ready Python package.

<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white"></a>
  <a href="https://github.com/SatyamSingh-Git/Gemini-Powered-Hybrid-Reasoning-Engine/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/SatyamSingh-Git/Gemini-Powered-Hybrid-Reasoning-Engine?logo=github"></a>
  <a href="https://github.com/SatyamSingh-Git/Gemini-Powered-Hybrid-Reasoning-Engine/issues"><img alt="Issues" src="https://img.shields.io/github/issues/SatyamSingh-Git/Gemini-Powered-Hybrid-Reasoning-Engine?logo=github"></a>
</p>

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Local (Python)](#local-python)
  - [Docker](#docker)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
This repository implements a hybrid reasoning engine on top of Google’s Gemini models. The engine is designed for:
- multi‑step chain‑of‑thought reasoning,
- modular tool calling (RAG, web search, math, code execution),
- structured orchestration with guardrails,
- simple deployment via Docker.

It aims to give developers a clean starting point to build AI agents that are more reliable than single‑prompt LLM calls.

---

## Key Features
- Hybrid reasoning pipeline (decompose → retrieve/tools → reason → verify)
- Pluggable tools and skills
- Configurable safety and constraints
- Pythonic API and CLI entry-points
- Containerized deployment with a minimal Dockerfile

---

## Architecture
High‑level data flow:

```
User Prompt
   │
   ├─► Planner (task decomposition / CoT)
   │
   ├─► Tool Router ──► [RAG] [Web] [Math] [Code] …
   │                      │
   │                  Evidence
   │                      ▼
   ├─► Reasoner (Gemini) ─► Draft Answer
   │
   ├─► Verifier / Refiner (optional)
   │
   └─► Final Response
```

You can adapt each stage (planner, tools, reasoner, verifier) to your use case.

---

## Quick Start

### Prerequisites
- Python 3.10+
- A Google Gemini API key (e.g., via Google AI Studio)
- Git and (optionally) Docker

### Local (Python)
```bash
# 1) Clone
git clone https://github.com/SatyamSingh-Git/Gemini-Powered-Hybrid-Reasoning-Engine.git
cd Gemini-Powered-Hybrid-Reasoning-Engine

# 2) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Set environment variables
# Power the engine with your Gemini key
# macOS/Linux:
export GEMINI_API_KEY="your_api_key_here"
# Windows (PowerShell):
$Env:GEMINI_API_KEY="your_api_key_here"

# 5) Run
python app.py
```

### Docker
```bash
# Build the image
docker build -t gemini-hybrid-engine .

# Run the container (pass your API key into the container environment)
docker run --rm -e GEMINI_API_KEY="your_api_key_here" -p 8000:8000 gemini-hybrid-engine
```

- If the app serves an HTTP API, it will typically be available at http://localhost:8000.
- Adjust the port to match the app’s configuration in app.py.

---

## Configuration
The engine reads configuration from environment variables. Common settings:

- GEMINI_API_KEY: Your Google Gemini API key
- MODEL_NAME: Gemini model to use (e.g., gemini-1.5-pro)
- MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K: Generation controls
- TOOLS_ENABLED: Comma‑separated tool list (e.g., rag,web,math,code)

You can also store these in a local .env file (ensuring it is listed in .gitignore).

Example .env:
```bash
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-1.5-pro
TEMPERATURE=0.3
TOOLS_ENABLED=rag,math
```

---

## Usage

### Run the demo/test script
A convenience script is provided:
```bash
python test.py
```

This is useful to validate your key and the basic reasoning flow.

### Programmatic usage (example)
Below is a generic usage sketch. Adapt names to your actual modules if needed.

```python
# example.py
import os

# from src.engine import HybridReasoner  # example import if you expose a class

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key, "Missing GEMINI_API_KEY"

    prompt = "Plan a 3‑day Osaka itinerary with budget and travel times."
    # engine = HybridReasoner(api_key=api_key)  # Example
    # result = engine.run(prompt)
    # print(result)

if __name__ == "__main__":
    main()
```

Run:
```bash
python example.py
```

---

## Project Structure
```
.
├─ app.py                # Entry point / server or CLI launcher
├─ test.py               # Simple test/demo harness
├─ src/                  # Engine source code (pipelines, tools, utils)
├─ requirements.txt      # Python dependencies
├─ Dockerfile            # Container build
├─ .gitignore            # Ignore rules
└─ README.md             # This file
```

---

## Testing
```bash
# Recommended: run inside your virtual environment
python -m pytest -q
# or if tests are scripted in test.py:
python test.py
```

Consider adding pytest and richer test coverage as the engine evolves.

---

## Roadmap
- Add more built‑in tools (calendar, email, code execution sandbox)
- Add structured logs and tracing (e.g., OpenTelemetry)
- Add streaming responses
- Add guardrails and eval harnesses
- Provide FastAPI/Flask HTTP endpoints (if not already enabled)
- Publish as a pip package

Have ideas? Please open an issue or PR!

---

## Contributing
Contributions are very welcome!
1) Fork the repo
2) Create a feature branch
3) Commit with clear messages
4) Open a PR with context and examples

For larger changes, please open an issue first to discuss.

---

## License
No license file is currently included. If you intend to use or distribute this project, please add a LICENSE file (e.g., MIT, Apache‑2.0).

---

## Acknowledgments
- Google Gemini team and the broader open‑source community for tools, SDKs, and examples.
- Inspiration from hybrid/agentic reasoning patterns (planner‑executor, toolformer, RAG).

—

If this project helps you, consider giving it a star. It helps others find it too!