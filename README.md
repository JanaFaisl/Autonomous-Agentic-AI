# Agentic AI — Requirements to Design

A **Streamlit** app that transforms a plain-language product idea into a full software specification — and a runnable starter codebase — through a sequential multi-agent pipeline. Six specialist AI agents — powered by **CrewAI** and the **Anthropic Claude API** — collaborate to produce structured requirements, database schema, UI/UX design, sprint plan, quality assurance strategy, and support documentation.

## Architecture

The system runs a sequential six-agent pipeline, each agent passing its output as structured JSON to the next:

```
Chat → Requirements → Database → Design → Planning → Quality → Support
```

Each agent uses **CrewAI** as the primary execution path, with a direct **Anthropic Messages API** call as a fallback if CrewAI times out, raises an exception, or returns content that fails JSON validation. The whole pipeline is coordinated by a `ProjectManagerAgent`, which decides execution order and can skip steps that already have valid output.

| Agent | Responsibility |
|---|---|
| `ChatAgent` | Conversational requirements gathering via MCQ-guided dialogue |
| `RequirementsAnalystAgent` | Converts conversation into structured requirements JSON |
| `DatabaseAgent` | Generates relational database schema (tables, columns, relationships) and DDL |
| `DesignAgent` | Produces UI/UX design (screens, components, navigation, color, typography) |
| `PlanningAgent` | Creates sprint plan and project milestones |
| `QualityAgent` | Generates test cases, risk analysis, and QA strategy |
| `SupportAgent` | Produces FAQ and support documentation |
| `ProjectManagerAgent` | Orchestrates pipeline execution order and artifact storage |

## Prerequisites

- **Python 3.10+**
- An **[Anthropic API key](https://console.anthropic.com/)**
- **CrewAI** installed (included in `requirements.txt`)
- **Node.js 22.5+** — only needed if you want to run the downloaded application scaffold (see [Downloadable output](#downloadable-output)), not to run the Streamlit app itself

## Setup

### 1. Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment variables

Create a **`.env`** file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Optional — override the default Claude model:

```env
DEFAULT_MODEL=claude-sonnet-4-5
```

See the [Anthropic model docs](https://docs.anthropic.com/en/docs/about-claude/models) for valid model IDs.

## Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Using the app

1. **Chat** — Describe your product idea. The assistant asks structured follow-up questions to gather requirements.
2. **Review** — Convert the conversation into a structured requirements JSON document. Confirm, edit, or download.
3. **Design** — Preview the generated UI/UX as a live, clickable device simulation (desktop or mobile), with real screen-to-screen navigation.
4. **Database** — Review the generated schema, optionally materialize it as real tables, and browse/export sample or live data.
5. **Run Pipeline** — Execute all remaining agents sequentially from the Review tab. Each tab populates as agents complete; per-step failures are reported without halting the rest of the pipeline.
6. **Export** — Download the full project — every generated spec plus a runnable application scaffold — as a single ZIP from the Review tab (see below). Individual JSON/CSV/DDL exports are also available from their respective tabs.

> **Note:** The full pipeline takes approximately 13 minutes under real API load. The Design Agent is the primary bottleneck (~400s+) due to the complexity of its output.

## Downloadable output

The **"Download project (.zip)"** button in the Review tab bundles everything generated so far into one archive:

- **Specs at the root**: `requirements.json`, `database_schema.json` + generated PostgreSQL/SQLite DDL, `design.json`, `plan.json`, `quality_report.json`, `support_package.json`.
- **A runnable scaffold under `app/`**, generated deterministically from the database schema and design (no extra LLM calls):
  - `app/backend/` — an Express server using Node's built-in **`node:sqlite`** module (no native compilation required), with full REST CRUD routes generated for every table in the schema.
  - `app/frontend/` — a React + Vite app with one page per generated screen, sharing the same CSS classes, icon logic, and screen-navigation matching as the live Design tab preview, so the scaffold's UI matches what was previewed in-app.

To run the scaffold after unzipping:

```bash
cd app/backend && npm install && npm start   # http://localhost:4000
cd app/frontend && npm install && npm run dev  # http://localhost:5173
```

It's a starting point, not a finished app — frontend screens render components as interactive placeholders, and backend routes are generic CRUD.

## Project layout

```
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env                          # Local secrets — do not commit
├── core/
│   ├── constants.py              # Model, timeout, token, and retry configuration
│   ├── llm.py                    # CrewAI LLM factory and availability check
│   ├── models.py                 # Pydantic schemas each agent's output is validated against
│   └── utils.py                  # Shared JSON parsing, icon resolution, error formatting
├── agents/
│   ├── chat_agent.py
│   ├── requirements_agent.py
│   ├── database_agent.py
│   ├── design_agent.py
│   ├── planning_agent.py
│   ├── quality_agent.py
│   ├── support_agent.py
│   └── project_manager_agent.py  # Orchestrates the full pipeline
├── db/
│   └── db_service.py             # Session/event/artifact persistence + domain table materialization
├── ui/
│   └── main_ui.py                # Streamlit UI, tab wiring, live design preview, all helpers
├── utils/
│   ├── io_suppression.py         # Suppresses CrewAI stderr noise
│   ├── project_export.py         # Builds the downloadable ZIP archives
│   └── code_scaffold.py          # Generates the runnable backend/frontend scaffold
└── tests/                        # pytest suite (see below)
```

## Running tests

```bash
pytest tests/ -v
```

The test suite includes:

| Test file | Coverage |
|---|---|
| `test_requirements_agent.py` | RequirementsAnalystAgent unit tests |
| `test_database_agent.py` | DatabaseAgent unit tests (schema generation + DDL for both dialects) |
| `test_chat_agent.py` | ChatAgent unit tests + MCQ format compliance |
| `test_db_service.py` | Persistence layer (sessions/events/artifacts, domain table creation) |
| `test_utils.py` | Utility and helper functions |
| `test_integration_real_llm.py` | Live CrewAI + Anthropic API calls (skipped without real key) |

Real LLM tests run only when `ANTHROPIC_API_KEY` is set to a valid key. All other tests use mocked API calls and run deterministically in a few seconds (88 tests as of this writing).

```bash
# Run only unit tests (no API key needed)
pytest tests/ -v --ignore=tests/test_integration_real_llm.py

# Run real LLM integration tests
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_integration_real_llm.py -v -s
```

## Configuration

Key values in `core/constants.py`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `claude-sonnet-4-5` | Claude model used by all agents |
| `DEFAULT_TEMPERATURE` | `0.2` | LLM temperature (low = more consistent) |
| `DEFAULT_MAX_TOKENS` | `16000` | Shared output token budget; sized for the most verbose agent (Design) so large JSON isn't truncated mid-generation |
| `MAX_EXECUTION_TIME` | `180` | CrewAI max execution time per agent (seconds), for the lighter agents |
| `DESIGN_MAX_EXECUTION_TIME` | `650` | Larger execution budget specifically for the Design Agent, which routinely takes longer |
| `MAX_RETRIES` | `2` | Retry attempts (with exponential backoff) before falling back to the direct API path |

The shared CrewAI `LLM` HTTP timeout is set in `core/llm.py` (`timeout=600`) to comfortably exceed the Design Agent's typical runtime.

## Troubleshooting

| Issue | What to try |
|---|---|
| **Auth errors** | Confirm `ANTHROPIC_API_KEY` in `.env` has no extra quotes or spaces |
| **Model 404** | Update `DEFAULT_MODEL` in `.env` to a valid model ID for your account |
| **`Your credit balance is too low`** | Add credits or upgrade your plan at [console.anthropic.com](https://console.anthropic.com) → Plans & Billing |
| **Design Agent timeout / "Connection issue detected"** | Increase `DESIGN_MAX_EXECUTION_TIME` in `core/constants.py` and `timeout` in `core/llm.py`; this is expected if Design's output exceeds the current budget |
| **`EOF while parsing` / JSON validation error on Design/Database output** | Usually means `DEFAULT_MAX_TOKENS` is too low for the model's output — raise it (check your model's real limit via the Anthropic API first) |
| **Streamlit session drop** | Set `server.maxUploadSize` and increase session timeout in `.streamlit/config.toml` |
| **`externally-managed-environment`** | Use a venv (see Setup above) |
| **CrewAI telemetry noise** | Already suppressed in `core/llm.py` via environment flags |
| **Downloaded zip won't open / contains plain text instead of a zip** | The download was intercepted or truncated by network/security software; try a different network and re-download |
| **Scaffold's `npm install` fails to build a native module** | Shouldn't happen — the generated backend intentionally uses Node's built-in `node:sqlite` instead of a native module like `better-sqlite3` specifically to avoid this |
| **Code changes not taking effect after editing source** | Streamlit's autoreload doesn't reliably pick up changes to deeply-imported modules (`core/`, `agents/`, `utils/`) — fully stop and restart `streamlit run app.py` |
