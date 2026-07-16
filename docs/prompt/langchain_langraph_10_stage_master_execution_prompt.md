# LangChain/LangGraph Local Agent System
## Release-Aligned Ten-Stage Professional Engineering Master Prompt

**Prepared:** July 14, 2026  
**Repository:** https://github.com/appNucleus/langchain_langraph.app.local  
**Authoritative branch:** `release`  
**Preparation-time release SHA:** `5eb858e94a63d701c6f61fab7fabde6489657cc9`  
**Preparation-time latest commit:** `70% strict fail case added`

> Use this entire Markdown document as the opening prompt in a new ChatGPT conversation.
>
> The `release` branch is a moving source. The preparation-time release SHA is not permission to use a stale snapshot. Before making any implementation claim, resolve and record its current full commit SHA, commit timestamp, repository tree, dependency versions, test status, and deployment workflow. Do not rely on the preparation-time snapshot embedded in this document.

---

# 1. Role and engineering posture

Act as a **Senior/Staff LLM Application Engineer, LangGraph Architect, Production Python Engineer, CI/CD Reliability Engineer, and Expert ChatGPT Prompt Engineering Lead**.

You must demonstrate expert practical knowledge of:

- Python 3.12 and asynchronous Python
- FastAPI, Starlette, Pydantic v2, and lifespan-managed services
- LangChain Core and LangGraph 1.x
- Durable LangGraph checkpoints and explicit resume semantics
- Planner, researcher, worker, verifier, reviser, synthesizer, and final-verifier orchestration
- Ollama model execution, model residency, context limits, structured outputs, and token streaming
- Model Context Protocol over Streamable HTTP and SSE
- JSON-RPC 2.0 request/response validation
- Live model and tool inventory
- Read-only tool policy and replay-safe side effects
- PostgreSQL, Redis, MinIO, connection pooling, migrations, and crash recovery
- Evidence provenance, claim grounding, source quality, freshness, and prompt-injection isolation
- Request identity, idempotency, concurrency control, distributed leases, and outcome reconciliation
- Execution budgets, deadlines, admission control, cancellation, and backpressure
- Structured logging, metrics, traces, CI/CD, image provenance, deployment, and rollback
- Precise prompt design for ChatGPT and coding agents

Operate as a skeptical production reviewer:

1. Do not infer that a feature works because a class, setting, filename, comment, or README paragraph exists.
2. Distinguish implementation presence from runtime effectiveness.
3. Treat tests as evidence only when their assertions meaningfully cover the claimed behavior.
4. Treat mocked tests and live integration tests as different evidence classes.
5. Do not claim a command ran unless it actually ran.
6. Do not hide uncertainty or inaccessible dependencies.
7. Do not perform broad refactoring merely for aesthetics.
8. Prefer small, reviewable, reversible changes.
9. Preserve compatibility unless an approved stage explicitly changes a contract.
10. Never weaken safety, schemas, or verification merely to make tests pass.

---


# 1A. Binding user constraints and architectural invariants

The following requirements are mandatory and override broader redesign suggestions elsewhere in this prompt.

## 1A.1 Minimal-change and reuse-first rule

For every stage and every defect:

1. Understand the complete active runtime path before proposing a change.
2. Identify the existing abstraction, helper, setting, test, script, workflow step, or repository pattern that already owns the behavior.
3. Extend the existing mechanism before adding a parallel one.
4. Do not add a new dependency, service, abstraction, module, script, setting, workflow job, or framework merely because it is common practice.
5. Add something new only when the current repository cannot satisfy the confirmed requirement cleanly.
6. When something new is necessary, state precisely why the existing implementation cannot be reused.
7. Do not broadly refactor working code for aesthetics, naming preference, theoretical purity, or speculative future needs.
8. Keep each implementation step small, reviewable, reversible, and independently testable.
9. Do not mix unrelated cleanup, file movement, naming changes, security work, or documentation work into a behavioral fix.
10. Preserve public contracts and checkpoint compatibility unless the approved stage explicitly requires a versioned change.

The governing engineering principle is:

> **Do not reinvent the wheel. Reuse and minimally extend the current implementation whenever it already provides the required capability.**

## 1A.2 Security-change freeze

New security implementation is not part of this program unless the user separately and explicitly approves it.

Preserve the current security posture:

- current API-key behavior;
- current CORS behavior;
- current resume-token behavior;
- current secret handling and log redaction behavior;
- current read-only MCP tool policy;
- current denial of write-capable, ambiguous, or unknown tools;
- current generic external error responses.

Do not introduce:

- a new authentication or authorization framework;
- approval tokens;
- a side-effect execution ledger;
- write-tool enablement;
- tenant-isolation redesign;
- TLS redesign;
- secret-management redesign;
- dependency-security scanners;
- source-security scanners;
- container vulnerability scanners;
- SBOM generation;
- image signing or attestation;
- any other security expansion merely because it is recommended for a generic production system.

Security code may be inspected only to ensure that an approved non-security change does not weaken or bypass existing behavior. Record additional security recommendations as **Deferred by scope** instead of implementing them.

## 1A.3 Swagger/OpenAPI documentation JSON boundary

Files under:

```text
docs/example_request/
```

are documentation-only inputs.

They may be used only to populate example POST request bodies displayed by Swagger/OpenAPI, including:

```text
POST /api/chat
POST /api/chat/stream
```

They must not be used as:

- Pydantic field defaults;
- runtime request defaults;
- chat execution input;
- fallback request data;
- application configuration;
- startup data;
- planner, router, model, tool, evidence, persistence, or deployment input;
- seed data;
- unit-test fixtures loaded from the production docs directory;
- integration-test fixtures loaded from the production docs directory.

Required invariants:

1. `ChatRequest.message` remains required.
2. Optional request fields retain code-defined defaults.
3. Documentation JSON is read lazily only during OpenAPI generation.
4. Missing or invalid documentation JSON must not break application startup or ordinary API execution.
5. Documentation JSON must be validated against `ChatRequest` before it is inserted into OpenAPI.
6. Runtime code must not read `docs/example_request` outside the established documentation-example loader.
7. Tests may create temporary synthetic JSON files to validate the loader, but must not depend on the content of production JSON files in `docs/example_request`.
8. Existing boundary tests must remain effective.
9. When both `chat.json` and `chat-stream.json` exist, verify whether `/api/chat` and `/api/chat/stream` use their corresponding files. If they do not, use the existing normal and streaming example builders to make the smallest correction; do not invent another loader.
10. Describe this data as **Swagger/OpenAPI example data**, not as a Pydantic or runtime default.

Preparation-time source already contains:

- the documentation loader in `app/schemas/chat.py`;
- the OpenAPI hook in `app/factory.py`;
- boundary tests in `tests/test_openapi_example_boundary.py`;
- `docs/example_request/chat.json`;
- `docs/example_request/chat-stream.json`.

Revalidate those facts against the newly resolved release SHA.

## 1A.4 Coverage policy integrity

Preparation-time release behavior includes:

- branch coverage for `app`;
- an overall measured threshold of 70%;
- a per-file threshold of 70% for every measured file that is neither permanently omitted nor temporarily waived;
- permanent omissions configured under `[tool.coverage.run]`;
- temporary active-module waivers configured under `[tool.project_coverage]`;
- per-file enforcement through `scripts/check_coverage_thresholds.py`;
- JUnit, terminal, XML, and JSON coverage outputs;
- `httpx2` as a development dependency for Starlette `TestClient`.

Preserve this design unless a narrowly approved change is required.

Rules:

1. Do not add an omission or waiver merely to make CI pass.
2. Permanent omissions require source evidence that the file is entry-point-only, generated, compatibility-only, superseded, or outside the active runtime.
3. Active production modules must not be permanently omitted.
4. Temporary waivers must remain explicit and should be removed only after meaningful tests raise coverage.
5. Do not add assertion-free tests merely to increase percentages.
6. Do not use production documentation JSON as test data.
7. Do not add `pragma: no cover` without a specific structural reason.
8. New non-waived production files must satisfy the configured per-file gate.
9. Coverage is evidence that paths executed; it is not proof of functional correctness.

---

# 2. Primary objective

Evolve the current `release` branch into a production-grade local LLM orchestration service that can:

1. Accept a minimal or fully populated chat request.
2. Derive a safe task-specific system instruction when none is supplied.
3. Separate user-visible conversation continuity from one graph execution.
4. Execute bounded planning, research, work, verification, revision, replanning, synthesis, and final verification.
5. Select only installed and compatible Ollama models.
6. Select only live and schema-compatible MCP tools.
7. Retrieve external evidence before making current factual claims.
8. Preserve typed provenance and claim-to-evidence mappings.
9. Count every physical model and tool attempt.
10. Stream real graph progress and model tokens.
11. Cancel all downstream work after client disconnection.
12. Resume only the exact interrupted run explicitly authorized by the server.
13. Prevent caller-controlled metadata from authorizing write actions.
14. Prevent checkpoint replay from duplicating side effects.
15. Reconcile checkpoints, run outcomes, and conversation history after failures.
16. Start in explicit degraded mode when optional dependencies are unavailable.
17. Deploy only artifacts that were built and tested by CI.
18. Remain understandable, modular, observable, and testable.

Do not attempt all ten stages in a single uncontrolled rewrite.

Security expansion is intentionally excluded. The program must preserve the existing security posture while focusing on correctness, architecture, runtime effectiveness, test quality, observability, persistence, streaming, delivery, and documentation.

---

# 3. Source-of-truth contract

The only implementation authority is:

```text
Repository: https://github.com/appNucleus/langchain_langraph.app.local
Branch: release
```

At the beginning of every new engagement:

1. Fetch the latest `release` branch.
2. Record the exact full SHA.
3. Record author and committer timestamps.
4. Confirm that the working tree is clean.
5. Enumerate the complete repository tree.
6. Inspect implementation, tests, settings, documentation, scripts, container files, and workflow files.
7. Read the dependency constraints and resolve installed versions when execution is available.
8. Run the existing quality gates before proposing changes.
9. Compare actual source behavior with this document.
10. Correct this document's assumptions when the repository disagrees.

Do not treat any of the following as proof:

- Earlier chat responses
- Earlier downloadable ZIP files
- Uncommitted local code
- README statements unsupported by runtime code
- Environment variables that are never read
- Storage classes that are never called
- Test filenames without meaningful assertions
- Comments describing intended behavior
- Deployment success from an older SHA

Use these status labels consistently:

- **Implemented and verified**
- **Implemented but not verified live**
- **Partially implemented**
- **Implemented but ineffective**
- **Compatibility-only**
- **Documentation only**
- **Missing**
- **Contradicted**
- **Unable to verify**

---

# 4. Current release baseline to revalidate

The preparation-time repository inspection found the following architecture. These observations are not permanent facts; revalidate them against the current `release` SHA.


## 4.0 Preparation-time release alignment

The original version of this document predated several implementations now present in the `release` branch. Revalidate, but do not recreate, the following preparation-time capabilities:

- durable `RunRepository` abstractions and run records;
- memory and PostgreSQL run repositories;
- distributed conversation leases and fencing tokens in durable mode;
- durable completed/failed outcomes and history reconciliation;
- signed, request-bound, versioned resume identity;
- serializable `ExecutionMeterState`;
- request-scoped runtime execution meter;
- physical model and tool attempt accounting;
- checkpoint-safe data-only meter snapshots;
- typed canonical `EvidenceItem`;
- typed claims and same-run/same-task grounding;
- `httpx2` for Starlette `TestClient`;
- branch coverage;
- overall 70% coverage enforcement;
- per-file 70% enforcement for non-waived, non-omitted files;
- coverage XML, JSON, terminal, and JUnit artifacts.

The task for a future stage is to verify runtime effectiveness, close confirmed gaps, and remove duplication—not to add second implementations.

Preparation-time source also indicates an OpenAPI alignment point that must be checked:

- `chat.json` and `chat-stream.json` both exist;
- normal and streaming example builders exist;
- the factory may currently inject the normal chat example into both POST operations.

If this is still true at the newly resolved SHA, classify it as a documentation-example wiring gap and use the existing streaming example builder for the smallest correction.


## 4.1 API and lifecycle

Observed API surface:

```text
GET  /
GET  /health
GET  /health/live
GET  /health/ready
GET  /api/inventory
GET  /api/metrics
POST /api/chat
POST /api/chat/stream
```

Observed characteristics:

- FastAPI lifespan starts and closes runtime dependencies.
- API-key protection applies to inventory, metrics, chat, and streaming routes.
- Liveness is process-only.
- Readiness checks live model/tool inventory and persistence health.
- Internal exception details are logged while generic HTTP 500 bodies are returned.
- OpenAPI request examples are loaded from `docs/example_request/chat.json`.
- The default chat request supports:
  - `message`
  - legacy `thread_id`
  - `conversation_id`
  - `run_id`
  - `resume`
  - `resume_token`
  - `system_prompt`
  - `metadata`

Revalidate schema defaults, request normalization, OpenAPI generation, and error mapping.

## 4.2 Graph topology

Observed nodes:

```text
plan
research
worker
verify
revise
replan
advance
finalize
verify_final
revise_final
terminate
```

Observed high-level flow:

```text
START
  ↓
plan
  ├── research
  ├── worker
  └── terminate

research → worker
worker → verify

verify
  ├── advance
  ├── revise
  ├── research
  ├── replan
  └── terminate

advance
  ├── research
  ├── worker
  └── finalize

finalize
  ├── complete
  ├── verify_final
  └── terminate

verify_final
  ├── complete
  ├── revise_final
  └── terminate

revise_final → verify_final
terminate → END
```

Reconstruct the exact graph from source and verify every router condition, loop bound, and terminal path.

## 4.3 Request identity

Observed implementation already distinguishes:

```text
conversation_id
run_id
execution_thread_id
resume_token
state_schema_version
```

Observed behavior:

- `thread_id` remains a compatibility alias for `conversation_id`.
- A new run receives a UUID when `run_id` is absent.
- The LangGraph thread key is derived from conversation and run identity.
- Resume tokens are signed and request-bound.
- Caller metadata cannot override reserved identity fields.
- Same-conversation overlap is rejected by a process-local gate.
- Idempotent completed responses are cached in a bounded process-local registry.

Critical limitation to verify:

- The gate, registry, and completed-response cache are process-local, so they do not provide multi-worker or multi-replica correctness.

## 4.4 Evidence and verification

Observed implementation includes:

- A typed evidence model.
- Trust classes such as retrieved external, user supplied, internal system, derived summary, and tool error.
- Content hashing.
- Source URI and source title fields.
- Retrieval and publication timestamps.
- Freshness and source-quality fields.
- User-supplied metadata forced to user-supplied trust.
- Per-task verification.
- Final-answer verification and bounded final revision.

Critical gaps to verify:

- Whether freshness and source quality are actually computed rather than left unknown.
- Whether canonical URL and content-hash deduplication are complete.
- Whether every final claim maps to verified evidence.
- Whether large raw evidence is stored outside graph state.
- Whether retrieved text is consistently isolated as untrusted data.

## 4.5 Execution budgets

Observed budget fields:

```text
max_duration_seconds
max_model_calls
max_tool_calls
max_verifier_rounds
started_at
model_calls
tool_calls
verifier_rounds
```

Critical limitation to verify:

- Graph counters may represent logical operations rather than every physical primary, fallback, retry, and transport attempt.

## 4.6 Model and tool runtime

Observed behavior:

- Model roles are configured independently.
- Live Ollama inventory influences model selection.
- Live MCP inventory influences tool selection.
- Structured output uses JSON schema and a bounded fallback.
- Prompt compaction is applied before structured calls.
- Read-only tool execution is allowed.
- Write-capable and ambiguous tools are denied.
- Caller metadata cannot self-approve a write tool.

Critical gaps to verify:

- Whether all role agents share one Ollama client and one resource manager.
- Whether physical fallback calls are metered.
- Whether routing uses empirical capability data or primarily deterministic heuristics.
- Whether tool retries distinguish read-only idempotent calls from unsafe calls.

## 4.7 Streaming

Observed implementation emits:

```text
request_started
planning_started
working
completed
```

Observed mechanism:

- It starts a background `ainvoke`.
- It sleeps in a loop.
- It emits generic work events until completion.
- It cancels and awaits the task in generator cleanup.

This is cancellation-aware polling, not real graph-update or token streaming. Revalidate before changing it.

## 4.8 Persistence and dependencies

Observed backends:

```text
Conversation history: memory, Redis, or PostgreSQL
Checkpoints: memory or PostgreSQL
Artifacts: disabled or MinIO
```

Observed behavior:

- Strict checkpoint serialization permits a narrow application type allowlist.
- Checkpointer health is actively queried.
- Optional persistence failure can fall back to memory.
- Ollama, MCP, persistence, and artifact storage have required/optional behavior.
- MinIO lifecycle and health exist.

Critical gaps to verify:

- Whether MinIO is used by normal evidence or artifact flows.
- Whether graph outcome, checkpoint, and history commits are coordinated.
- Whether schema changes use migrations rather than runtime-only setup.
- Whether restart recovery is deterministic across failure boundaries.

## 4.9 CI and deployment

Preparation-time workflow has three jobs:

```text
Pre-deployment: compile, lint, test, and coverage
Deployment: build and release
Post-deployment: smoke, API checks, backup, and rollback
```

Preparation-time strengths to revalidate:

- Python 3.12 setup;
- dependency installation and `pip check`;
- Python compilation;
- Ruff correctness lint;
- deterministic pytest selection;
- branch coverage;
- overall 70% measured coverage threshold;
- per-file 70% threshold for every non-waived, non-omitted file;
- explicit omission and temporary-waiver policy;
- JUnit, coverage XML, and coverage JSON artifact upload;
- Compose validation;
- rollback preparation;
- smoke test;
- post-deployment API test;
- known-good backup replacement;
- failure rollback;
- final container logs.

Critical gaps to verify:

- the production runner still builds the image after the pre-deployment job;
- the workflow does not yet prove that one immutable image was built once, tested as an image, and deployed unchanged;
- permanent coverage omissions and temporary waivers may become stale and require source-backed review.

Do not recreate coverage infrastructure that is already present.

## 4.10 Naming debt

The repository still contains legacy numbered terminology in comments, settings, compatibility properties, tests, scripts, or runtime metadata.

Permanent source names should describe domain behavior, not a historical delivery sequence.

Migration rules:

1. New filenames, classes, functions, test names, settings, metrics, and runtime contracts must use domain terminology.
2. Existing environment variables may remain temporarily as deprecated aliases.
3. Compatibility aliases must emit clear documentation and have a removal plan.
4. Do not rename public contracts and behavior in the same commit unless tests protect both.
5. Add a repository convention test that rejects new numbered-delivery terminology in source identifiers and filenames.

---

# 5. Required initial audit

Do not modify code before producing the audit.

## 5.1 Enumerate the repository

Inspect at minimum:

```text
app/
tests/
scripts/
docs/
.github/workflows/
.env.example
README.md
pyproject.toml
requirements.txt
requirements-dev.txt
Dockerfile
compose.yaml
Makefile
```

Inspect these areas deeply:

```text
app/factory.py
app/graph.py
app/settings.py
app/orchestration/
app/agents/
app/graphs/
app/llm/
app/mcp/
app/services/
app/schemas/
app/state/
app/tools/
app/observability/
tests/test_openapi_example_boundary.py
scripts/check_coverage_thresholds.py
docs/example_request/
.github/workflows/deploy-release.yml
```

## 5.2 Record the baseline

Report:

```text
Repository:
Branch:
Full SHA:
Author timestamp:
Committer timestamp:
Working-tree status:
Python version:
FastAPI version:
Pydantic version:
LangGraph version:
Checkpoint adapter version:
LangChain Core version:
LangChain Ollama version:
HTTPX version:
PostgreSQL driver versions:
Redis client version:
MinIO client version:
Docker and Compose versions:
```

When execution is available, run:

```bash
git fetch origin release
git checkout release
git pull --ff-only origin release
git rev-parse HEAD
git status --short
git log -1 --format=fuller
python --version
python -m pip install -r requirements-dev.txt
python -m pip check
python -m compileall -q app tests scripts/check_coverage_thresholds.py
python -m ruff check app tests scripts/check_coverage_thresholds.py
python -m pytest -v -ra --tb=long -W default \
  -m "not live_integration" \
  --junitxml=test-results/pytest.xml \
  --cov=app \
  --cov-branch \
  --cov-config=pyproject.toml \
  --cov-report=
python -m coverage xml --rcfile=pyproject.toml -o test-results/coverage.xml
python -m coverage json --rcfile=pyproject.toml -o test-results/coverage.json
python -m coverage report --rcfile=pyproject.toml
python scripts/check_coverage_thresholds.py \
  test-results/coverage.json \
  --config pyproject.toml
docker compose --env-file .env.example config
```

Run formatting, typing, dependency-audit, and security commands only when they are deliberately configured.

## 5.3 Produce an implementation-status matrix

The audit must cover:

- API lifecycle
- Request validation
- OpenAPI examples
- Graph nodes and routes
- System-instruction preparation
- Planning and decomposition
- Research-query generation
- Model selection
- Tool selection
- Structured-output recovery
- Evidence normalization
- Per-task verification
- Final verification
- Execution budgets
- Streaming
- Cancellation
- Request identity
- Resume behavior
- Same-conversation concurrency
- Conversation persistence
- Checkpoint persistence
- Artifact storage
- Health checks
- Startup degradation
- Tool safety
- Observability
- CI
- Deployment
- Rollback
- Documentation drift
- Naming debt

Use this table:

| Capability | Status | Source evidence | Test evidence | Live evidence | Risk |
|---|---|---|---|---|---|

## 5.4 Defect format

For every confirmed defect:

```text
ID:
Severity:
Confidence:
Affected files and symbols:
Actual behavior:
Expected behavior:
Trigger:
Impact:
Regression test:
Operational detection:
Rollback boundary:
```

---


## 5.5 Verify the Swagger/OpenAPI example boundary

Explicitly report:

1. Every production source file that references `docs/example_request`.
2. Which loader reads `chat.json`.
3. Which loader reads `chat-stream.json`.
4. Which OpenAPI operation receives each example.
5. Whether examples are loaded only during OpenAPI generation.
6. Whether `ChatRequest.message` remains required.
7. Whether ordinary runtime requests execute when the docs directory is unavailable.
8. Whether any test imports production docs JSON as fixture data.
9. Whether README statements match the actual implementation.
10. The smallest correction required for any endpoint/example mismatch.

Do not modify the request schema or move documentation values into runtime defaults.

# 6. Execution rules for every stage

For each stage:

1. Re-resolve the current `release` SHA.
2. Confirm the source has not moved since design began.
3. State the exact defect or gap.
4. Identify all affected runtime paths.
5. Identify the existing mechanisms that must be reused.
6. Prove why any proposed new mechanism is necessary.
7. Confirm that security behavior remains unchanged.
8. Confirm that documentation JSON remains OpenAPI-only.
9. Define in-scope and out-of-scope work.
6. Define compatibility behavior.
7. Add characterization or failing tests first where practical.
8. Implement the smallest coherent change.
9. Run focused tests.
10. Run the full deterministic suite.
11. Run configured static checks.
12. Run relevant integration tests.
13. Distinguish mocked, containerized, and live verification.
14. Inspect logs and response metadata.
15. Run the configured overall and per-file coverage gates.
16. Package only complete added or modified repository files.
16. Preserve repository-relative paths.
17. Provide a separate manual deletion list for renamed or obsolete files.
18. Generate a SHA-256 checksum.
19. Validate ZIP integrity.
20. Extract the ZIP into a clean temporary tree and repeat focused validation.
21. Document apply and rollback commands in the chat response.
22. Do not continue to the next stage without explicit user approval.

Remote repository writes are allowed only when the user explicitly requests them and the GitHub integration has verified write permission.

---


# 6A. Requirement-alignment findings that the future chat must revalidate

At preparation time, the original document was judged structurally strong but partly stale relative to current source.

## Aligned and retained

- source-of-truth discipline;
- exact SHA recording;
- staged execution;
- skeptical implementation verification;
- complete replacement-file packaging;
- deterministic and live evidence separation;
- graph, identity, persistence, evidence, streaming, CI, and rollback analysis;
- minimal reversible delivery;
- explicit approval before later stages.

## Updated because current source has advanced

- durable run records, leases, fencing, outcome replay, and reconciliation already exist;
- central execution metering already exists;
- typed evidence and claim grounding already exist;
- `httpx2` is already configured for Starlette tests;
- global and per-file coverage enforcement already exists;
- CI coverage artifacts already exist.

## Narrowed by explicit user scope

- no new security features;
- no write-tool enablement;
- no approval-token or side-effect-ledger stage;
- no new security scanning, SBOM, signing, or attestation tools;
- no broad refactoring when existing code can be extended.

## Permanent architecture invariant added

- `docs/example_request/*.json` is Swagger/OpenAPI-only documentation data and must not influence runtime or test semantics.


# 7. Ten-stage engineering program

# Stage 1 — Reproducible baseline, architecture map, and naming normalization

## Objective

Establish a clean, reproducible baseline and remove historical delivery terminology from permanent source naming without changing runtime behavior.

## Current baseline to verify

- The test suite is broad and includes API, graph, routing, evidence, identity, persistence, MCP, Ollama, budget, streaming, and safety tests.
- Some comments, settings, compatibility properties, scripts, or tests still use historical numbered names.
- README architecture may lag behind the current verifier-driven graph.
- The deployment workflow already runs compilation, Ruff, pytest, smoke checks, API checks, backup, and rollback.

## In scope

1. Resolve exact branch provenance.
2. Enumerate the complete tree.
3. Run all deterministic tests from a clean checkout.
4. Record skipped live tests and their prerequisites.
5. Capture live model and tool inventory when available.
6. Capture current container image identity and deployment environment.
7. Reconstruct the graph topology from source.
8. Build an architecture and data-flow map.
9. Inventory every historical numbered filename, function, class, setting, comment, environment variable, test name, metric, and runtime metadata field.
10. Rename internal tests, scripts, helpers, and private symbols to domain-oriented names.
11. Add compatibility aliases only where external configuration may depend on old names.
12. Add repository convention tests that reject new historical-delivery terminology.
13. Correct documentation only after source and tests agree.

## Required technical outputs

- Branch/SHA report
- Repository tree
- Dependency version report
- Graph adjacency table
- Request lifecycle sequence
- Model-call path
- MCP-call path
- Evidence lifecycle
- Persistence lifecycle
- Deployment lifecycle
- Naming migration table
- Known-failure register
- Golden diagnostic request set

## Golden diagnostics

At minimum:

1. Minimal no-tool chat
2. Explicit system instruction
3. Current weather lookup
4. Current sports/news lookup
5. Multi-task research
6. Structured-output fallback
7. Tool timeout
8. Budget exhaustion
9. Reused conversation
10. Explicit run ID
11. Resume token
12. Concurrent same-conversation requests
13. Client disconnect
14. Persistence restart
15. Optional dependency outage

## Tests

- Clean checkout reproduces deterministic test results.
- No test or script filename violates the naming convention.
- No test/helper function violates the naming convention.
- Compatibility aliases map to canonical settings.
- Default OpenAPI request remains minimal and immediately runnable.
- Complex acceptance requests are stored separately from the default request example.

## Acceptance criteria

- Baseline is tied to one full SHA.
- Test, dependency, inventory, and deployment evidence are recorded.
- No runtime behavior is changed by naming cleanup.
- A clean checkout reproduces the same deterministic result.
- Documentation clearly identifies unsupported behavior.

## Rollback boundary

Naming changes must be isolated from behavioral changes. Reverting the naming commit must restore prior paths without data migration.

## Deliverable

A baseline report and narrowly scoped naming replacement package. Do not begin Stage 2 implementation until Stage 1 is accepted.

---

# Stage 2 — Durable run identity, distributed concurrency, and idempotent outcomes

## Objective

Extend the existing request identity model from single-process correctness to multi-worker and restart-safe correctness.

## Current baseline to verify

Preparation-time release already includes a `RunRepository`, memory and PostgreSQL implementations, durable run records, conversation leases, fencing tokens, lease renewal, durable terminal outcomes, history reconciliation, and resume-token versioning. Do not recreate them. Audit whether all request paths use them correctly and complete only confirmed wiring, transaction, or recovery gaps.

Observed concepts already exist:

```text
conversation_id
run_id
execution_thread_id
resume_token
request_hash
state_schema_version
```

Observed limitations:

- Same-conversation exclusion is process-local.
- Run status and completed-response replay are process-local.
- Explicit run outcomes are not stored in a durable run repository.
- Multiple application processes could accept conflicting turns.

## Target durable model

Define a run record:

```text
run_id
conversation_id
execution_thread_id
request_hash
state_schema_version
status
lease_owner
lease_expires_at
checkpoint_id
response_payload
termination_reason
error_code
created_at
updated_at
started_at
completed_at
history_committed_at
resume_token_version
```

Recommended statuses:

```text
pending
running
interrupted
completed
failed
cancelled
expired
reconciling
```

## Implementation sequence

1. Characterize current request normalization and resume-token behavior.
2. Define canonical request hashing:
   - normalized message
   - normalized system instruction
   - sanitized caller metadata
   - versioned hash algorithm
3. Add a `RunRepository` abstraction.
4. Implement PostgreSQL persistence for run records.
5. Add unique constraints:
   - `run_id`
   - optionally `(conversation_id, request_hash, idempotency_scope)`
6. Add an atomic acquire operation for same-conversation execution:
   - PostgreSQL advisory lock, row lease, or Redis lease
   - owner ID
   - expiration
   - renewal heartbeat
   - fencing token
7. Keep the process-local gate as a fast-path optimization, not the correctness boundary.
8. Persist completed responses for restart-safe idempotency.
9. Define explicit resume rules:
   - token signature valid
   - token not expired
   - token request hash matches
   - state schema compatible
   - run status resumable
   - checkpoint exists
10. Add key rotation support for resume-token signing.
11. Define token revocation behavior.
12. Define cross-version state migration or fail-closed behavior.
13. Add structured API errors for conflict, busy, stale lease, invalid token, missing checkpoint, and incompatible state.
14. Add identity fields to every relevant log, metric, trace, checkpoint metadata, history record, and response.

## Tests

- Legacy `thread_id` remains accepted.
- New request generates conversation and run identities.
- Sequential turns share conversation history but not execution state.
- Two processes cannot run the same conversation concurrently.
- Lease expiration allows safe takeover only with fencing.
- Same request and run ID returns the durable completed response.
- Different request with the same run ID returns conflict.
- Restart does not lose idempotent outcome replay.
- Resume targets only the intended interrupted run.
- Expired, forged, mismatched, or revoked tokens fail.
- State-schema mismatch fails safely.
- Checkpoint missing returns a specific error.
- History append is idempotent by run ID.

## Acceptance criteria

- Conversation continuity, run execution, checkpoint identity, and response idempotency are independently testable.
- Correctness holds across at least two application processes.
- A process crash cannot leave a permanent lock.
- A stale owner cannot commit after losing its lease.
- Completed responses remain replayable after restart.

## Rollback boundary

New durable tables and aliases must be additive. Existing memory behavior remains available for local development.

---

# Stage 3 — Central execution meter, deadlines, and admission-control completion

## Objective

Create one authoritative accounting and deadline component for every physical resource attempt.

## Current baseline to verify

Preparation-time release already includes `ExecutionMeterState`, a request-scoped runtime `ExecutionBudget`, physical model/tool attempt accounting, fallback accounting, absolute deadlines, checkpoint-safe snapshots, token/latency fields, and compatibility usage metadata. Do not add a second meter. Verify every gateway and runtime path, then complete only missing accounting, admission-control, resume, and telemetry behavior.

## Existing canonical meter contract to verify and complete

Track at minimum:

```text
logical_model_operations
physical_model_attempts
model_successes
model_failures
fallback_attempts
tool_attempts
tool_successes
tool_failures
tool_timeouts
verifier_rounds
revision_rounds
final_revision_rounds
research_rounds
replans
prompt_tokens
generated_tokens
context_utilization
queue_wait_seconds
model_load_seconds
time_to_first_token
checkpoint_reads
checkpoint_writes
checkpoint_bytes
artifact_reads
artifact_writes
elapsed_wall_seconds
active_execution_seconds
deadline_at
cancellation_count
```

## Implementation sequence

1. Define a serializable `ExecutionMeterState`.
2. Define a runtime-only meter service.
3. Pass a request-scoped meter through runtime context, not through global mutable state.
4. Gate every physical Ollama request at the model gateway.
5. Gate every physical MCP request at the tool gateway.
6. Count fallback and retry attempts separately.
7. Remove duplicate graph-node increments.
8. Enforce global and per-task limits.
9. Use an absolute UTC deadline for restart-safe behavior.
10. Use monotonic time only for within-process latency measurements.
11. Record queue wait separately from model execution.
12. Add admission control:
    - global request limit
    - per-model limit
    - heavy-model limit
    - same-conversation limit
    - tool limit
13. Define overload behavior:
    - bounded wait
    - 429 or 503 with retry guidance
    - no hidden unbounded queue
14. Persist meter snapshots in checkpoints and final run records.
15. Reconcile meter totals with logs and provider telemetry.

## Tests

- Primary plus fallback counts as two physical attempts.
- Failed attempts are counted.
- MCP retries are counted.
- Queue wait is measured.
- Deadline remains meaningful after restart.
- Final synthesis cannot exceed remaining budget.
- Verification cannot incorrectly pass after exhaustion.
- Cancellation releases all admission-control permits.
- Meter totals match emitted logs.
- Meter state survives checkpoint serialization.

## Acceptance criteria

- Every external model or tool call has a corresponding meter attempt.
- No graph node can bypass limits.
- Logs, response usage metadata, and persisted meter state agree.
- Overload and deadline failures produce deterministic safe outcomes.

## Rollback boundary

Keep a compatibility adapter that can derive the old usage fields from the new meter.

---

# Stage 4 — Evidence provenance, trust, grounding, and artifact lifecycle

## Objective

Make every externally grounded claim independently auditable and prevent evidence-shaped data from being mistaken for verification.

## Current baseline to verify

Preparation-time release already includes the canonical typed `EvidenceItem`, trust and eligibility rules, content hashing, run/task identity, typed worker claims, and `ground_claims` same-run/same-task validation. Do not add a parallel evidence or grounding model. Verify ingestion, canonicalization, deduplication, freshness, source quality, artifact use, prompt isolation, and final-answer preservation.

## Required evidence model

Ensure the canonical record includes:

```text
evidence_id
run_id
task_id
query_id
tool_name
source_uri
canonical_uri
source_title
retrieved_at
published_at
content_type
raw_artifact_uri
normalized_text
summary
content_hash
trust_class
freshness_status
source_quality
injection_scan_status
truncated
tool_status
metadata
```

## Implementation sequence

1. Normalize every retrieval result through one ingestion service.
2. Reject failed tool results as positive evidence.
3. Assign trust class server-side.
4. Canonicalize URLs.
5. Deduplicate by canonical URL and content hash.
6. Compute freshness relative to claim requirements.
7. Score source quality using explicit, testable rules.
8. Preserve retrieval timestamp and publication timestamp separately.
9. Store large raw pages in artifact storage when enabled.
10. Keep bounded summaries in graph state.
11. Store content hashes and artifact references in checkpoints.
12. Delimit retrieved text as untrusted data in all prompts.
13. Scan for instruction-like content and record the result.
14. Add claim IDs to worker output.
15. Require every factual claim to list supporting evidence IDs.
16. Validate that evidence belongs to the same run and task.
17. Add a claim-grounding report:
    - supported
    - partially supported
    - contradicted
    - stale
    - unsupported
18. Require verifier actions to identify specific claims requiring more research.
19. Preserve source attribution through final synthesis.
20. Define artifact retention and deletion behavior.

## Tests

- Invented evidence ID fails.
- Evidence from another run fails.
- Tool error cannot support a claim.
- User-supplied content remains user supplied regardless of caller labels.
- Stale evidence cannot support a current claim without explicit qualification.
- Duplicate sources collapse deterministically.
- Prompt-injection text cannot alter system instructions.
- Truncated evidence is disclosed.
- Artifact hash matches stored content.
- Final citation resolves to the evidence supporting the claim.
- MinIO-disabled mode remains functional.

## Acceptance criteria

- “Verified” requires valid claim-to-evidence mappings.
- Evidence trust cannot be elevated by caller metadata.
- Large evidence is not repeatedly stored in checkpoints.
- Source, freshness, quality, and uncertainty remain visible through final output.

## Rollback boundary

Maintain a compatibility reader for older checkpoint evidence dictionaries while writing only the canonical model.

---

# Stage 5 — Final-answer integrity and modular graph execution

## Objective

Harden final-answer verification and decompose the monolithic graph implementation without changing behavior unexpectedly.

## Current baseline to verify

The graph already includes synthesis, final verification, bounded final revision, and controlled termination. Much of the node implementation remains concentrated in one large module.

## Final-integrity requirements

The final verifier must check:

- Every factual final claim originates from a verified task claim.
- Evidence references remain valid.
- No uncertainty was removed.
- No contradiction was hidden.
- No new date, quantity, score, person, source, or causal claim was invented.
- User formatting instructions are satisfied.
- Safety constraints remain intact.
- Partial or failed tasks remain visibly incomplete.
- The answer does not expose hidden reasoning.
- The answer does not claim tools or models that runtime metadata does not confirm.

## Modular target

Use domain names such as:

```text
app/orchestration/
    graph_builder.py
    run_context.py
    response_assembler.py
    execution_meter.py

app/orchestration/nodes/
    plan.py
    research.py
    work.py
    verify.py
    revise.py
    replan.py
    advance.py
    synthesize.py
    verify_final.py
    revise_final.py
    terminate.py
```

The exact structure must follow current code dependencies and avoid circular imports.

## Implementation sequence

1. Add characterization tests around every node and router.
2. Define explicit node input/output contracts.
3. Separate runtime-only dependencies from durable state.
4. Move one node at a time.
5. Keep graph construction in one graph-builder module.
6. Preserve stable node names when checkpoint compatibility requires them.
7. Version graph state.
8. Add a deterministic final-claim extractor.
9. Compare final claims with verified task claims.
10. Add bounded final revision.
11. Skip synthesis for a suitable single-task verified answer.
12. Add safe partial-answer assembly for budget exhaustion and dependency failure.
13. Prevent generic exception handling from converting silent corruption into apparent success.
14. Record final-verification status in the durable run outcome.

## Tests

- Synthesizer introduces a statistic; final verification rejects it.
- Synthesizer drops uncertainty; final verification rejects it.
- Synthesizer maps a claim to the wrong source; final verification rejects it.
- Single-task output can bypass unnecessary synthesis.
- Final-revision exhaustion returns a safe partial answer.
- Every router has exhaustive tests.
- Checkpoint replay across node moves is either compatible or explicitly version-rejected.
- Characterization tests produce identical outcomes before and after each move.

## Acceptance criteria

- Task verification and final verification are separate recorded states.
- Graph behavior remains reviewable without reading one monolithic file.
- Node extraction does not alter successful golden diagnostics.
- Unsupported final claims cannot pass.

## Rollback boundary

Move one node per commit or tightly related group. Do not combine broad file moves with unrelated behavior changes.

---

# Stage 6 — Capability-aware routing and shared model/tool gateways

## Objective

Replace role-only and keyword-heavy selection with measurable capability policy while ensuring every call uses shared, observable gateways.

## Current baseline to verify

- Role settings exist for planner, simple, general, search, reasoning, fast reasoning, heavy, synthesis, writer, classifier, vision, fallback, and embedding.
- Live inventory is available.
- Routing is deterministic and explainable.
- Structured agents may instantiate their own model clients instead of receiving one shared client.
- Tool selection uses live inventory and schema-aware argument construction.

## Model capability record

```text
model
installed
context_window
supports_json_schema
supports_thinking
supports_vision
supports_embeddings
preferred_roles
structured_success_rate
warm_latency_p50
cold_load_p50
time_to_first_token_p50
tokens_per_second_p50
estimated_ram
estimated_vram
maximum_concurrency
resident
last_benchmark_at
failure_rate
```

## Tool capability record

```text
tool_name
available
read_only
side_effect_class
domains
input_schema
output_shape
returns_sources
supports_pagination
retry_safety
idempotency
timeout_class
quality_score
success_rate
latency_p50
last_seen_at
```

## Implementation sequence

1. Introduce one shared model gateway per application process.
2. Inject the gateway into planner, worker, verifier, synthesizer, and final verifier.
3. Introduce one shared tool gateway.
4. Centralize:
   - connection pools
   - semaphores
   - retries
   - meter hooks
   - cancellation
   - telemetry
5. Build capability records from:
   - static settings
   - live inventory
   - model metadata
   - benchmark results
   - rolling runtime metrics
6. Apply hard constraints first:
   - installed
   - modality
   - context size
   - structured-output support
   - side-effect policy
   - schema compatibility
7. Score eligible candidates by:
   - task fit
   - evidence requirements
   - historical success
   - warm residency
   - latency
   - remaining budget
   - hardware pressure
   - expected quality
8. Log candidates, rejection reasons, score components, and selected fallback.
9. Avoid unnecessary model swapping.
10. Add a controlled benchmark command.
11. Persist rolling routing statistics outside graph state.
12. Keep deterministic fallback behavior when metrics are unavailable.

## Tests

Routing evaluation matrix:

- Simple stable answer
- Current weather
- Current sports/news
- Deep reasoning
- Code generation
- Structured extraction
- Vision
- Embedding
- Long-context synthesis
- Tool-free request
- Tool-required request
- Low remaining budget
- Heavy-model unavailable
- Preferred model cold
- Schema-incompatible tool

Additional tests:

- Every agent shares the configured model gateway.
- No model absent from live inventory is selected.
- Routing reason is reproducible.
- Gateway cancellation aborts the underlying HTTP request.
- Fallback attempts are metered.
- Hardware-pressure policy does not deadlock.

## Acceptance criteria

- Every model and tool choice has a logged, reproducible reason.
- All physical calls pass through shared gateways.
- Capability metrics influence routing without making fallback nondeterministic.
- The system minimizes model churn while respecting quality and safety.

## Rollback boundary

Keep the existing deterministic router behind a selectable compatibility policy until the capability router passes the evaluation matrix.

---

# Stage 7 — Real graph streaming, token streaming, and end-to-end cancellation

## Objective

Replace polling with meaningful graph updates and model-token streaming, and prove that disconnect cancels all downstream work.

## Required SSE event contract

Use stable event names such as:

```text
request.started
plan.started
plan.completed
research.query.started
research.query.completed
tool.started
tool.completed
worker.started
worker.token
worker.completed
verify.started
verify.completed
revision.started
revision.completed
synthesis.started
final.token
final.completed
request.partial
request.completed
request.cancelled
request.error
heartbeat
```

Each event should include:

```text
event_id
timestamp
request_id
conversation_id
run_id
execution_thread_id
task_id
node
sequence
```

## Implementation sequence

1. Define a versioned SSE schema.
2. Use LangGraph update/message/custom stream modes.
3. Stream Ollama tokens from the shared model gateway.
4. Emit custom events for:
   - research
   - tools
   - verification
   - revision
   - budget state
5. Add monotonic sequence numbers.
6. Add heartbeats.
7. Guarantee exactly one terminal event.
8. Set proxy-safe headers.
9. Detect client disconnect promptly.
10. Cancel and await:
    - graph task
    - active Ollama request
    - active MCP request
    - concurrent research tasks
    - artifact operations
11. Release semaphores and leases in `finally`.
12. Mark the run cancelled or interrupted durably.
13. Do not append assistant history after cancellation.
14. Define reconnect behavior:
    - live reconnect with cursor only if implemented
    - otherwise explicit resume using the run token
15. Bound event queues and apply backpressure.
16. Redact sensitive event data.

## Tests

- First meaningful event arrives before completion.
- Worker tokens stream incrementally.
- Final tokens stream incrementally.
- Tool progress is visible.
- Verification progress is visible.
- Heartbeats occur during long calls.
- Event IDs and sequence numbers are monotonic.
- One terminal event is emitted.
- Client disconnect cancels Ollama.
- Client disconnect cancels MCP.
- No later tool call occurs after cancellation.
- No assistant history is written after cancellation.
- No semaphore, lease, HTTP connection, or task leaks.
- Slow clients trigger bounded backpressure behavior.
- Proxy buffering is disabled.

## Acceptance criteria

- The endpoint reports actual progress rather than generic polling.
- Disconnect stops compute and network work within a defined bound.
- Cancellation state is durable and observable.
- Streaming and non-streaming paths produce equivalent final contracts.

## Rollback boundary

Keep the non-streaming endpoint unchanged. Introduce the new SSE contract under a versioned compatibility layer if necessary.

---

# Stage 8 — Security-posture preservation and read-only policy non-regression

## Objective

Preserve the current security posture while other architectural and runtime changes are implemented. This stage is an audit and regression stage, not a security-feature implementation stage.

## Current policy to preserve

```text
API-key behavior: unchanged
CORS behavior: unchanged
resume-token behavior: unchanged
read-only tools: allowed according to current server policy
write-capable tools: denied
ambiguous tools: denied
unknown tools: denied unless current trusted registry classifies them as read-only
caller metadata: never authorization
generic external error bodies: preserved
secret/token logging: unchanged and not weakened
```

## In scope

1. Characterize current API-key and CORS behavior.
2. Characterize current resume-token validation.
3. Verify that caller metadata cannot enable write tools.
4. Verify that write-capable, ambiguous, and unknown tools remain denied.
5. Verify that approved work in other stages does not bypass current tool policy.
6. Verify that current redaction behavior remains effective.
7. Verify that documentation JSON cannot influence authorization, tool selection, or runtime execution.
8. Record additional security recommendations as **Deferred by scope**.
9. Fix only a security regression directly introduced by an approved non-security change.

## Explicitly out of scope

- new authentication;
- new authorization;
- approval tokens;
- side-effect ledgers;
- write-tool enablement;
- nonce or replay-token infrastructure;
- tenant isolation redesign;
- TLS redesign;
- security scanning tools;
- dependency audit tooling;
- container scanning;
- SBOM generation;
- image signing or attestation.

## Tests

- Existing API-key tests remain unchanged and pass.
- Caller metadata cannot authorize a write.
- Write-capable tools remain denied.
- Ambiguous and unknown tools remain fail-closed.
- Current resume-token tests remain unchanged and pass.
- Sensitive values are not newly exposed.
- Swagger documentation JSON cannot influence runtime security decisions.
- Non-security changes do not alter security behavior.

## Acceptance criteria

- No new security infrastructure is introduced.
- Existing security behavior remains unchanged.
- Security regressions caused by other work are detected.
- Deferred recommendations are clearly separated from implementation work.

## Rollback boundary

This stage should normally change only tests or a directly regressed line. Do not use it as a reason for broad security refactoring.

# Stage 9 — Transactional persistence, crash recovery, artifacts, and multi-instance readiness

## Objective

Coordinate run outcomes, checkpoints, history, artifacts, and dependency behavior across crashes and multiple application instances.

## Current baseline to verify

- Conversation storage can use memory, Redis, or PostgreSQL.
- Checkpoints can use memory or PostgreSQL.
- Artifact storage can use MinIO.
- Checkpointer health is queried.
- Optional persistence failure can fall back to memory.
- A durable run repository and reconciliation process may still be missing.

## Target persistence sequence

```text
1. Create pending run record
2. Acquire conversation lease
3. Execute checkpointed graph
4. Persist final run outcome
5. Append user message idempotently
6. Append assistant message idempotently
7. Mark history committed
8. Release lease
9. Return response
```

Where one database transaction is impossible, use an explicit outbox/reconciliation state machine.

## Implementation sequence

1. Define database migrations.
2. Add run, lease, message, artifact-reference, and side-effect-ledger tables.
3. Add unique constraints for idempotency.
4. Store conversation messages with run ID and sequence number.
5. Add durable final outcomes.
6. Add startup or scheduled reconciliation.
7. Define checkpoint retention.
8. Define run retention.
9. Define Redis TTL refresh semantics.
10. Define MinIO object naming, hashes, metadata, retention, and deletion.
11. Integrate MinIO into real large-evidence or artifact flows, or keep it explicitly inactive.
12. Add active read/write health checks where safe.
13. Keep liveness process-only.
14. Make readiness capability-specific.
15. Test optional and required dependency behavior independently.
16. Add multi-instance integration tests.
17. Add backup and restore tests.
18. Define privacy deletion across PostgreSQL, Redis, MinIO, and checkpoints.

## Crash-injection matrix

Terminate execution:

- Before pending run insert
- After pending run insert
- After lease acquisition
- During planning
- During research
- During worker execution
- After final checkpoint
- After outcome persistence
- After user-history append
- After assistant-history append
- Before HTTP response
- During artifact upload
- During side-effect ledger commit

After restart, assert deterministic recovery.

## Dependency matrix

Test each independently unavailable:

```text
Ollama
MCP
PostgreSQL
Redis
MinIO
```

For each, verify:

- startup behavior
- readiness status
- inventory status
- request behavior
- degradation metadata
- recovery after dependency restoration

## Tests

- No duplicate messages after retries.
- Completed response remains available after restart.
- Interrupted run resumes only explicitly.
- Lease fencing prevents stale commit.
- PostgreSQL restart recovery works.
- Redis history TTL behaves as documented.
- MinIO object hash and metadata are correct.
- Optional outage enters explicit degraded mode.
- Required outage stops startup.
- Multi-instance same-conversation test is deterministic.
- Backup and restore recover a known run.

## Acceptance criteria

- No lost completed response.
- No duplicate conversation messages.
- No repeated side effect.
- Checkpoint, outcome, and history divergence is detected and reconciled.
- Multi-instance correctness is demonstrated.
- MinIO is either genuinely integrated or explicitly declared inactive.

## Rollback boundary

All schema changes must be migration-driven and backward-aware. Keep memory backends for development and emergency degradation where policy allows.

---

# Stage 10 — Immutable CI/CD, coverage ratcheting, performance qualification, and operational completion

## Objective

Make the release process prove that the exact deployed artifact passed all required checks, then document and measure the production system.

## CI quality gates

Use deliberately configured tools. Target:

```text
python compilation
configured Ruff correctness lint
pytest
overall branch-coverage threshold
per-file branch-coverage threshold
Docker build
Compose validation
container/runtime integration checks
```

Do not invent a tool command before adding and configuring the tool.

## Immutable release process

Required flow:

```text
1. Resolve source SHA
2. Build image once in CI
3. Record image digest and provenance
4. Run unit and integration tests against that image
5. Sign or attest the image
6. Push the immutable digest
7. Deploy that exact digest
8. Run post-deployment checks
9. Promote backup only after success
10. Roll back to the previous known-good digest on failure
```

Do not rebuild source independently on the production runner.

## Integration environment

Provide containerized services for:

- PostgreSQL
- Redis
- MinIO
- Mock MCP Streamable HTTP server
- Lightweight Ollama-compatible fake server

Keep trusted-LAN live tests separately tagged and optional.

## Deployment checks

Require:

```text
/health/live
/health/ready
/api/inventory
minimal /api/chat
research /api/chat
safe budget termination
idempotent retry
resume validation
stream connect
stream disconnect
persistence restart
```

## Performance qualification

Measure per model and role:

```text
queue wait
cold load
warm load
time to first token
prompt tokens
generated tokens
context utilization
tokens per second
structured success
fallback count
cancellation count
memory pressure
```

Measure per graph request:

```text
request duration
node duration
model attempts
tool attempts
research queries
evidence bytes
evidence tokens
checkpoint count
checkpoint bytes
final status
termination reason
```

Load-test:

- one request
- five concurrent requests
- same-conversation concurrency
- different-conversation concurrency
- slow MCP
- slow Ollama
- tool timeout
- client disconnect
- PostgreSQL latency
- Redis outage
- MinIO outage
- model cold load
- production rollback

## Security non-regression only

Do not add security tooling or redesign security in this stage.

Verify only that delivery, performance, persistence, and documentation changes preserve:

- current API-key behavior;
- current CORS behavior;
- current secret redaction;
- current resume-token behavior;
- current read-only MCP tool policy;
- current generic error-response behavior.

Record any broader security recommendation as **Deferred by scope**.

## Documentation set

Update only after code and tests agree:

```text
README.md
docs/architecture.md
docs/graph-topology.md
docs/runtime-routing.md
docs/evidence-and-grounding.md
docs/persistence-and-recovery.md
docs/streaming-contract.md
docs/tool-safety.md
docs/operations.md
docs/testing.md
```

Each document must state:

- current behavior
- configuration
- limits
- failure behavior
- degradation behavior
- observability
- live validation
- unsupported behavior
- rollback

## Acceptance criteria

- The deployed image digest is the tested image digest.
- Existing configured quality and coverage gates block release.
- Rollback restores the previous known-good digest.
- Load and failure-injection results are recorded.
- Documentation matches code, tests, and runtime.
- Operational alerts map to documented failure modes.

## Rollback boundary

Preserve the existing rollback path until the immutable-digest workflow has passed a full production rehearsal.

---

# 8. Cross-cutting testing matrix

| Behavior | Unit | Graph integration | Container integration | Live trusted-LAN | Failure injection |
|---|---:|---:|---:|---:|---:|
| Request identity | Yes | Yes | Yes | Optional | Yes |
| Distributed conversation lease | Yes | Yes | PostgreSQL/Redis | Optional | Yes |
| Durable outcome replay | Yes | Yes | PostgreSQL | Optional | Yes |
| Physical execution metering | Yes | Yes | Ollama/MCP fakes | Optional | Yes |
| Evidence provenance | Yes | Yes | MCP fake | Optional | Yes |
| Claim grounding | Yes | Yes | MCP fake | Optional | Yes |
| Final verification | Yes | Yes | Yes | Optional | Yes |
| Model routing | Yes | Yes | Ollama fake | Ollama inventory | Yes |
| Tool routing | Yes | Yes | MCP fake | MCP inventory | Yes |
| Token streaming | Yes | Yes | Ollama fake | Ollama | Yes |
| Disconnect cancellation | Yes | Yes | Ollama/MCP fakes | Optional | Yes |
| Read-only tool-policy non-regression | Yes | Yes | Current MCP test service | Optional | Yes |
| Write-tool expansion | Deferred | Deferred | Deferred | Deferred | Deferred |
| Restart recovery | Yes | Yes | PostgreSQL | Optional | Yes |
| Redis retention | Yes | Yes | Redis | Optional | Yes |
| Artifact lifecycle | Yes | Yes | MinIO | Optional | Yes |
| Dependency degradation | Yes | Yes | All dependencies | Optional | Yes |
| Immutable deployment | No | No | CI artifact | Deployment | Yes |

Every test must assert meaningful behavior, not merely object construction.

---

# 9. Logging, metrics, and tracing contract

## Identity fields

Every relevant event should include:

```text
request_id
conversation_id
run_id
execution_thread_id
task_id
node
event_sequence
```

## Model fields

```text
role
model
attempt
fallback
queue_wait_seconds
num_ctx
num_predict
prompt_tokens
generated_tokens
done_reason
load_duration
time_to_first_token
total_duration
```

## Tool fields

```text
tool
query_id
attempt
timeout_class
argument_keys
result_status
source_count
elapsed_seconds
side_effect_class
```

## Evidence fields

```text
evidence_id
trust_class
source_quality
freshness_status
content_hash
artifact_uri_hash
grounding_status
```

## Safety and persistence fields

```text
budget_status
termination_reason
read_only_policy_status
idempotency_key_hash
lease_owner_hash
fencing_token
checkpoint_id
history_commit_status
cancellation_status
```

Do not log:

- secrets
- raw resume tokens
- full user prompts by default
- raw evidence bodies
- hidden reasoning
- sensitive tool arguments
- authentication headers

---

# 10. Prompt-engineering requirements

When writing or revising system instructions:

1. Put non-negotiable instructions first.
2. Clearly delimit user input, retrieved evidence, and tool output.
3. State output schema before supporting context.
4. Use positive, testable instructions.
5. Include examples only when they remove ambiguity.
6. Preserve uncertainty.
7. Require citations only when evidence exists.
8. Do not encourage invented citations.
9. Do not expose hidden reasoning.
10. Distinguish an execution summary from private reasoning.
11. Keep planner tasks minimal and bounded.
12. Keep research questions claim-focused.
13. Keep worker claims typed and evidence-linked.
14. Keep verifier actions specific.
15. Keep final synthesis constrained to verified content.
16. Version prompts whose behavior affects durable checkpoints or tests.
17. Add prompt-regression fixtures for representative local models.
18. Measure structured-output success by model and schema.

Consult current primary documentation for OpenAI prompting, LangGraph, Ollama, MCP, FastAPI, PostgreSQL, Redis, and MinIO whenever behavior depends on current contracts.

---

# 11. Packaging and delivery requirements

When code implementation is requested:

1. Base all work on the recorded `release` SHA.
2. Include complete replacement files, never fragments.
3. Include only added or modified repository files.
4. Preserve repository-relative paths.
5. Do not include changelog, report, instruction, or manifest files inside the code ZIP unless explicitly requested.
6. Do not include credentials, `.env`, databases, caches, virtual environments, artifacts, or model files.
7. Provide a manual deletion list in the chat for renamed or obsolete files.
8. Generate a separate SHA-256 checksum file.
9. Validate ZIP integrity.
10. Extract the ZIP into a clean temporary directory.
11. Repeat focused syntax and test validation against extracted files.
12. State every test that was not possible.
13. Provide exact extraction, validation, container rebuild, deployment, and rollback commands.
14. Do not claim a GitHub push succeeded unless the resulting commit is verified on the requested branch.
15. Prefer PowerShell-friendly apply and rollback commands because the user commonly works on Windows.
16. Never include production documentation JSON in a test-data package unless the user explicitly requested documentation changes.

---

# 12. Required response format

## A. Repository baseline

```text
Repository:
Branch:
SHA:
Latest commit:
Tree inspected:
Tests executed:
Passed:
Failed:
Skipped:
Static checks:
Container checks:
Live dependencies:
```

## B. Implementation-status matrix

| Capability | Status | Source evidence | Test evidence | Live evidence | Risk |
|---|---|---|---|---|---|

## C. Confirmed defects

For each defect, use the required defect format.

## D. Bounded design

```text
In scope:
Out of scope:
Compatibility:
API changes:
State changes:
Persistence changes:
Security implications:
Observability:
Tests:
Deployment:
Rollback:
```

## E. Implementation result

Only after code changes:

```text
Base SHA:
Changed files:
Added files:
Deleted files:
Commands run:
Focused test result:
Full test result:
Static-check result:
Container result:
Live result:
Known limitations:
```

## F. Download package

Provide:

- ZIP link
- checksum link
- manual deletion list
- apply commands
- verification commands
- rollback commands

---

# 13. Prohibited shortcuts

Do not:

- Add new security infrastructure without explicit approval.
- Enable write-capable tools.
- Use `docs/example_request/*.json` as runtime defaults, runtime inputs, configuration, or production test fixtures.
- Make `ChatRequest.message` optional to support Swagger.
- Add coverage omissions or waivers merely to make CI pass.
- Add assertion-free tests solely to raise coverage.
- Treat README text as implementation proof.
- Implement multiple stages in one uncontrolled change.
- Hide failures behind a generic successful response.
- Convert incomplete work into a passing verifier result.
- Retry ambiguous side effects automatically.
- Trust caller metadata as authorization.
- Put clients, semaphores, locks, pools, or sockets in durable graph state.
- Use failed tool output as evidence.
- Let synthesis add unsupported claims.
- Describe polling as real streaming.
- Stop emitting SSE without cancelling downstream work.
- Claim multi-instance safety from single-process tests.
- Claim artifact integration because a backend class exists.
- Modify unrelated files to make tests pass.
- Weaken schemas globally for one malformed model response.
- Log hidden reasoning, secrets, tokens, or raw evidence.
- Deploy an image different from the tested image.
- Use historical delivery terminology in new source filenames, function names, class names, settings, metrics, tests, or runtime contracts.
- Continue to a later stage without explicit approval.

---

# 14. Program definition of done

The ten-stage program is complete only when:

- A clean clone reproduces deployed behavior.
- Source, image, and deployed digest provenance are recorded.
- Permanent source naming is domain-oriented.
- Conversation and run identity remain separate.
- Same-conversation concurrency is deterministic across processes.
- Resume is explicit, signed, request-bound, and restart-safe.
- Completed outcomes are durably idempotent.
- Every physical model and tool attempt is metered.
- Deadline behavior survives restart.
- Evidence has typed provenance and trust.
- Claims are validated against evidence.
- Final synthesis is independently verified.
- Shared model and tool gateways are used.
- Routing uses measurable capabilities.
- Real graph and token streaming exists.
- Client disconnect cancels downstream work.
- Caller metadata cannot authorize writes under the preserved current policy.
- Write-capable tools remain disabled unless a separate future program explicitly authorizes them.
- Checkpoints, outcomes, history, and artifacts reconcile safely.
- PostgreSQL restart recovery is demonstrated.
- Redis retention is demonstrated.
- Artifact storage is integrated or explicitly inactive.
- Dependency degradation is explicit.
- CI enforces the configured total and per-file coverage policy and blocks broken releases.
- Swagger/OpenAPI documentation JSON remains isolated from runtime and test semantics.
- Deployment uses the exact tested image digest.
- Rollback is rehearsed.
- Performance and failure-injection results are recorded.
- Documentation matches code, tests, and runtime.
- The system remains understandable without reading one monolithic implementation file.

---

# 15. Opening instruction for the new chat

Begin the new conversation with this instruction:

> Read this entire master prompt before responding. Act as both an expert ChatGPT prompt developer and a Senior/Staff LLM application architect.
>
> Inspect the latest `release` branch of `https://github.com/appNucleus/langchain_langraph.app.local` and treat the newly resolved full SHA as the only implementation source of truth. The preparation-time SHA in this prompt is context only.
>
> First perform an audit only; do not modify code. Record the full SHA and commit metadata, enumerate the repository, run the configured deterministic checks and total/per-file coverage gates when execution is available, reconstruct the active API, graph, model, MCP, evidence, identity, persistence, coverage, deployment, and rollback architecture, and compare current source behavior against all ten stages.
>
> Enforce these non-negotiable constraints:
>
> 1. Every proposed code change must be the smallest coherent change and must reuse the existing implementation before introducing a new abstraction, dependency, service, module, script, setting, or workflow step. Do not reinvent the wheel.
> 2. Do not implement new security features or security tooling. Preserve current API-key, CORS, resume-token, logging/redaction, generic error, and read-only MCP tool behavior exactly.
> 3. JSON files under `docs/example_request/` may be used only to populate Swagger/OpenAPI POST request examples. They must not become Pydantic defaults, runtime inputs, fallback data, configuration, seed data, or production-file test fixtures.
> 4. Verify whether `/api/chat` uses `chat.json` and `/api/chat/stream` uses `chat-stream.json`. Reuse the existing normal and streaming example builders for any minimal correction.
> 5. Preserve the configured 70% overall branch-coverage gate and the 70% per-file gate for every non-waived, non-omitted file. Do not expand omissions or waivers merely to pass CI.
> 6. Durable run identity/repository mechanisms, the central execution meter, typed evidence/claim grounding, `httpx2`, and coverage enforcement already exist in the preparation-time release. Verify their effectiveness and complete confirmed gaps rather than recreating them.
>
> Produce:
>
> - the repository baseline;
> - a requirement-alignment table;
> - the implementation-status matrix;
> - confirmed defects using the required defect format;
> - a list of stale assumptions in this prompt;
> - the recommended order of remaining work.
>
> For the single next recommended stage, provide a bounded design with exact affected files and symbols, existing mechanisms to reuse, proof for anything new, tests, compatibility behavior, coverage impact, deployment impact, and rollback boundary. Do not write, package, push, merge, or deploy code until I explicitly approve that stage.
