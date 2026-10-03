# AgentMail — Secure RAG-Based AI Email Assistant

## 1. Project Overview

**AgentMail** is a local-first AI email assistant that processes incoming Gmail emails through a secure, multi-stage pipeline.

The system:

1. Fetches incoming emails from Gmail.
2. Screens them for **spam, phishing, and prompt injection** before LLM processing.
3. Determines whether an email requires a reply.
4. Understands the email using classification, priority, deadline, task, and summary extraction.
5. Retrieves relevant previous emails from **ChromaDB** using RAG.
6. Generates a grounded reply using a local LLM.
7. Creates a **Gmail Draft only** — it never automatically sends emails.
8. Future versions will add person-aware personalization, Streamlit automation, Redis short-term state, feedback learning, and career insights.

The core principle is:

> **Untrusted Email → Security Gate → Understanding → Retrieval → Grounded Generation → Gmail Draft → Human Review**

---

# 2. Core Architecture

```text
                    Gmail
                      │
                      ▼
              ┌───────────────┐
              │ Email Fetcher │
              └───────┬───────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Security Screening│
             │ Spam / Phishing  │
             │ Prompt Injection │
             └────────┬─────────┘
                      │
                Safe Email
                      │
                      ▼
             ┌──────────────────┐
             │ Understanding    │
             │ Agent             │
             │                  │
             │ Category         │
             │ Priority         │
             │ Deadline         │
             │ Task             │
             │ Summary          │
             └────────┬─────────┘
                      │
                      ▼
              Reply Eligibility
                      │
                      ▼
             ┌──────────────────┐
             │    Reply Agent   │
             │    LangGraph     │
             └────────┬─────────┘
                      │
             ┌────────┴─────────┐
             ▼                  ▼
       ChromaDB RAG       Email Context
             │                  │
             └────────┬─────────┘
                      ▼
               Context Validation
                      │
                      ▼
                Local LLM
                      │
                      ▼
                Draft Reply
                      │
                      ▼
                 Gmail Drafts
                      │
                      ▼
                Human Review
```

---

# 3. Email Processing Pipeline

## Pipeline A — Incoming Email

The incoming-email pipeline is the main production flow.

```text
Gmail Inbox
    ↓
Fetch unread email
    ↓
Security Screening
    ↓
Spam / Phishing / Injection detection
    ↓
Reject / Flag unsafe email
        OR
Continue
    ↓
Understanding Agent
    ↓
Category + Priority + Deadline + Task + Summary
    ↓
Reply Eligibility Gate
    ↓
Reply Agent
    ↓
RAG Retrieval
    ↓
Context Validation
    ↓
Reply Generation
    ↓
Gmail Draft
    ↓
Human Review
```

**Important:** raw untrusted email content must not reach the normal LLM/reply-generation pipeline before security screening.

---

# 4. Security Pipeline

Security is a separate boundary before normal LLM processing.

Current screening consists of:

```text
Email
 ↓
Local Spam Classifier
 ↓
Phishing Heuristics
 ↓
Prompt-Injection Detection
 ↓
Safe / Flagged Routing
```

The old prompt-injection detector used Gemini inside:

```text
scripts/injection.py
```

That cloud path has been removed. The active Security Agent uses local
heuristics and classifiers; do not restore the Gemini path.

---

# 5. Understanding Agent

The Understanding Agent is responsible for extracting structured information from a safe email.

It currently uses a local LangGraph/Ollama workflow.

```text
Safe Email
   ↓
Understanding Agent
   ├── Category
   ├── Priority
   ├── Deadline
   ├── Task
   └── Summary
```

This information is passed downstream to retrieval and reply generation.

The Understanding Agent should not duplicate functionality already implemented elsewhere.

---

# 6. Reply Agent

The Reply Agent is a local LangGraph `StateGraph`.

Current conceptual flow:

```text
decide_retrieval
      ↓
retrieve
      ↓
validate
      ↓
generate
      ↓
retry / fallback if required
      ↓
assemble
```

The reply agent uses the local Qwen model for decision/retrieval and generation.

It must remain separate from the Gmail UI and Streamlit layer.

---

# 7. ChromaDB / RAG Pipeline

ChromaDB is the project's **long-term semantic email memory**.

Historical sent emails are ingested into ChromaDB so that previous communication can ground future replies.

### Ingestion

```text
Gmail Sent Emails
      ↓
Fetch sent emails
      ↓
Normalize email data
      ↓
Create embeddings
      ↓
ChromaDB
```

The ingestion is deterministic and uses the Gmail message ID for idempotent upserts.

Existing metadata includes:

```text
gmail_message_id
thread_id
sender
recipients
subject
timestamp
category
source
type
```

Current sent-email records use:

```text
source = gmail_sent
type   = sent_email_reply_example
```

Do not rebuild this ingestion pipeline unnecessarily.

---

# 8. RAG Retrieval

The current RAG pipeline is conceptually:

```text
Incoming Email
      ↓
Build Retrieval Query
      ↓
ChromaDB
      ↓
Semantic Retrieval
      ↓
Category-aware Filtering
      ↓
Context Validation
      ↓
Reply Generation
```

The provisional distance threshold is:

```text
DISTANCE_THRESHOLD = 0.60
```

Retrieval must prioritize **relevance**, not simply finding emails from the same person.

---

# 9. Person-Aware RAG — Next Personalization Layer

Person-aware retrieval is the next major RAG enhancement.

The intended flow is:

```text
Incoming Email
      ↓
Identify Sender
      ↓
Identify Category
      ↓
Identify Thread
      ↓
Build Retrieval Query
      ↓
ChromaDB
      ↓
Semantic Relevance
      +
Person Relevance
      +
Category Relevance
      +
Thread Relevance
      ↓
Context Validation
      ↓
Reply Generation
```

### Critical rule

**Same person ≠ automatically relevant email.**

A previous email should be retrieved because it is relevant to the current conversation, with person/category/thread information acting as additional relevance signals.

Person matching must use actual Gmail/Chroma metadata rather than trying to infer identity from email body text.

No separate **Persona Agent** should be created unless explicitly decided.

---

# 10. Three-Email Batch Processing

The system should support processing a small batch of emails rather than only one email at a time.

The intended batch flow is:

```text
Gmail Inbox
     ↓
Fetch 3 eligible unread emails
     ↓
Email 1 ──→ Pipeline
Email 2 ──→ Pipeline
Email 3 ──→ Pipeline
     ↓
Collect Results
     ↓
Create / update Gmail Drafts
     ↓
Return processing summary
```

Each email must retain its own:

- Gmail message ID
- thread ID
- sender
- category
- security result
- understanding result
- retrieval result
- draft result
- stopped/reason reason

Batch processing must **reuse the same underlying pipeline**.

Do not create a separate "batch pipeline" containing duplicated classification, retrieval, or generation logic.

---

# 11. Streamlit Integration

Streamlit is a **future UI/orchestration interface**, not the place where agent logic lives.

The intended architecture is:

```text
                    Streamlit
                       │
                       ▼
              Central Orchestrator
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Gmail          Security      Understanding
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                    RAG
                       ▼
                 Reply Agent
                       ▼
                  Gmail Draft
```

Streamlit should be able to trigger the same pipeline used by CLI execution.

### Example future Streamlit operation

```text
User opens dashboard
        ↓
Clicks "Process Emails"
        ↓
Orchestrator fetches 3 eligible emails
        ↓
Runs common AgentMail pipeline
        ↓
Shows:
  - processed emails
  - security results
  - categories
  - priorities
  - generated drafts
  - retrieval information
  - errors / stopped reasons
```

**No agent logic should be implemented directly inside Streamlit.**

---

# 12. Automatic Gmail Trigger — Future

A future Gmail trigger should call the **same central orchestrator**.

The architecture should eventually support:

```text
CLI
 │
 ├──────────────┐
 │              │
Gmail Trigger  Streamlit
 │              │
 └──────┬───────┘
        ▼
Central Orchestrator
        ▼
Single AgentMail Pipeline
```

There must not be separate implementations for:

- CLI processing
- Gmail-trigger processing
- Streamlit processing
- batch processing

All entry points should eventually invoke the same orchestration layer.

---

# 13. Central Orchestrator

The central orchestrator is a future architectural layer.

Its responsibility will be to coordinate:

```text
Gmail Fetch
   ↓
Security
   ↓
Understanding
   ↓
Reply Eligibility
   ↓
RAG
   ↓
Reply Agent
   ↓
Gmail Draft
```

It should become the single callable entry point for:

- CLI
- future Gmail trigger
- Streamlit
- batch processing
- automated tests

Central orchestration should not be implemented prematurely if required security architecture is still incomplete.

---

# 14. Redis — Future Short-Term Memory

Redis is **not part of the current core architecture**.

ChromaDB and Redis have different purposes.

### ChromaDB

Long-term semantic memory:

```text
Previous emails
Past replies
Semantic retrieval
RAG context
```

### Redis

Future short-term operational state:

```text
Current processing state
Temporary agent state
Batch state
Job/status tracking
Short-lived caches
Rate limiting
Future event/trigger coordination
```

Redis should only be introduced when a concrete short-term-state requirement exists.

**Do not replace ChromaDB with Redis.**

---

# 15. Future Person / Persona Memory

Future personalization can extend the current person-aware RAG system.

Potential flow:

```text
Current Email
     ↓
Person Identification
     ↓
Relevant Historical Communication
     ↓
Person-specific communication patterns
     ↓
RAG Context
     ↓
Reply Generation
```

The goal is to make replies aware of previous communication with a person while still grounding every reply in retrieved evidence.

This should evolve from the existing RAG system rather than becoming an independent Persona Agent.

---

# 16. Gmail Draft Policy

AgentMail **never automatically sends emails**.

The final output is always:

```text
Generated Reply
      ↓
Gmail Draft
      ↓
Human Review
      ↓
Manual Send
```

The Gmail writer should only create drafts.

Human approval remains the final control point.

---

# 17. Data / State Responsibilities

| Component | Responsibility |
|---|---|
| Gmail | Source of incoming and historical email |
| Security pipeline | Spam, phishing, injection screening |
| Understanding Agent | Structured email understanding |
| ChromaDB | Long-term semantic email memory |
| RAG | Relevant historical context retrieval |
| Reply Agent | Grounded response generation |
| Gmail Draft API | Stores generated replies |
| Streamlit | Future UI / trigger interface |
| Orchestrator | Future single pipeline coordinator |
| Redis | Future short-term operational state |
| Feedback system | Future learning from user edits/approval |
| Career module | Future career-related email insights |

---

# 18. Non-Negotiable Architecture Rules

1. **No auto-send.**
2. Untrusted email must pass security screening before normal LLM processing.
3. Do not add Gemini or another cloud LLM to the active pipelines.
4. Use local models for the main agent pipeline.
5. Do not replace ChromaDB with Redis.
6. Do not add Redis without a concrete short-term-state requirement.
7. Do not create a separate Persona Agent without an explicit architecture decision.
8. Person-aware retrieval must supplement semantic relevance, not replace it.
9. Streamlit must call the pipeline; it must not contain agent logic.
10. CLI, Gmail triggers, Streamlit, and batch processing must eventually share one pipeline.
11. Do not duplicate Gmail fetching, classification, embedding, retrieval, or generation logic.
12. Do not silently change retrieval thresholds or source filters.
13. Do not invent metadata, metrics, or evaluation results.
14. Inspect existing code before creating new modules.
15. Run real tests and inspect raw output before declaring a change successful.

---

# 19. Current vs Future

### Currently Implemented

```text
Gmail Fetching
      ↓
Security Screening
      ↓
Understanding Agent
      ↓
Reply Eligibility
      ↓
Reply Agent
      ↓
ChromaDB RAG
      ↓
Context Validation
      ↓
Reply Generation
      ↓
Gmail Draft
```

Sent-email ingestion into ChromaDB is already implemented and verified as idempotent.

### Next

```text
Person-aware RAG
        ↓
Real human-email → Gmail Draft verification
        ↓
Fabrication regression verification
        ↓
Security Agent
        ↓
Central Orchestrator
        ↓
3-email batch processing
```

### Future

```text
Automatic Gmail Trigger
        ↓
Streamlit Dashboard
        ↓
Persona-aware memory
        ↓
Redis short-term state
        ↓
Feedback / learning loop
        ↓
Career insights
```

---

# 20. Coding-Agent Decision Rule

Before modifying AgentMail, a coding agent must determine:

1. **Does the required functionality already exist?**
2. **Which existing pipeline should own it?**
3. **Can the existing component be extended instead of creating another implementation?**
4. **Does the change alter an architectural decision or retrieval behavior?**
5. **Does it require an explicit design decision first?**
6. **Can the change be tested through the existing pipeline?**

The default strategy is:

> **Extend existing pipelines, do not duplicate them.**

Do not implement future architecture merely because a target folder or component has been discussed.

The README describes the intended architecture, but **actual repository code and real execution output are authoritative when verifying the current implementation**.
