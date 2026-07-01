# Provenance Guard

Provenance Guard is a Flask-based backend system that analyzes submitted writing and estimates whether it contains stronger AI-generated or human-written characteristics.

The system combines two independent detection signals, produces a confidence-aware transparency label, stores every decision in a structured audit log, limits abusive request volume, and gives creators a way to appeal classifications.

The project is designed around an important limitation: automated AI-content detection cannot prove authorship. Provenance Guard therefore communicates uncertainty rather than presenting its output as a definitive accusation.

---

## Features

- `POST /submit` endpoint for content analysis
- Groq LLM-based detection signal
- Pure-Python stylometric detection signal
- Weighted multi-signal confidence scoring
- Three reader-facing transparency labels
- `POST /appeal` creator appeals workflow
- Structured JSON audit log
- `GET /log` audit-log endpoint
- Rate limiting with Flask-Limiter
- Short-text uncertainty handling
- Graceful Groq API failure fallback

---

## Technology Stack

- Python
- Flask
- Flask-Limiter
- Groq API
- `llama-3.3-70b-versatile`
- JSON-based local storage
- python-dotenv

---

## Project Structure

```text
ai201-project4-provenance-guard/
├── app.py
├── detector.py
├── storage.py
├── planning.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env
├── submissions.json
└── audit_log.json
