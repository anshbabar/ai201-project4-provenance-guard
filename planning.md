# Provenance Guard — Project Planning

## Project Overview

Provenance Guard is a backend attribution system for platforms where users publish original writing. The system analyzes submitted text, estimates whether it is likely AI-generated or human-written, communicates uncertainty through a reader-facing transparency label, records every decision in a structured audit log, and allows creators to appeal classifications.

The system is not intended to prove authorship with certainty. Its purpose is to provide useful context while reducing the harm caused by false accusations of AI use.

---

## System Goals

The system will:

1. Accept text content and a creator ID through an API endpoint.
2. Analyze the text using at least two independent detection signals.
3. Combine the signals into an AI-likelihood score from 0.0 to 1.0.
4. Produce one of three attribution categories:
   - Likely AI-generated
   - Uncertain
   - Likely human-written
5. Return a plain-language transparency label.
6. Store the classification and signal scores in a structured audit log.
7. Allow a creator to appeal a classification.
8. Apply rate limiting to reduce automated abuse.

---

## API Design

### POST `/submit`

Accepts a piece of text for attribution analysis.

#### Request

```json
{
  "creator_id": "creator-123",
  "text": "The text that the creator wants analyzed."
}