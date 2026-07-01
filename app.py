from datetime import datetime, timezone
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from detector import analyze_text
from storage import (
    add_audit_entry,
    get_audit_entries,
    get_submission,
    save_submission,
    update_submission,
)


load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


def current_timestamp() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "Provenance Guard",
            "message": "The Provenance Guard API is running.",
            "endpoints": {
                "health": "GET /health",
                "submit": "POST /submit",
                "appeal": "POST /appeal",
                "log": "GET /log",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "Provenance Guard",
        }
    )


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    if not request.is_json:
        return (
            jsonify(
                {
                    "error": "A JSON request body is required.",
                }
            ),
            400,
        )

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return (
            jsonify(
                {
                    "error": "A valid JSON object is required.",
                }
            ),
            400,
        )

    creator_id = data.get("creator_id")
    text = data.get("text")

    if not isinstance(creator_id, str) or not creator_id.strip():
        return (
            jsonify(
                {
                    "error": "creator_id is required.",
                }
            ),
            400,
        )

    if not isinstance(text, str) or not text.strip():
        return (
            jsonify(
                {
                    "error": "text is required.",
                }
            ),
            400,
        )

    creator_id = creator_id.strip()
    text = text.strip()

    if len(text) > 15000:
        return (
            jsonify(
                {
                    "error": "text must contain no more than 15,000 characters.",
                }
            ),
            400,
        )

    content_id = str(uuid.uuid4())
    timestamp = current_timestamp()

    analysis = analyze_text(text)

    submission_record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "text": text,
        "created_at": timestamp,
        "updated_at": timestamp,
        "attribution": analysis["attribution"],
        "confidence": analysis["confidence"],
        "ai_likelihood": analysis["ai_likelihood"],
        "signals": analysis["signals"],
        "label": analysis["label"],
        "status": "classified",
        "appeal_filed": False,
        "appeal_reasoning": None,
        "appeal_timestamp": None,
    }

    save_submission(submission_record)

    audit_entry = {
        "event_type": "classification",
        "timestamp": timestamp,
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": analysis["attribution"],
        "confidence": analysis["confidence"],
        "ai_likelihood": analysis["ai_likelihood"],
        "llm_score": analysis["signals"]["llm_score"],
        "llm_available": analysis["signals"]["llm_available"],
        "llm_reasoning": analysis["signals"]["llm_reasoning"],
        "stylometric_score": analysis["signals"]["stylometric_score"],
        "stylometric_metrics": analysis["signals"][
            "stylometric_metrics"
        ],
        "label": analysis["label"],
        "status": "classified",
        "appeal_filed": False,
    }

    add_audit_entry(audit_entry)

    response = {
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": analysis["attribution"],
        "confidence": analysis["confidence"],
        "ai_likelihood": analysis["ai_likelihood"],
        "signals": {
            "llm_score": analysis["signals"]["llm_score"],
            "stylometric_score": analysis["signals"][
                "stylometric_score"
            ],
        },
        "label": analysis["label"],
        "status": "classified",
    }

    return jsonify(response), 200


@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per minute;20 per day")
def appeal():
    if not request.is_json:
        return (
            jsonify(
                {
                    "error": "A JSON request body is required.",
                }
            ),
            400,
        )

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return (
            jsonify(
                {
                    "error": "A valid JSON object is required.",
                }
            ),
            400,
        )

    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")

    if not isinstance(content_id, str) or not content_id.strip():
        return (
            jsonify(
                {
                    "error": "content_id is required.",
                }
            ),
            400,
        )

    if (
        not isinstance(creator_reasoning, str)
        or not creator_reasoning.strip()
    ):
        return (
            jsonify(
                {
                    "error": "creator_reasoning is required.",
                }
            ),
            400,
        )

    content_id = content_id.strip()
    creator_reasoning = creator_reasoning.strip()

    original_submission = get_submission(content_id)

    if original_submission is None:
        return (
            jsonify(
                {
                    "error": (
                        "No submission was found for the provided content_id."
                    ),
                }
            ),
            404,
        )

    appeal_timestamp = current_timestamp()

    updated_submission = update_submission(
        content_id,
        {
            "status": "under_review",
            "appeal_filed": True,
            "appeal_reasoning": creator_reasoning,
            "appeal_timestamp": appeal_timestamp,
            "updated_at": appeal_timestamp,
        },
    )

    audit_entry = {
        "event_type": "appeal",
        "timestamp": appeal_timestamp,
        "content_id": content_id,
        "creator_id": original_submission["creator_id"],
        "original_attribution": original_submission["attribution"],
        "original_confidence": original_submission["confidence"],
        "original_ai_likelihood": original_submission[
            "ai_likelihood"
        ],
        "llm_score": original_submission["signals"]["llm_score"],
        "stylometric_score": original_submission["signals"][
            "stylometric_score"
        ],
        "creator_reasoning": creator_reasoning,
        "status": "under_review",
        "appeal_filed": True,
    }

    add_audit_entry(audit_entry)

    return (
        jsonify(
            {
                "content_id": content_id,
                "message": (
                    "Appeal received. This content is now under review."
                ),
                "status": updated_submission["status"],
            }
        ),
        200,
    )


@app.route("/log", methods=["GET"])
def log():
    limit_value = request.args.get("limit", default=50, type=int)

    if limit_value is None:
        limit_value = 50

    limit_value = max(1, min(limit_value, 100))

    return jsonify(
        {
            "entries": get_audit_entries(limit=limit_value),
        }
    )


@app.errorhandler(429)
def rate_limit_exceeded(error):
    return (
        jsonify(
            {
                "error": "Rate limit exceeded.",
                "details": str(error.description),
            }
        ),
        429,
    )


@app.errorhandler(404)
def not_found(_error):
    return (
        jsonify(
            {
                "error": "Endpoint not found.",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_server_error(_error):
    return (
        jsonify(
            {
                "error": "An unexpected server error occurred.",
            }
        ),
        500,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)