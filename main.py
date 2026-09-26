import os
import json
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

IS_VERCEL = os.getenv("VERCEL", "").lower() == "1"

if IS_VERCEL:
    DB_PATH = Path("/tmp/chatbot.db")
else:
    DB_PATH = STORAGE_DIR / "chatbot.db"

BYNARA_API_KEY = os.getenv("BYNARA_API_KEY", "")
BYNARA_URL = "https://router.bynara.id/v1/chat/completions"
BYNARA_MODEL = "agnes-2.5-flash"

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")

APP_NAME = "AI Backend"
APP_VERSION = "1.0.0"

VALID_DEMOS = {
    "dentist",
    "real_estate",
    "restaurant",
    "hvac",
    "lawyer",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Multi-demo AI chatbot backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    if not IS_VERCEL:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    if not IS_VERCEL:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    db = get_db()

    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            demo_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id);

        CREATE TABLE IF NOT EXISTS cached_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            demo_id TEXT NOT NULL,
            question_normalized TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_type TEXT,
            source_ids TEXT,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            use_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_cached_question
        ON cached_answers(demo_id, question_normalized);

        CREATE TABLE IF NOT EXISTS admin_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            demo_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source_id TEXT,
            source_type TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS knowledge_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            demo_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_url TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            demo_id TEXT,
            session_id TEXT,
            event_type TEXT NOT NULL,
            message TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    if ADMIN_PASS:
        existing = db.execute(
            "SELECT id FROM admin_users WHERE username = ?",
            (ADMIN_USER,),
        ).fetchone()

        if not existing:
            password_hash = hashlib.sha256(
                ADMIN_PASS.encode("utf-8")
            ).hexdigest()

            db.execute(
                """
                INSERT INTO admin_users
                (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    ADMIN_USER,
                    password_hash,
                    now_iso(),
                ),
            )

    db.commit()
    db.close()


# ============================================================
# HELPERS
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    """
    Normalize customer messages using the generated normalization map.
    Safely falls back to basic normalization if the file cannot be loaded.
    """
    import re

    if not text:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s$.-]", " ", text)
    text = " ".join(text.split())

    normalization_file = DATA_DIR / "normalization.json"

    if not normalization_file.exists():
        return text

    try:
        with open(
            normalization_file,
            "r",
            encoding="utf-8",
        ) as file:
            mappings = json.load(file)

        if not isinstance(mappings, dict):
            return text

        # Ignore metadata fields from normalization.json.
        mappings = {
            str(key).lower().strip(): str(value).lower().strip()
            for key, value in mappings.items()
            if isinstance(value, str)
            and str(key).lower()
            not in {
                "version",
                "description",
            }
        }

        # Apply longer phrases first so phrase mappings work.
        for source, target in sorted(
            mappings.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if not source:
                continue

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(source)
                + r"(?![a-z0-9])"
            )

            text = re.sub(
                pattern,
                target,
                text,
            )

        return " ".join(text.split())

    except Exception:
        return text

def validate_demo(demo_id: str) -> str:
    demo_id = demo_id.strip().lower()

    if demo_id not in VALID_DEMOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid demo_id. Valid demos: {sorted(VALID_DEMOS)}",
        )

    return demo_id


def log_event(
    event_type: str,
    message: str = "",
    demo_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    try:
        db = get_db()

        db.execute(
            """
            INSERT INTO system_events
            (demo_id, session_id, event_type, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                demo_id,
                session_id,
                event_type,
                message[:2000],
                now_iso(),
            ),
        )

        db.commit()
        db.close()

    except Exception:
        pass


# ============================================================
# FILE LOADING
# ============================================================

def load_json_file(path: Path, default=None):
    if default is None:
        default = []

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def load_jsonl_file(path: Path):
    items = []

    if not path.exists():
        return items

    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    except Exception:
        pass

    return items


def demo_file(demo_id: str, filename: str) -> Path:
    return DATA_DIR / demo_id / filename


def load_demo_data(demo_id: str):
    return {
        "faqs": load_jsonl_file(
            demo_file(demo_id, "faqs.jsonl")
        ),
        "facts": load_jsonl_file(
            demo_file(demo_id, "facts.jsonl")
        ),
        "knowledge": load_jsonl_file(
            demo_file(demo_id, "knowledge.jsonl")
        ),
        "instructions": load_json_file(
            demo_file(demo_id, "instructions.json"),
            [],
        ),
    }


def load_structured_data(demo_id: str):
    filename_map = {
        "dentist": "services.json",
        "real_estate": "properties.json",
        "restaurant": "menu.json",
        "hvac": "services.json",
        "lawyer": "services.json",
    }

    filename = filename_map[demo_id]

    return load_json_file(
        demo_file(demo_id, filename),
        [],
    )


# ============================================================
# SIMPLE TEXT MATCHING
# ============================================================

def _search_tokens(value: str):
    import re

    return set(
        token
        for token in re.findall(
            r"[a-z0-9]+",
            str(value or "").lower(),
        )
        if len(token) > 1
    )


def text_score(question: str, candidate: str) -> float:
    question = normalize_text(question)
    candidate = normalize_text(candidate)

    if not question or not candidate:
        return 0.0

    if question == candidate:
        return 1.0

    if question in candidate or candidate in question:
        return 0.90

    question_words = _search_tokens(question)
    candidate_words = _search_tokens(candidate)

    if not question_words or not candidate_words:
        return 0.0

    intersection = question_words.intersection(
        candidate_words
    )

    return len(intersection) / len(question_words)


def _best_match(
    items,
    question: str,
    text_builder,
    threshold: float,
):
    best_item = None
    best_score = 0.0

    for item in items:
        if not isinstance(item, dict):
            continue

        candidate = text_builder(item)

        if not candidate:
            continue

        score = text_score(
            question,
            candidate,
        )

        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None:
        return None

    if best_score < threshold:
        return None

    return best_item


def find_faq_answer(
    demo_id: str,
    question: str,
):
    data = load_demo_data(demo_id)

    item = _best_match(
        data["faqs"],
        question,
        lambda value: " ".join(
            str(value.get(key) or "")
            for key in (
                "question",
                "q",
                "keywords",
                "topic",
            )
        ),
        0.75,
    )

    if not item:
        return None

    answer = str(
        item.get("answer")
        or item.get("a")
        or ""
    ).strip()

    if not answer:
        return None

    return {
        "answer": answer,
        "source_type": "faq",
        "source_ids": [
            str(
                item.get("id")
                or item.get("faq_id")
                or ""
            )
        ],
    }


def find_fact(
    demo_id: str,
    question: str,
):
    data = load_demo_data(demo_id)

    item = _best_match(
        data["facts"],
        question,
        lambda value: " ".join(
            str(value.get(key) or "")
            for key in (
                "text",
                "fact",
                "content",
                "title",
                "topic",
                "keywords",
            )
        ),
        0.45,
    )

    if not item:
        return None

    answer = str(
        item.get("text")
        or item.get("fact")
        or item.get("content")
        or ""
    ).strip()

    if not answer:
        return None

    return {
        "answer": answer,
        "source_type": "fact",
        "source_ids": [
            str(
                item.get("id")
                or item.get("fact_id")
                or ""
            )
        ],
    }


def find_knowledge(
    demo_id: str,
    question: str,
):
    data = load_demo_data(demo_id)

    item = _best_match(
        data["knowledge"],
        question,
        lambda value: " ".join(
            str(value.get(key) or "")
            for key in (
                "text",
                "content",
                "paragraph",
                "title",
                "topic",
                "keywords",
            )
        ),
        0.35,
    )

    if not item:
        return None

    answer = str(
        item.get("text")
        or item.get("content")
        or item.get("paragraph")
        or ""
    ).strip()

    if not answer:
        return None

    return {
        "answer": answer,
        "source_type": "knowledge",
        "source_ids": [
            str(
                item.get("id")
                or item.get("knowledge_id")
                or ""
            )
        ],
    }


def find_admin_knowledge(
    demo_id: str,
    question: str,
):
    db = get_db()

    rows = db.execute(
        """
        SELECT *
        FROM admin_knowledge
        WHERE demo_id = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (demo_id,),
    ).fetchall()

    db.close()

    best_row = None
    best_score = 0.0

    for row in rows:
        searchable = " ".join(
            [
                str(row["title"] or ""),
                str(row["content"] or ""),
                str(row["source_id"] or ""),
                str(row["source_type"] or ""),
            ]
        )

        score = text_score(
            question,
            searchable,
        )

        if score > best_score:
            best_score = score
            best_row = row

    if not best_row or best_score < 0.35:
        return None

    return {
        "answer": str(
            best_row["content"] or ""
        ).strip(),
        "source_type": "admin_knowledge",
        "source_ids": [
            str(
                best_row["source_id"]
                or best_row["id"]
            )
        ],
    }


def find_structured_match(
    demo_id: str,
    question: str,
):
    items = load_structured_data(demo_id)

    if not isinstance(items, list):
        return None

    best_item = None
    best_score = 0.0

    for item in items:
        if not isinstance(item, dict):
            continue

        searchable_parts = []

        for key, value in item.items():
            if isinstance(
                value,
                (str, int, float),
            ):
                searchable_parts.append(
                    f"{key} {value}"
                )
            elif isinstance(value, list):
                searchable_parts.append(
                    f"{key} "
                    + " ".join(
                        str(value_item)
                        for value_item in value
                    )
                )

        score = text_score(
            question,
            " ".join(searchable_parts),
        )

        if score > best_score:
            best_score = score
            best_item = item

    if not best_item or best_score < 0.30:
        return None

    return {
        "answer": format_structured_answer(
            demo_id,
            best_item,
        ),
        "source_type": "structured",
        "source_ids": [
            str(
                best_item.get("id")
                or best_item.get("property_id")
                or best_item.get("service_id")
                or best_item.get("name")
                or ""
            )
        ],
    }


def format_structured_answer(
    demo_id: str,
    item: dict,
) -> str:
    parts = []

    if demo_id == "dentist":
        parts.append(
            str(
                item.get("service_name")
                or item.get("name")
                or "Dental service"
            )
        )

        if item.get("description"):
            parts.append(
                str(item["description"])
            )

        if item.get("price") is not None:
            parts.append(
                f"Price: {item.get('currency', '')} "
                f"{item['price']}".strip()
            )

        duration = (
            item.get("duration")
            or item.get("duration_minutes")
        )

        if duration is not None:
            parts.append(
                f"Duration: {duration}"
                + (
                    " minutes"
                    if isinstance(
                        duration,
                        (int, float),
                    )
                    else ""
                )
            )

    elif demo_id == "real_estate":
        parts.append(
            str(
                item.get("id")
                or item.get("name")
                or "Property"
            )
        )

        for label, key in [
            ("Type", "type"),
            ("Transaction", "transaction"),
            ("Price", "price"),
            ("Currency", "currency"),
            ("Location", "location"),
            ("Bedrooms", "bedrooms"),
            ("Bathrooms", "bathrooms"),
            ("Area (sq ft)", "area_sqft"),
            ("Availability", "availability"),
        ]:
            if item.get(key) is not None:
                parts.append(
                    f"{label}: {item[key]}"
                )

        if item.get("amenities"):
            amenities = item["amenities"]

            parts.append(
                "Amenities: "
                + (
                    ", ".join(
                        str(value)
                        for value in amenities
                    )
                    if isinstance(
                        amenities,
                        list,
                    )
                    else str(amenities)
                )
            )

        if item.get("description"):
            parts.append(
                str(item["description"])
            )

    elif demo_id == "restaurant":
        parts.append(
            str(
                item.get("name")
                or item.get("item_name")
                or "Menu item"
            )
        )

        if item.get("category"):
            parts.append(
                f"Category: {item['category']}"
            )

        if item.get("description"):
            parts.append(
                str(item["description"])
            )

        if item.get("price") is not None:
            parts.append(
                f"Price: {item.get('currency', '')} "
                f"{item['price']}".strip()
            )

        if item.get("ingredients"):
            ingredients = item["ingredients"]

            parts.append(
                "Ingredients: "
                + (
                    ", ".join(
                        str(value)
                        for value in ingredients
                    )
                    if isinstance(
                        ingredients,
                        list,
                    )
                    else str(ingredients)
                )
            )

        if item.get("dietary_info"):
            parts.append(
                f"Dietary information: "
                f"{item['dietary_info']}"
            )

        if item.get("availability"):
            parts.append(
                f"Availability: "
                f"{item['availability']}"
            )

    elif demo_id == "hvac":
        parts.append(
            str(
                item.get("service")
                or item.get("service_name")
                or "HVAC service"
            )
        )

        if item.get("description"):
            parts.append(
                str(item["description"])
            )

        if item.get("price") is not None:
            parts.append(
                f"Price: {item.get('currency', '')} "
                f"{item['price']}".strip()
            )

        if item.get("price_range"):
            parts.append(
                f"Price range: {item['price_range']}"
            )

        if item.get("estimated_duration_minutes") is not None:
            parts.append(
                "Estimated duration: "
                f"{item['estimated_duration_minutes']} minutes"
            )

        if item.get("availability"):
            parts.append(
                f"Availability: {item['availability']}"
            )

        if item.get("details"):
            parts.append(
                str(item["details"])
            )

    elif demo_id == "lawyer":
        parts.append(
            str(
                item.get("service_name")
                or "Legal service"
            )
        )

        if item.get("practice_area"):
            parts.append(
                f"Practice area: "
                f"{item['practice_area']}"
            )

        if item.get("consultation_fee") is not None:
            parts.append(
                "Consultation fee: "
                f"{item.get('currency', '')} "
                f"{item['consultation_fee']}".strip()
            )

        if item.get("process_description"):
            parts.append(
                str(item["process_description"])
            )

        if item.get("required_documents"):
            documents = item["required_documents"]

            parts.append(
                "Required documents: "
                + (
                    ", ".join(
                        str(value)
                        for value in documents
                    )
                    if isinstance(
                        documents,
                        list,
                    )
                    else str(documents)
                )
            )

    return " ".join(
        str(part).strip()
        for part in parts
        if str(part).strip()
    )


# ============================================================
# DRAFT GENERATION
# ============================================================

def build_draft(
    demo_id: str,
    question: str,
    context=None,
):
    """
    Local factual retrieval priority:

    FAQ
    -> admin knowledge
    -> structured data
    -> facts
    -> knowledge
    -> safe fallback
    """

    result = find_faq_answer(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_admin_knowledge(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_structured_match(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_fact(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_knowledge(
        demo_id,
        question,
    )

    if result:
        return result

    return {
        "answer": (
            "I don't have a reliable answer from the available "
            "information yet. I can help with the information "
            "available in this demo, or you can provide more details."
        ),
        "source_type": "fallback",
        "source_ids": [],
    }


# ============================================================
# CACHE
# ============================================================

def get_cached_answer(demo_id: str, question: str):
    db = get_db()

    row = db.execute(
        """
        SELECT *
        FROM cached_answers
        WHERE demo_id = ?
        AND question_normalized = ?
        ORDER BY use_count DESC, id DESC
        LIMIT 1
        """,
        (
            demo_id,
            question,
        ),
    ).fetchone()

    if row:
        db.execute(
            """
            UPDATE cached_answers
            SET last_used_at = ?,
                use_count = use_count + 1
            WHERE id = ?
            """,
            (
                now_iso(),
                row["id"],
            ),
        )

        db.commit()

        result = {
            "answer": row["answer"],
            "source_type": row["source_type"] or "cache",
            "source_ids": json.loads(
                row["source_ids"] or "[]"
            ),
        }

        db.close()
        return result

    db.close()
    return None


def save_cached_answer(
    demo_id: str,
    question: str,
    answer: str,
    source_type: str,
    source_ids=None,
):
    if not answer:
        return

    if source_ids is None:
        source_ids = []

    db = get_db()

    existing = db.execute(
        """
        SELECT id
        FROM cached_answers
        WHERE demo_id = ?
        AND question_normalized = ?
        LIMIT 1
        """,
        (
            demo_id,
            question,
        ),
    ).fetchone()

    if existing:
        db.execute(
            """
            UPDATE cached_answers
            SET answer = ?,
                source_type = ?,
                source_ids = ?,
                last_used_at = ?
            WHERE id = ?
            """,
            (
                answer,
                source_type,
                json.dumps(source_ids),
                now_iso(),
                existing["id"],
            ),
        )
    else:
        db.execute(
            """
            INSERT INTO cached_answers
            (
                demo_id,
                question_normalized,
                answer,
                source_type,
                source_ids,
                created_at,
                last_used_at,
                use_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                demo_id,
                question,
                answer,
                source_type,
                json.dumps(source_ids),
                now_iso(),
                now_iso(),
                0,
            ),
        )

    db.commit()
    db.close()


# ============================================================
# CONVERSATIONS
# ============================================================

def ensure_conversation(session_id: str, demo_id: str):
    db = get_db()

    existing = db.execute(
        """
        SELECT id
        FROM conversations
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    if not existing:
        db.execute(
            """
            INSERT INTO conversations
            (session_id, demo_id, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                demo_id,
                now_iso(),
                now_iso(),
            ),
        )
    else:
        db.execute(
            """
            UPDATE conversations
            SET demo_id = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (
                demo_id,
                now_iso(),
                session_id,
            ),
        )

    db.commit()
    db.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
):
    db = get_db()

    db.execute(
        """
        INSERT INTO messages
        (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            now_iso(),
        ),
    )

    db.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE session_id = ?
        """,
        (
            now_iso(),
            session_id,
        ),
    )

    db.commit()
    db.close()


def get_recent_messages(
    session_id: str,
    limit: int = 12,
):
    db = get_db()

    rows = db.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            session_id,
            limit,
        ),
    ).fetchall()

    db.close()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        for row in reversed(rows)
    ]


# ============================================================
# DRAFT GENERATION
# ============================================================

def build_draft(
    demo_id: str,
    question: str,
    context=None,
):
    """
    Creates a factual local draft before Agnes is called.
    """

    result = find_faq_answer(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_structured_match(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_fact(
        demo_id,
        question,
    )

    if result:
        return result

    result = find_knowledge(
        demo_id,
        question,
    )

    if result:
        return result

    return {
        "answer": (
            "I don't have a reliable answer from the available "
            "information yet. I can help with the information "
            "available in this demo, or you can provide more details."
        ),
        "source_type": "fallback",
        "source_ids": [],
    }


# ============================================================
# LOCAL POLISHING
# ============================================================

def local_polish(text: str) -> str:
    if not text:
        return (
            "I’m sorry, but I’m unable to provide an answer "
            "right now."
        )

    text = text.strip()

    if not text:
        return (
            "I’m sorry, but I’m unable to provide an answer "
            "right now."
        )

    return text


# ============================================================
# AGNES / NARAROUTER
# ============================================================

def call_agnes(
    demo_id: str,
    customer_message: str,
    draft: str,
    context=None,
):
    if not BYNARA_API_KEY:
        raise RuntimeError(
            "BYNARA_API_KEY is not configured."
        )

    instructions = load_demo_data(
        demo_id
    ).get("instructions", [])

    if isinstance(instructions, list):
        instruction_text = "\n".join(
            str(item)
            if not isinstance(item, dict)
            else str(
                item.get("text")
                or item.get("instruction")
                or ""
            )
            for item in instructions
        )
    elif isinstance(instructions, dict):
        instruction_text = json.dumps(
            instructions,
            ensure_ascii=False,
        )
    else:
        instruction_text = str(instructions)

    recent_context = context or []

    messages = [
        {
            "role": "system",
            "content": (
                "You are the response-generation layer of a "
                "business chatbot. Use the supplied factual "
                "draft as the primary source. Do not invent "
                "facts, prices, availability, laws, policies, "
                "or business details. Keep the response "
                "professional, clear, and useful.\n\n"
                f"Demo: {demo_id}\n\n"
                f"Demo instructions:\n{instruction_text}"
            ),
        }
    ]

    for item in recent_context[-8:]:
        role = item.get("role")

        if role not in {"user", "assistant"}:
            continue

        messages.append(
            {
                "role": role,
                "content": str(
                    item.get("content", "")
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                f"Customer message:\n{customer_message}\n\n"
                f"Factual draft/context:\n{draft}\n\n"
                "Create the final customer-facing response."
            ),
        }
    )

    response = requests.post(
        BYNARA_URL,
        headers={
            "Authorization": (
                f"Bearer {BYNARA_API_KEY}"
            ),
            "Content-Type": "application/json",
        },
        json={
            "model": BYNARA_MODEL,
            "messages": messages,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(
            "Agnes returned no choices."
        )

    message = choices[0].get("message") or {}

    content = message.get("content")

    if not content:
        raise RuntimeError(
            "Agnes returned an empty response."
        )

    return str(content).strip()


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    demo_id: str
    message: str


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class KnowledgeRequest(BaseModel):
    demo_id: str
    title: str
    content: str
    source_id: Optional[str] = None
    source_type: Optional[str] = "manual"


class DemoRequest(BaseModel):
    demo_id: str


# ============================================================
# ADMIN AUTH
# ============================================================

def password_hash(password: str) -> str:
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def create_admin_token(username: str) -> str:
    """
    Lightweight demo token.

    Production authentication can later be upgraded to
    full JWT handling without changing the public API.
    """
    timestamp = str(int(time.time()))

    payload = f"{username}:{timestamp}"

    signature = hashlib.sha256(
        f"{payload}:{JWT_SECRET}".encode("utf-8")
    ).hexdigest()

    return f"{payload}:{signature}"


def verify_admin_token(
    authorization: Optional[str],
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization required.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format.",
        )

    token = authorization.split(
        " ",
        1,
    )[1].strip()

    parts = token.split(":")

    if len(parts) != 3:
        raise HTTPException(
            status_code=401,
            detail="Invalid token.",
        )

    username, timestamp, signature = parts

    if not JWT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="JWT_SECRET is not configured.",
        )

    expected = hashlib.sha256(
        f"{username}:{timestamp}:{JWT_SECRET}".encode(
            "utf-8"
        )
    ).hexdigest()

    if not secrets.compare_digest(
        signature,
        expected,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid token.",
        )

    try:
        token_time = int(timestamp)
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token timestamp.",
        )

    if abs(int(time.time()) - token_time) > 86400:
        raise HTTPException(
            status_code=401,
            detail="Token expired.",
        )

    return username


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    init_database()


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
    }


@app.get("/health")
def health():
    try:
        db = get_db()

        db.execute(
            "SELECT 1"
        ).fetchone()

        db.close()

        return {
            "status": "ok",
            "database": "ok",
            "vercel": IS_VERCEL,
            "db_path": str(DB_PATH),
            "demos": sorted(VALID_DEMOS),
        }

    except Exception as exc:
        return {
            "status": "degraded",
            "database": "error",
            "error": str(exc),
        }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(request: ChatRequest):
    demo_id = validate_demo(
        request.demo_id
    )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    session_id = request.session_id.strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required.",
        )

    ensure_conversation(
        session_id,
        demo_id,
    )

    save_message(
        session_id,
        "user",
        message,
    )

    normalized = normalize_text(
        message
    )

    # --------------------------------------------------------
    # 1. FAQ
    # --------------------------------------------------------

    faq_result = find_faq_answer(
        demo_id,
        normalized,
    )

    # --------------------------------------------------------
    # 2. CACHE
    # --------------------------------------------------------

    cached = None

    if not faq_result:
        cached = get_cached_answer(
            demo_id,
            normalized,
        )

    if cached:
        answer = local_polish(
            cached["answer"]
        )

        save_message(
            session_id,
            "assistant",
            answer,
        )

        log_event(
            "cache_hit",
            "Cached answer returned.",
            demo_id,
            session_id,
        )

        return {
            "success": True,
            "answer": answer,
            "source_type": "cache",
            "agnes_used": False,
            "fallback_used": False,
        }

    # --------------------------------------------------------
    # 3. BUILD LOCAL FACTUAL DRAFT
    # --------------------------------------------------------

    draft_result = faq_result or build_draft(
        demo_id,
        normalized,
        get_recent_messages(
            session_id
        ),
    )

    draft = draft_result["answer"]

    # --------------------------------------------------------
    # 4. TRY AGNES
    # --------------------------------------------------------

    answer = None
    agnes_used = False
    fallback_used = False

    try:
        answer = call_agnes(
            demo_id=demo_id,
            customer_message=message,
            draft=draft,
            context=get_recent_messages(
                session_id
            ),
        )

        answer = local_polish(
            answer
        )

        agnes_used = True

    except Exception as exc:
        fallback_used = True

        log_event(
            "agnes_error",
            str(exc),
            demo_id,
            session_id,
        )

        # Critical resilience rule:
        # Agnes failure NEVER crashes the chatbot.
        answer = local_polish(
            draft
        )

        log_event(
            "fallback_used",
            "Returned local factual draft.",
            demo_id,
            session_id,
        )

    # --------------------------------------------------------
    # 5. CACHE SUCCESSFUL ANSWER
    # --------------------------------------------------------

    if answer and not fallback_used:
        save_cached_answer(
            demo_id=demo_id,
            question=normalized,
            answer=answer,
            source_type=(
                "agnes"
                if agnes_used
                else draft_result["source_type"]
            ),
            source_ids=draft_result[
                "source_ids"
            ],
        )

    save_message(
        session_id,
        "assistant",
        answer,
    )

    return {
        "success": True,
        "answer": answer,
        "source_type": (
            "agnes"
            if agnes_used
            else draft_result["source_type"]
        ),
        "agnes_used": agnes_used,
        "fallback_used": fallback_used,
    }


# ============================================================
# CONVERSATION HISTORY
# ============================================================

@app.get(
    "/api/conversations/{session_id}"
)
def conversation_history(
    session_id: str,
):
    db = get_db()

    conversation = db.execute(
        """
        SELECT *
        FROM conversations
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    rows = db.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()

    db.close()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "session_id": session_id,
        "demo_id": conversation["demo_id"],
        "messages": [
            dict(row)
            for row in rows
        ],
    }


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.post("/admin/login")
def admin_login(
    request: AdminLoginRequest,
):
    if not ADMIN_PASS:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_PASS is not configured.",
        )

    if (
        request.username != ADMIN_USER
        or request.password != ADMIN_PASS
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "success": True,
        "access_token": create_admin_token(
            request.username
        ),
        "token_type": "bearer",
    }


# ============================================================
# ADMIN: DEMOS
# ============================================================

@app.get("/admin/demos")
def admin_demos(
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    return {
        "demos": [
            {
                "id": demo,
                "data_path": str(
                    DATA_DIR / demo
                ),
            }
            for demo in sorted(
                VALID_DEMOS
            )
        ]
    }


# ============================================================
# ADMIN: KNOWLEDGE
# ============================================================

@app.post("/admin/knowledge")
def admin_add_knowledge(
    request: KnowledgeRequest,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    demo_id = validate_demo(
        request.demo_id
    )

    if not request.title.strip():
        raise HTTPException(
            status_code=400,
            detail="Title is required.",
        )

    if not request.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Content is required.",
        )

    db = get_db()

    timestamp = now_iso()

    cursor = db.execute(
        """
        INSERT INTO admin_knowledge
        (
            demo_id,
            title,
            content,
            source_id,
            source_type,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            demo_id,
            request.title.strip(),
            request.content.strip(),
            request.source_id,
            request.source_type,
            timestamp,
            timestamp,
        ),
    )

    db.commit()

    record_id = cursor.lastrowid

    db.close()

    return {
        "success": True,
        "id": record_id,
        "demo_id": demo_id,
    }


@app.get("/admin/knowledge")
def admin_get_knowledge(
    demo_id: Optional[str] = None,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    db = get_db()

    if demo_id:
        demo_id = validate_demo(
            demo_id
        )

        rows = db.execute(
            """
            SELECT *
            FROM admin_knowledge
            WHERE demo_id = ?
            ORDER BY id DESC
            """,
            (demo_id,),
        ).fetchall()

    else:
        rows = db.execute(
            """
            SELECT *
            FROM admin_knowledge
            ORDER BY id DESC
            """
        ).fetchall()

    db.close()

    return {
        "items": [
            dict(row)
            for row in rows
        ]
    }


# ============================================================
# ADMIN: FACTS
# ============================================================

@app.get("/admin/facts")
def admin_facts(
    demo_id: str,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    demo_id = validate_demo(
        demo_id
    )

    return {
        "demo_id": demo_id,
        "items": load_jsonl_file(
            demo_file(
                demo_id,
                "facts.jsonl",
            )
        ),
    }


# ============================================================
# ADMIN: FAQS
# ============================================================

@app.get("/admin/faqs")
def admin_faqs(
    demo_id: str,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    demo_id = validate_demo(
        demo_id
    )

    return {
        "demo_id": demo_id,
        "items": load_jsonl_file(
            demo_file(
                demo_id,
                "faqs.jsonl",
            )
        ),
    }


# ============================================================
# ADMIN: STRUCTURED DATA
# ============================================================

@app.get("/admin/structured")
def admin_structured(
    demo_id: str,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    demo_id = validate_demo(
        demo_id
    )

    return {
        "demo_id": demo_id,
        "items": load_structured_data(
            demo_id
        ),
    }


# ============================================================
# ADMIN: CLEAR CACHE
# ============================================================

@app.post("/api/cache/clear")
def clear_cache(
    authorization: Optional[str] = Header(
        default=None
    ),
):
    verify_admin_token(
        authorization
    )

    db = get_db()

    db.execute(
        "DELETE FROM cached_answers"
    )

    db.commit()
    db.close()

    return {
        "success": True,
        "message": "Cache cleared.",
    }


# ============================================================
# VOICE PLACEHOLDER
# ============================================================

@app.post("/api/voice/placeholder")
def voice_placeholder():
    return {
        "success": True,
        "status": "placeholder",
        "message": (
            "Voice infrastructure is reserved "
            "for a future implementation."
        ),
    }


# ============================================================
# END
# ============================================================
