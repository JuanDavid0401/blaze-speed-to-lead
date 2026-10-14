"""
Speed-to-lead pipeline: Facebook lead ad webhook -> Gemini scoring -> Google Sheet -> Discord.

Run locally:
    pip install fastapi uvicorn google-generativeai gspread requests python-dotenv
    uvicorn main:app --reload --port 8000
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
import gspread
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speed-to-lead")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "").strip()
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

SHEET_HEADERS = [
    "timestamp_utc", "name", "email", "phone", "company",
    "number_of_doors", "message", "fit_score", "reason", "draft_reply",
]

app = FastAPI(title="Speed-to-Lead")
_worksheet = None  # cached gspread worksheet


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Lead(BaseModel):
    name: str
    email: str
    phone: str
    company: str
    number_of_doors: int = Field(ge=0)
    message: str = ""


# --------------------------------------------------------------------------- #
# Step 1: LLM evaluation
# --------------------------------------------------------------------------- #
def evaluate_lead(lead: Lead) -> dict:
    try:
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError(f"GOOGLE_API_KEY is missing in {ENV_PATH}")

        genai.configure(api_key=api_key)

        prompt = f"""
        Evaluate this lead and return ONLY a valid JSON object with exactly these 3 keys:
        "fit_score": an integer from 1 to 5 (50-500 doors is 4-5)
        "reason": a short string explaining the score
        "draft_reply": a short SMS reply draft to the customer under 300 chars

        Lead Data:
        Name: {lead.name}
        Email: {lead.email}
        Phone: {lead.phone}
        Company: {lead.company}
        Doors: {lead.number_of_doors}
        Message: {lead.message}
        """

        # Lista de nombres posibles + consulta automática a Google de modelos activos
        candidate_models = [
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
        ]
        try:
            for m in genai.list_models():
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    if (
                        m.name not in candidate_models
                        and "2.5-flash" not in m.name
                        and "1.5" not in m.name
                    ):
                        candidate_models.append(m.name)
        except Exception:
            pass

        response = None
        last_err = None
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json"
                    ),
                )
                log.info("Evaluación exitosa usando el modelo: %s", model_name)
                break
            except Exception as err:
                last_err = err
                continue

        if response is None:
            raise RuntimeError(f"Ningún modelo respondió. Último error: {last_err}")

        data = json.loads(response.text)

        return {
            "fit_score": max(1, min(5, int(data.get("fit_score", 1)))),
            "reason": str(data.get("reason", "")).strip(),
            "draft_reply": str(data.get("draft_reply", "")).strip(),
        }
    except Exception as e:
        log.warning("LLM attempt failed: %s", e)
        raise RuntimeError(f"LLM evaluation failed: {e}")


# --------------------------------------------------------------------------- #
# Step 2: Google Sheets
# --------------------------------------------------------------------------- #
def get_worksheet():
    global _worksheet
    if _worksheet is None:
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
        creds_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json").strip()
        gc = gspread.service_account(filename=creds_file)
        ws = gc.open_by_key(sheet_id).sheet1
        if not ws.row_values(1):  # empty sheet -> add header row
            ws.append_row(SHEET_HEADERS, value_input_option="RAW")
        _worksheet = ws
    return _worksheet


def append_to_sheet(lead: Lead, ai: dict) -> None:
    row = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        lead.name, lead.email, lead.phone, lead.company,
        lead.number_of_doors, lead.message,
        ai["fit_score"], ai["reason"], ai["draft_reply"],
    ]
    try:
        get_worksheet().append_row(row, value_input_option="RAW")
    except Exception:
        global _worksheet
        _worksheet = None  # force a fresh connection next time
        raise


# --------------------------------------------------------------------------- #
# Step 3: Discord notification
# --------------------------------------------------------------------------- #
def notify_discord(lead: Lead, ai: dict) -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not set")

    stars = "⭐" * ai["fit_score"] if ai["fit_score"] else "n/a"
    content = (
        f"**🚨 New lead: {lead.name}** ({lead.company})\n"
        f"**AI fit score:** {ai['fit_score']}/5 {stars}\n"
        f"**Reason:** {ai['reason']}\n\n"
        f"📞 {lead.phone}  |  ✉️ {lead.email}  |  🚪 {lead.number_of_doors} doors\n"
        f"💬 Message: {lead.message or '(none)'}\n\n"
        f"**Drafted SMS reply:**\n> {ai['draft_reply']}"
    )
    resp = requests.post(
        webhook_url,
        json={
            "content": content[:1900],
            "allowed_mentions": {"parse": []},
        },
        timeout=10,
    )
    resp.raise_for_status()


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@app.post("/webhook")
def webhook(lead: Lead):
    log.info("Lead received: %s <%s>", lead.name, lead.email)
    result = {"lead": lead.model_dump(), "ai": None, "sheet": "skipped", "discord": "skipped", "errors": []}

    # 1) Score the lead
    try:
        ai = evaluate_lead(lead)
    except Exception as e:  # noqa: BLE001
        log.exception("LLM step failed")
        result["errors"].append(f"llm: {e}")
        ai = {"fit_score": 0, "reason": "AI evaluation failed - review manually", "draft_reply": ""}
    else:
        result["ai"] = ai

    # 2) Google Sheet
    try:
        append_to_sheet(lead, ai)
        result["sheet"] = "ok"
    except Exception as e:  # noqa: BLE001
        log.exception("Sheets step failed")
        result["sheet"] = "failed"
        result["errors"].append(f"sheet: {repr(e)} | cause: {getattr(e, '__cause__', '')}")

    # 3) Discord
    try:
        notify_discord(lead, ai)
        result["discord"] = "ok"
    except Exception as e:  # noqa: BLE001
        log.exception("Discord step failed")
        result["discord"] = "failed"
        result["errors"].append(f"discord: {e}")

    return result


@app.get("/health")
def health():
    return {"status": "ok"}