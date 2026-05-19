"""
api.py
------
FastAPI server — exposes the AI pipeline over HTTP for the EduNexus backend.
Run:
  $ uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Does NOT replace main.py. The CLI (python main.py) still works as before.
"""

import os
import uuid
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import BACKEND_WEBHOOK_URL, SHARED_SECRET, validate_config
from session import LectureSession
from utils import logger

# ── Validate environment on startup ──────────────────────────────────────────
try:
    validate_config()
except EnvironmentError as e:
    print(f"\n{e}\n")
    raise SystemExit(1)

app = FastAPI(title="EduNexus AI Service")

# ── Session registry ──────────────────────────────────────────────────────────
# Maps sessionId → { "lecture": LectureSession | None, "meetingId": str, "status": str }
# Lives here only — has nothing to do with session.py
active_sessions: dict = {}


# ── Request / Response models ─────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    meetingId: str
    instructorId: str

class SessionResponse(BaseModel):
    sessionId: str


# =============================================================================
#  ENDPOINT 1 — Start a session
#  Backend calls this when an instructor starts a meeting.
#  We register the session but wait for the PDF before starting the pipeline.
# =============================================================================
@app.post("/ai/sessions/start", response_model=SessionResponse)
async def start_session(body: StartSessionRequest):
    session_id = str(uuid.uuid4())

    active_sessions[session_id] = {
        "lecture":   None,
        "meetingId": body.meetingId,
        "status":    "waiting_for_pdf",
    }

    logger.info("API", f"Session registered: {session_id} (meeting: {body.meetingId})")
    return {"sessionId": session_id}


# =============================================================================
#  ENDPOINT 2 — Stop a session
#  Backend calls this when the meeting ends.
# =============================================================================
@app.post("/ai/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    entry = active_sessions[session_id]
    lecture: LectureSession | None = entry["lecture"]

    if lecture is not None:
        lecture._stop()     # your existing _stop() → recorder.stop() + final report

    entry["status"] = "stopped"
    logger.info("API", f"Session stopped: {session_id}")
    return {"status": "stopped"}


# =============================================================================
#  ENDPOINT 3 — Upload the course PDF
#  Backend calls this after the instructor uploads the lecture PDF.
#  This creates the LectureSession and starts the full pipeline:
#  load_pdf_text → prepare_pdf_embeddings → MCQ timer loop
# =============================================================================
@app.post("/ai/sessions/{session_id}/pdf")
async def upload_pdf(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if active_sessions[session_id]["status"] == "stopped":
        raise HTTPException(status_code=400, detail="Session is already stopped")

    # Save the PDF to disk
    os.makedirs("uploads", exist_ok=True)
    pdf_path = f"uploads/{session_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    meeting_id = active_sessions[session_id]["meetingId"]

    # Create LectureSession in API mode:
    #   - on_mcq_ready fires instead of _present_question()
    #   - recorder.start() is skipped (audio comes from the backend)
    lecture = LectureSession(
        pdf_path=pdf_path,
        on_mcq_ready=lambda mcq, focus_log: _handle_mcq_ready(
            session_id, meeting_id, mcq, focus_log
        ),
    )

    active_sessions[session_id]["lecture"] = lecture
    active_sessions[session_id]["status"]  = "active"

    # Start the pipeline in the background so this endpoint returns instantly.
    # lecture.start() runs: load_pdf → prepare_pdf_embeddings → MCQ timer loop
    background_tasks.add_task(lecture.start)

    logger.success("API", f"PDF received — pipeline started for session {session_id}")
    return {"status": "pipeline started"}


# =============================================================================
#  ENDPOINT 4 — Receive an audio chunk
#  Backend calls this every ~30 seconds with a WAV chunk from the instructor's mic.
#  We feed it directly into the existing pipeline via _on_new_chunk(wav_path).
#
#  _on_new_chunk() runs your full per-chunk pipeline:
#  TranscriptionBuffer.add_chunk() → IntelligentMemory → TopicTracker
# =============================================================================
@app.post("/ai/sessions/{session_id}/audio-chunk")
async def receive_audio_chunk(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sequence: int = 0,
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if active_sessions[session_id]["status"] == "stopped":
        raise HTTPException(status_code=400, detail="Session is already stopped")

    entry = active_sessions[session_id]
    lecture: LectureSession | None = entry["lecture"]

    if lecture is None:
        raise HTTPException(status_code=400, detail="PDF not uploaded yet — pipeline not started")

    # Reject files over 25MB (OpenAI Whisper hard limit)
    MAX_BYTES = 25 * 1024 * 1024
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio chunk too large ({len(audio_bytes) // 1024 // 1024}MB). Max is 25MB.")

    # Save the WAV chunk — same format AudioRecorder produces: 16kHz, mono, int16
    os.makedirs("chunks", exist_ok=True)
    wav_path = f"chunks/{session_id}_{sequence}.wav"
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)

    # Feed into the existing pipeline
    background_tasks.add_task(lecture._on_new_chunk, wav_path)

    return {"status": "received", "sequence": sequence}


# =============================================================================
#  MCQ CALLBACK
#  Called from inside LectureSession._run_focus_check() in API mode,
#  replacing the terminal _present_question() call.
#
#  Runs inside a background thread (the MCQ timer loop), so we use
#  synchronous httpx — NOT async httpx.
# =============================================================================
def _handle_mcq_ready(session_id: str, meeting_id: str, mcq: dict, focus_log: list) -> None:

    # mcq keys (from MCQQuiz/generator.py generate_mcq()):
    #   question (str), choices (list[str]), correct_answer (str)
    try:
        correct_index = mcq["choices"].index(mcq["correct_answer"])
    except ValueError:
        correct_index = 0

    mcq_payload = {
        "question":     mcq["question"],
        "choices":      mcq["choices"],
        "correctIndex": correct_index,
    }

    # Engagement score from focus_log
    # focus_log entries (from MCQQuiz/evaluator.py evaluate_answer()):
    #   { question, selected_answer, correct_answer, is_correct (bool), focus_status }
    total   = len(focus_log)
    focused = sum(1 for r in focus_log if r["is_correct"])
    score   = round(focused / total, 2) if total > 0 else 0.0

    engagement_payload = {
        "overallScore": score,
        "quizAccuracy": score,
        "responseRate": 1.0,
    }

    _send_to_backend(session_id, meeting_id, "mcq",             mcq_payload)
    _send_to_backend(session_id, meeting_id, "engagementScore", engagement_payload)


def _send_to_backend(session_id: str, meeting_id: str, result_type: str, payload: dict) -> None:
    """Synchronous webhook call — runs inside a thread, not an async context."""
    body = {
        "sessionId": session_id,
        "meetingId": meeting_id,
        "type":      result_type,
        "payload":   payload,
    }
    headers = {
        "X-AI-Secret":  SHARED_SECRET,
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=10) as client:
            response = client.post(BACKEND_WEBHOOK_URL, json=body, headers=headers)
            logger.success("API", f"Sent '{result_type}' to backend → HTTP {response.status_code}")
    except Exception as e:
        logger.error("API", f"Webhook failed for '{result_type}': {e}")