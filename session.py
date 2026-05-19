"""
session.py
----------
Main orchestrator of the AI Lecture Monitoring System.

Upgraded pipeline:
  ┌──────────────────────────────────────────────────────────────┐
  │  PDF Loader ──► pdf_text                                     │
  │       │                                                      │
  │  RAG_Matcher.prepare_pdf_embeddings()                        │
  │       │                                                      │
  │  AudioRecorder ──► WAV chunks ──► TranscriptionBuffer        │
  │       │                                                      │
  │       │  (each new segment)                                  │
  │       ▼                                                      │
  │  IntelligentMemory.add_transcript_chunk()                    │
  │       │                                                      │
  │  TopicTracker.ingest()                                       │
  │       │                                                      │
  │       │  (every MCQ_INTERVAL_SECONDS)                        │
  │       ▼                                                      │
  │  memory.get_context_for_mcq()  ──► rich intelligent context  │
  │  RAG retrieval                 ──► PDF grounding             │
  │       │                                                      │
  │  generate_mcq(transcript, context, topic)                    │
  │       │                                                      │
  │  Student answers ──► evaluate_answer()                       │
  │       │                                                      │
  │  Focus log ──► Final report                                  │
  └──────────────────────────────────────────────────────────────┘

Threading:
  Main thread   : MCQ timer loop + student input
  Background    : AudioRecorder (continuous, daemon)
  Memory/Topics : Updated on each buffer segment (recorder thread)
"""

import sys
import time
import threading
import numpy as np

from config import MCQ_INTERVAL_SECONDS
from pdf_loader.loader import load_pdf_text
from utils.RAG_Matcher import prepare_pdf_embeddings, find_relevant_pdf_chunks
from utils.embedding_cache import load_or_create_embeddings
from utils.RAG_Matcher import embed, chunk_pdf
from Buffering import TranscriptionBuffer
from MCQQuiz.generator import generate_mcq
from MCQQuiz.evaluator import evaluate_answer
from Features.intelligent_memo import IntelligentMemory
from Features.topic_tracker import TopicTracker
from utils import logger


class LectureSession:
    """Manages one full lecture session end-to-end."""

    def __init__(self, pdf_path: str, on_mcq_ready=None):
        self.pdf_path         = pdf_path
        self.pdf_chunks:      list[str]   = []
        self.pdf_embeddings:  np.ndarray  = np.array([])
        self.is_running       = False
        self.focus_log:       list[dict]  = []

        # Core pipeline components
        self.buffer   = TranscriptionBuffer()
        self.memory   = IntelligentMemory()
        self.topics   = TopicTracker()
        if on_mcq_ready is None:
            # CLI mode only — don't import sounddevice in API mode
            from recorder import AudioRecorder
            self.recorder = AudioRecorder(on_chunk_ready=self._on_new_chunk)
        else:
            self.recorder = None
        self._lock = threading.Lock()
        self.on_mcq_ready = on_mcq_ready  # None = CLI mode, callable = API mode

    # ── Public ────────────────────────────────────────────────────

    def start(self) -> None:
        """Load PDF → prepare RAG → start recording → run MCQ loop."""

        # 1. Load PDF
        logger.info("Session", "Loading PDF...")
        pdf_text = load_pdf_text(self.pdf_path)
        if not pdf_text:
            logger.error("Session", "Could not load PDF. Aborting.")
            return
        logger.success("Session", f"PDF loaded ({len(pdf_text):,} chars).")

        # 2. Prepare embeddings (cached on disk)
        logger.info("Session", "Preparing PDF embeddings (disk cache)...")
        self.pdf_chunks, self.pdf_embeddings = load_or_create_embeddings(
            pdf_text=pdf_text,
            embed_fn=embed,
            chunk_fn=chunk_pdf,
        )
        logger.success("Session", f"RAG ready — {len(self.pdf_chunks)} chunks.")

        # 3. Start audio recording (CLI only — API mode receives chunks from backend)
        if not self.on_mcq_ready:
            self.recorder.start()
        self.is_running = True
        logger.success("Session", "Lecture is LIVE.\n")
        
        # 4. MCQ loop on main thread
        try:
            self._mcq_loop()
        except KeyboardInterrupt:
            logger.info("Session", "Lecture ended by user.")
        finally:
            self._stop()

    # ── Chunk ingestion (called from recorder thread) ─────────────

    def _on_new_chunk(self, wav_path: str) -> None:
        """
        Full per-chunk pipeline:
          transcribe → buffer → memory → topic tracker
        """
        # TranscriptionBuffer handles transcription internally
        if not self.is_running:
            return          # session was stopped, discard this chunk
        self.buffer.add_chunk(wav_path)

        # Feed latest segment into intelligent memory + topic tracker
        segments = self.buffer.get_structured_transcript()
        if segments:
            latest_text = segments[-1]["text"]
            if latest_text:
                self.memory.add_transcript_chunk(
                    chunk=latest_text,
                    pdf_chunks=self.pdf_chunks,
                    pdf_embeddings=self.pdf_embeddings,
                )
                self.topics.ingest(latest_text)

    # ── MCQ loop ──────────────────────────────────────────────────

    def _mcq_loop(self) -> None:
        minutes = MCQ_INTERVAL_SECONDS // 60
        logger.info("Session", f"First focus check in {minutes} minute(s).")

        while self.is_running:
            for _ in range(MCQ_INTERVAL_SECONDS):
                if not self.is_running:
                    return
                time.sleep(1)

            if self.is_running:
                self._run_focus_check()

    def _run_focus_check(self) -> None:
        """Full focus-check pipeline using intelligent memory context."""
        logger.section("FOCUS CHECK")

        transcript = self.buffer.get_full_transcript()
        if not transcript.strip():
            logger.warning("Session", "No transcript yet — skipping.")
            return

        word_count = len(transcript.split())
        logger.info("Session", f"Transcript: {word_count} words | "
                               f"{self.buffer.get_segment_count()} segments")

        if word_count < 50:
            logger.warning("Session", "Transcript too short — skipping.")
            return

        # ── Build intelligent context ──────────────────────────────
        memory_context = self.memory.get_context_for_mcq()
        rag_context    = find_relevant_pdf_chunks(
            transcript=transcript,
            pdf_chunks=self.pdf_chunks,
            pdf_embeddings=self.pdf_embeddings,
        )
        full_context = (
            f"{memory_context}\n\n"
            f"[PDF Reference Material]\n{rag_context}"
        )

        # ── Current topic ──────────────────────────────────────────
        current_topic = self.topics.get_current_topic()
        logger.info("Session", f"Current topic: {current_topic}")

        # ── Generate MCQ ───────────────────────────────────────────
        mcq = generate_mcq(
            transcript=transcript,
            context=full_context,
            topic=current_topic,
        )

        if not mcq:
            logger.error("Session", "MCQ generation failed — skipping.")
            return

        # ── Present to student (CLI) or send to backend (API) ──────────────────
        if self.on_mcq_ready:
            # API mode: fire MCQ to backend via webhook (defined in api.py)
            self.on_mcq_ready(mcq, self.focus_log)
        else:
            # CLI mode: original terminal interaction (unchanged)
            selected = self._present_question(mcq)
            result   = evaluate_answer(mcq=mcq, selected_answer=selected)
            self.focus_log.append(result)
            self._print_running_score()
        # ── Log topic timeline ─────────────────────────────────────
        timeline = self.topics.get_timeline_summary()
        if timeline:
            logger.info("Session", f"Lecture timeline:\n{timeline}")

    def _present_question(self, mcq: dict) -> str:
        print(f"\n  ❓  {mcq['question']}\n")
        for i, choice in enumerate(mcq["choices"], start=1):
            print(f"      {i}. {choice}")
        print()

        while True:
            try:
                raw   = input("  👉  Your answer (1–4): ").strip()
                index = int(raw) - 1
                if 0 <= index <= 3:
                    return mcq["choices"][index]
                print("  ⚠️   Please enter 1, 2, 3, or 4.")
            except (ValueError, EOFError):
                print("  ⚠️   Invalid input.")

    def _print_running_score(self) -> None:
        total   = len(self.focus_log)
        focused = sum(1 for r in self.focus_log if r["is_correct"])
        pct     = int((focused / total) * 100)
        bar     = ("█" * focused) + ("░" * (total - focused))
        print(f"\n  📊  Score: {bar}  {focused}/{total}  ({pct}% focused)\n")

    # ── Shutdown ──────────────────────────────────────────────────

    def _stop(self) -> None:
        if not self.is_running:
            return          # already stopped, don't run twice
        self.is_running = False
        if self.recorder:
            self.recorder.stop()
        self._print_final_report()

    def _print_final_report(self) -> None:
        logger.section("FINAL REPORT")

        if not self.focus_log:
            logger.warning("Session", "No focus checks completed.")
            return

        total   = len(self.focus_log)
        focused = sum(1 for r in self.focus_log if r["is_correct"])
        pct     = int((focused / total) * 100)

        print(f"  Total Checks    : {total}")
        print(f"  Correct Answers : {focused}")
        print(f"  Focus Rate      : {pct}%")
        print(f"  Verdict         : {'✅ Engaged' if pct >= 60 else '⚠️  Needs improvement'}")

        # Topic timeline
        timeline = self.topics.get_timeline_summary()
        if timeline:
            print(f"\n  ── Lecture Topics ────────────────────────────────────")
            print(timeline)

        print(f"\n  ── Detailed Answer Log ───────────────────────────────")
        for i, r in enumerate(self.focus_log, start=1):
            status = "✅ Focused" if r["is_correct"] else "❌ Not Focused"
            print(f"  Q{i}: {status}")
            print(f"       Q        : {r['question'][:70]}...")
            print(f"       Student  : {r['selected_answer']}")
            print(f"       Correct  : {r['correct_answer']}")
            print()