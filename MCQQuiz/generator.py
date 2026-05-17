"""
MCQQuiz/generator.py
--------------------
Generates context-aware MCQs using GPT-4o.

Input context comes from IntelligentMemory (not raw transcript slices).
RAG-retrieved PDF chunks are used for factual grounding.
"""

import json
import re
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MODEL
from utils import logger

_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_mcq(
    transcript: str,
    context: str,
    topic: str = "",
) -> dict | None:
    """
    Generate one MCQ from intelligent memory context + RAG PDF chunks.

    Args:
        transcript: Recent raw transcript (fallback grounding).
        context:    Rich context from IntelligentMemory.get_context_for_mcq().
        topic:      Current topic label from TopicTracker (optional).

    Returns:
        MCQ dict with keys: question, choices, correct_answer
    """
    topic_hint = f"Current lecture topic: {topic}\n\n" if topic else ""

    prompt = f"""
You are an educational AI assistant generating a focus-check MCQ.

{topic_hint}--- LECTURE CONTEXT (from course material + memory) ---
{context}

--- RECENT TRANSCRIPT ---
{transcript[-600:]}

Task:
- Generate 1 high-quality MCQ that tests conceptual understanding.
- The question must be directly grounded in the lecture content above.
- Provide exactly 4 answer choices.
- Only 1 choice is correct.

Return ONLY valid JSON (no markdown, no explanation):

{{
  "question": "...",
  "choices": ["...", "...", "...", "..."],
  "correct_answer": "..."
}}
"""

    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Return only valid JSON. No explanation."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            raw  = response.choices[0].message.content.strip()
            mcq  = _parse_json(raw)
            if mcq:
                logger.success("Generator", f"MCQ generated: {mcq['question'][:60]}...")
                return mcq

        except Exception as e:
            logger.error("Generator", f"Attempt {attempt+1} failed: {e}")

    logger.error("Generator", "MCQ generation failed after 3 attempts.")
    return None


def _parse_json(text: str) -> dict | None:
    try:
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
        data    = json.loads(cleaned)

        if len(data.get("choices", [])) != 4:
            logger.warning("Generator", "MCQ has wrong number of choices.")
            return None

        # Ensure correct_answer is one of the choices
        if data["correct_answer"] not in data["choices"]:
            logger.warning("Generator", "correct_answer not in choices — fixing.")
            data["correct_answer"] = data["choices"][0]

        return data
    except Exception as e:
        logger.error("Generator", f"JSON parse failed: {e}")
        return None