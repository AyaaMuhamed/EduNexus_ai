"""
MCQQuiz/evaluator.py
--------------------
Evaluates a student's MCQ answer and returns a focus result.
"""


def evaluate_answer(mcq: dict, selected_answer: str) -> dict:
    correct  = mcq.get("correct_answer", "").strip()
    selected = selected_answer.strip()

    # Case-insensitive comparison
    is_correct = selected.lower() == correct.lower()

    return {
        "question":        mcq.get("question"),
        "selected_answer": selected,
        "correct_answer":  correct,
        "is_correct":      is_correct,
        "focus_status":    "focused" if is_correct else "not_focused",
    }