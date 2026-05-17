"""
main.py
-------
Entry point for the AI Lecture Monitoring System (EDUNEXUS).

Run:
  $ python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import validate_config
from session import LectureSession
from utils import logger


def main() -> None:
    try:
        validate_config()
    except EnvironmentError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    logger.section("EDUNEXUS — AI Lecture Monitoring System")
    print("  Real-time lecture monitoring with intelligent semantic memory.")
    print("  MCQs are generated from your lecture PDF every interval.\n")

    while True:
        pdf_path = input("  📄  Enter path to lecture PDF: ").strip()
        if not pdf_path:
            print("  ⚠️   Path cannot be empty.")
        elif not os.path.exists(pdf_path):
            print(f"  ⚠️   File not found: {pdf_path}")
        elif not pdf_path.lower().endswith(".pdf"):
            print("  ⚠️   Must be a .pdf file.")
        else:
            break

    print()
    session = LectureSession(pdf_path=pdf_path)
    session.start()


if __name__ == "__main__":
    main()