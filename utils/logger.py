"""
utils/logger.py
---------------
Colored timestamped console logger.
"""

from datetime import datetime

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GREY   = "\033[90m"
BLUE   = "\033[94m"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def info(tag: str, msg: str) -> None:
    print(f"{GREY}[{_now()}]{RESET} {CYAN}{tag:<14}{RESET} {msg}")


def success(tag: str, msg: str) -> None:
    print(f"{GREY}[{_now()}]{RESET} {GREEN}{tag:<14}{RESET} ✅ {msg}")


def warning(tag: str, msg: str) -> None:
    print(f"{GREY}[{_now()}]{RESET} {YELLOW}{tag:<14}{RESET} ⚠️  {msg}")


def error(tag: str, msg: str) -> None:
    print(f"{GREY}[{_now()}]{RESET} {RED}{tag:<14}{RESET} ❌ {msg}")


def section(title: str) -> None:
    bar = "─" * 50
    print(f"\n{BLUE}{BOLD}  ── {title} {bar[:max(0,46-len(title))]}{RESET}\n")