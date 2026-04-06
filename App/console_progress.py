"""Single-line status updates (carriage return + flush + pad) for Windows terminals."""

import sys

_DEFAULT_WIDTH = 100


def status_line(message: str, width: int = _DEFAULT_WIDTH) -> None:
    """Overwrite the current line with message (no trailing newline)."""
    sys.stdout.write(f"\r{message:<{width}}")
    sys.stdout.flush()


def status_line_done(message: str, width: int = _DEFAULT_WIDTH) -> None:
    """Finish a status sequence: clear width, print message, newline."""
    sys.stdout.write(f"\r{message:<{width}}\n")
    sys.stdout.flush()
