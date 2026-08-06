import io
import sys
from contextlib import contextmanager


@contextmanager
def suppress_io():
    """Redirect stdout/stderr/stdin for the duration of a block.

    Used to silence CrewAI/LLM library console chatter that would corrupt
    Streamlit's output stream.
    """
    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = io.StringIO("n\n")
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def suppress_stderr():
    """Redirect stderr only."""
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = old_stderr
