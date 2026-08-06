"""Builds an in-memory ZIP from the artifacts a pipeline run generates."""
import io
import zipfile
from typing import Dict, Union


def build_files_zip(files: Dict[str, Union[str, bytes]]) -> bytes:
    """Zip a flat mapping of relative path -> text/bytes content (e.g. pipeline output artifacts)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()
