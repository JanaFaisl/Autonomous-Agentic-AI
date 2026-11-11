"""SSRF / local-resource guards for user-supplied URLs.

Imports from a URL or database URL are user-controlled, so by default they may
only reach public hosts. Set ALLOW_LOCAL_IMPORTS=1 to permit localhost, private
networks and local sqlite files (e.g. on a developer machine).
"""
import ipaddress
import os
import socket
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests

MAX_REDIRECTS = 3
_REMOTE_DB_SCHEMES = ("postgresql", "postgres", "mysql", "mariadb", "mssql")


def local_imports_allowed() -> bool:
    return os.getenv("ALLOW_LOCAL_IMPORTS", "").strip().lower() in ("1", "true", "yes")


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return False
    return bool(infos)


def check_http_url(url: str) -> Optional[str]:
    """Return an error message if the URL must not be fetched, else None."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "Only http:// and https:// URLs are supported."
    if not local_imports_allowed() and not _host_is_public(p.hostname):
        return "That address is not a public host, so it can't be fetched."
    return None


def safe_get(url: str, timeout: int = 120) -> Tuple[Optional[requests.Response], Optional[str]]:
    """GET with the host re-checked on every redirect hop. Returns (response, error)."""
    for _ in range(MAX_REDIRECTS + 1):
        err = check_http_url(url)
        if err:
            return None, err
        resp = requests.get(url, timeout=timeout, stream=True, allow_redirects=False)
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, resp.headers.get("Location", ""))
            resp.close()
            continue
        return resp, None
    return None, "Too many redirects."


def check_database_url(url: str) -> Optional[str]:
    """Return an error message if the database URL must not be opened, else None."""
    if local_imports_allowed():
        return None
    try:
        p = urlparse(url.strip())
    except ValueError:
        return "That doesn't look like a valid database URL."
    scheme = (p.scheme or "").split("+")[0].lower()
    if scheme not in _REMOTE_DB_SCHEMES:
        return ("Only remote PostgreSQL/MySQL URLs can be imported here. "
                "Local sqlite files are disabled (set ALLOW_LOCAL_IMPORTS=1 to enable).")
    if not p.hostname or not _host_is_public(p.hostname):
        return "That address is not a public host, so it can't be connected to."
    return None
