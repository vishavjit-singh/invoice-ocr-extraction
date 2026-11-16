"""A tiny synchronous ASGI client, so API tests need neither httpx nor a running server."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlencode


class Response:
    def __init__(self, status: int, headers: list, body: bytes):
        self.status_code = status
        self.headers = {k.decode().lower(): v.decode() for k, v in headers}
        self.content = body

    def json(self):
        return json.loads(self.content)


def call(app, method: str, path: str, body: bytes = b"", headers: dict | None = None,
         query: dict | None = None) -> Response:
    headers = {"host": "test", **(headers or {})}
    if body:
        headers.setdefault("content-length", str(len(body)))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": urlencode(query or {}).encode(),
             "root_path": "", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": ("127.0.0.1", 1234), "server": ("test", 80)}
    sent = {"done": False}
    out: dict = {"status": None, "headers": [], "body": b""}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return Response(out["status"], out["headers"], out["body"])


def multipart(field: str, filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = "----invoiceocr-test-boundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"
