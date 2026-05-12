#!/usr/bin/env python3
"""NeMo-Gym side-car proxy for codex.

Codex CLI doesn't expose hooks to inspect raw LLM request/response payloads.
We fix that by pointing codex's `base_url` at this proxy (running on
127.0.0.1:<free-port>); the proxy forwards each request to NeMo-Gym's model
server and mirrors the request + response to disk as
`<completionsDir>/<model>-<session>-<turn>-<ts>.json` in the
openhands-compatible shape. This means the existing gym-side trajectory
extractor (`get_openhands_trajectory_from_completions`) picks the dump up
without any per-harness branching.

Wire APIs handled:
  * `/v1/chat/completions` (vllm_model)
  * `/v1/responses`        (openai_model — Responses API)

Both keep `prompt_token_ids` / `generation_token_ids` / `generation_log_probs`
under `provider_specific_fields` in the response. We extract them and stash
them at the top level of the dumped JSON so the gym side can read them
exactly like an openhands per-turn dump.

Stdlib-only (http.server + urllib) so the proxy works inside any SWE-bench
SIF that has python3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TOKEN_ID_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
SESSION_HEADER_CANDIDATES = ("x-session-affinity", "x-codex-session", "x-session-id")
PARENT_SESSION_HEADER_CANDIDATES = ("x-parent-session-id", "x-codex-parent-session")


class ProxyState:
    """Shared state: dump dir, instance_id, model_name, per-session turn counter."""

    def __init__(
        self,
        upstream_url: str,
        completions_dir: Path,
        instance_id: str,
        model_name: str,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.completions_dir = completions_dir
        self.instance_id = instance_id
        self.model_name = model_name
        self.lock = threading.Lock()
        self._turn_counters: dict[str, int] = {}
        completions_dir.mkdir(parents=True, exist_ok=True)

    def next_turn(self, session_id: str) -> int:
        with self.lock:
            n = self._turn_counters.get(session_id, -1) + 1
            self._turn_counters[session_id] = n
            return n


def _pick_session_id(headers: dict[str, str]) -> str:
    for h in SESSION_HEADER_CANDIDATES:
        v = headers.get(h) or headers.get(h.title())
        if v:
            return v
    return "main"


def _pick_parent_session_id(headers: dict[str, str]) -> str | None:
    for h in PARENT_SESSION_HEADER_CANDIDATES:
        v = headers.get(h) or headers.get(h.title())
        if v:
            return v
    return None


def _safe_for_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _extract_provider_specific_fields(response_obj: object) -> dict[str, object]:
    """Pull token-ID fields out of a Chat-Completions or Responses payload.

    Chat Completions: `response["choices"][0]["message"]` carries the fields.
    Responses API: token IDs may be at the top level under `prompt_token_ids`
    etc., or nested under `provider_specific_fields`. We check both.
    """
    out: dict[str, object] = {}
    if not isinstance(response_obj, dict):
        return out

    # Chat Completions
    try:
        msg = response_obj["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        msg = None
    if isinstance(msg, dict):
        for f in TOKEN_ID_FIELDS:
            if f in msg:
                out[f] = msg[f]

    # Responses API — top-level or nested
    psf = response_obj.get("provider_specific_fields")
    if isinstance(psf, dict):
        for f in TOKEN_ID_FIELDS:
            if f in psf:
                out[f] = psf[f]
    for f in TOKEN_ID_FIELDS:
        if f not in out and f in response_obj:
            out[f] = response_obj[f]

    return out


def _normalize_messages(request_obj: object) -> list[object]:
    """Best-effort extraction of messages from either wire API."""
    if not isinstance(request_obj, dict):
        return []
    # Chat Completions
    if isinstance(request_obj.get("messages"), list):
        return request_obj["messages"]
    # Responses — `input` is the canonical list
    if isinstance(request_obj.get("input"), list):
        return request_obj["input"]
    return []


def _extract_request_tools(request_obj: object) -> list[object]:
    if not isinstance(request_obj, dict):
        return []
    tools = request_obj.get("tools")
    return tools if isinstance(tools, list) else []


def _sanitize_request_body(body: bytes) -> bytes:
    """Drop tools whose `type` isn't `function` from the outgoing request.

    Codex 0.x adds `{"type": "web_search", ...}` and other hosted-tool entries
    to its tool list. The gym's openai_model server validates request bodies
    with a strict pydantic schema that only accepts FunctionToolParam (`type:
    "function"`), so the whole request 422s when a non-function tool is
    present.

    We parse the JSON, strip any tool whose `type` is set and not `function`,
    and re-serialize. Returns the original body if it isn't valid JSON or
    doesn't have a tools list — this keeps the proxy resilient to wire-format
    variations.
    """
    if not body:
        return body
    try:
        obj = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(obj, dict):
        return body
    tools = obj.get("tools")
    if not isinstance(tools, list) or not tools:
        return body

    cleaned: list[object] = []
    dropped: list[str] = []
    for t in tools:
        if isinstance(t, dict):
            t_type = t.get("type")
            # Keep tools with type == "function" OR no type set (some wire
            # variants leave `type` implicit for the OpenAI Chat path).
            if t_type is None or t_type == "function":
                cleaned.append(t)
                continue
            dropped.append(str(t_type))
        else:
            cleaned.append(t)

    if len(cleaned) == len(tools):
        return body  # nothing to strip — keep original bytes

    obj["tools"] = cleaned
    if dropped:
        sys.stderr.write(f"[proxy] stripped non-function tools: {dropped}\n")
    return json.dumps(obj).encode("utf-8")


def make_handler(state: ProxyState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # Quiet the default access log; we already log via _dump_completion.
        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("[proxy] " + (fmt % args) + "\n")

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or "0")
            return self.rfile.read(length) if length > 0 else b""

        def _forward(self, body: bytes) -> tuple[int, dict[str, str], bytes]:
            upstream = f"{state.upstream_url}{self.path}"
            headers: dict[str, str] = {}
            for k in self.headers.keys():
                if k.lower() in ("host", "content-length", "connection"):
                    continue
                headers[k] = self.headers[k]
            req = Request(upstream, data=body, method=self.command, headers=headers)
            try:
                with urlopen(req, timeout=600) as resp:
                    return resp.status, dict(resp.getheaders()), resp.read()
            except HTTPError as e:
                return e.code, dict(e.headers or {}), e.read() if hasattr(e, "read") else b""
            except URLError as e:
                msg = f'{{"error":"upstream URLError: {e}"}}'.encode()
                return 502, {"Content-Type": "application/json"}, msg

        def _dump_completion(
            self,
            request_obj: object,
            response_obj: object,
            session_id: str,
            parent_session_id: str | None,
        ) -> None:
            turn = state.next_turn(session_id)
            provider_specific = _extract_provider_specific_fields(response_obj)
            messages = _normalize_messages(request_obj)
            tools = _extract_request_tools(request_obj)

            kwargs: dict[str, object] = {"tools": tools}
            if isinstance(request_obj, dict):
                for k in ("temperature", "top_p", "max_tokens", "max_output_tokens", "stop", "seed"):
                    if k in request_obj:
                        kwargs[k] = request_obj[k]

            payload = {
                "messages": messages,
                "response": response_obj,
                "provider_specific_fields": provider_specific,
                "kwargs": kwargs,
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "turn": turn,
                "timestamp": time.time(),
            }

            safe_model = _safe_for_filename(state.model_name or "model")
            safe_session = _safe_for_filename(session_id)
            turn_str = str(turn).zfill(4)
            fname = f"{safe_model}-{safe_session}-{turn_str}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}.json"
            fpath = state.completions_dir / fname
            tmp = fpath.with_suffix(fpath.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, fpath)
            sys.stderr.write(f"[proxy] dumped {fpath.name} (session={session_id[:16]} turn={turn})\n")

        def do_POST(self) -> None:
            body = self._read_body()
            headers_lower = {k.lower(): v for k, v in self.headers.items()}
            session_id = _pick_session_id(headers_lower)
            parent_session_id = _pick_parent_session_id(headers_lower)

            # Sanitize the outgoing payload. Codex injects built-in tools whose
            # `type` is not `"function"` (e.g. `web_search`, hosted tools); the
            # gym's openai_model server's pydantic schema only accepts
            # FunctionToolParam and 422s the whole request when it sees any
            # other type. Drop those before forwarding so the request validates.
            if any(p in self.path for p in ("/chat/completions", "/responses")):
                body = _sanitize_request_body(body)

            status, resp_headers, resp_body = self._forward(body)

            # Forward to client.
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() in ("transfer-encoding", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            try:
                self.wfile.write(resp_body)
            except (BrokenPipeError, ConnectionResetError):
                pass

            # Mirror to disk only for chat / responses calls.
            if 200 <= status < 300 and any(
                p in self.path for p in ("/chat/completions", "/responses")
            ):
                try:
                    request_obj = json.loads(body.decode("utf-8")) if body else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_obj = None
                try:
                    response_obj = json.loads(resp_body.decode("utf-8")) if resp_body else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response_obj = None
                try:
                    self._dump_completion(request_obj, response_obj, session_id, parent_session_id)
                except Exception as e:
                    sys.stderr.write(f"[proxy] dump failed: {e}\n")

        def do_GET(self) -> None:
            # Health probe — useful for the bench wrapper's readiness loop.
            if self.path in ("/health", "/healthz", "/"):
                payload = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self._forward_and_passthrough_get()

        def _forward_and_passthrough_get(self) -> None:
            status, resp_headers, resp_body = self._forward(b"")
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() in ("transfer-encoding", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            try:
                self.wfile.write(resp_body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-url", required=True, help="NeMo-Gym model server base URL")
    parser.add_argument("--completions-dir", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    state = ProxyState(
        upstream_url=args.upstream_url,
        completions_dir=Path(args.completions_dir),
        instance_id=args.instance_id,
        model_name=args.model_name,
    )
    handler = make_handler(state)
    server = ThreadingHTTPServer(("127.0.0.1", args.listen_port), handler)
    sys.stderr.write(
        f"[proxy] listening on 127.0.0.1:{args.listen_port} -> {state.upstream_url} "
        f"(instance={state.instance_id}, model={state.model_name})\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
