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


def _maybe_disable_streaming(path: str, body: bytes) -> tuple[bool, bytes]:
    """Detect client-side streaming intent and rewrite the body for the gym.

    Returns (client_wanted_stream, body_for_upstream).

    NeMo-Gym's openai_model server only accepts `stream: false` (its pydantic
    schema enforces `Literal[False]`). Codex hardcodes `stream: true` for both
    /v1/responses and /v1/chat/completions and parses the response as SSE.
    Force-flip `stream` to false on the wire so the gym validates the request;
    we'll synthesize an SSE stream back from the non-streaming JSON.
    """
    if not body or not any(p in path for p in ("/chat/completions", "/responses")):
        return False, body
    try:
        obj = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, body
    if not isinstance(obj, dict):
        return False, body
    if obj.get("stream") is not True:
        return False, body
    obj["stream"] = False
    return True, json.dumps(obj).encode("utf-8")


def _sse_event(event_type: str, data_obj: object) -> bytes:
    """Format a single SSE event in the OpenAI Responses API style."""
    payload = json.dumps(data_obj, ensure_ascii=False)
    return (f"event: {event_type}\ndata: {payload}\n\n").encode("utf-8")


def _synthesize_responses_sse(response_obj: dict[str, object]) -> bytes:
    """Build an SSE event stream that mirrors what codex would have read.

    Codex's Responses-API stream parser walks the events to assemble the final
    `Response`. The minimum-viable sequence for a complete, non-streamed
    response is `response.created` followed by `response.completed`, each
    carrying the full Response object. Codex's parser tolerates the missing
    intermediate `response.output_*` deltas as long as `response.completed`
    contains the assembled `output` array.

    We emit:
      event: response.created       data: {type, sequence_number, response}
      event: response.completed     data: {type, sequence_number, response}

    `response_obj` has a `status` field from the gym ("completed" usually);
    we surface that as-is. If absent, we set "completed" on the second event.
    """
    # Make sure the response is marked completed on the final event regardless
    # of what the gym sent.
    completed_response = dict(response_obj)
    completed_response.setdefault("status", "completed")

    parts: list[bytes] = []
    parts.append(
        _sse_event(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": response_obj,
            },
        )
    )
    parts.append(
        _sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": completed_response,
            },
        )
    )
    return b"".join(parts)


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

            # Codex hardcodes stream=true (client.rs:756), but NeMo-Gym's
            # openai_model server only accepts stream=false. Detect the client's
            # streaming intent, force non-streaming on the wire, then synthesize
            # an SSE event stream back from the JSON we get. Codex parses the
            # SSE events as if the upstream had streamed natively.
            client_wants_stream, body_for_upstream = _maybe_disable_streaming(self.path, body)

            status, resp_headers, resp_body = self._forward(body_for_upstream)

            if client_wants_stream and 200 <= status < 300 and "/responses" in self.path:
                # Wrap the non-streaming JSON response as SSE events for codex.
                try:
                    response_obj = json.loads(resp_body.decode("utf-8")) if resp_body else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response_obj = None
                if isinstance(response_obj, dict):
                    sse_bytes = _synthesize_responses_sse(response_obj)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(sse_bytes)))
                    self.end_headers()
                    try:
                        self.wfile.write(sse_bytes)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    # Fall through to disk-mirror below using the parsed JSON.
                    resp_for_dump = response_obj
                else:
                    # Couldn't parse — passthrough the original (likely error) body.
                    self._passthrough_response(status, resp_headers, resp_body)
                    resp_for_dump = None
            else:
                self._passthrough_response(status, resp_headers, resp_body)
                resp_for_dump = None

            # Mirror to disk only for successful chat / responses calls.
            if 200 <= status < 300 and any(
                p in self.path for p in ("/chat/completions", "/responses")
            ):
                try:
                    request_obj = json.loads(body.decode("utf-8")) if body else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_obj = None
                if resp_for_dump is None:
                    try:
                        resp_for_dump = json.loads(resp_body.decode("utf-8")) if resp_body else None
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        resp_for_dump = None
                try:
                    self._dump_completion(request_obj, resp_for_dump, session_id, parent_session_id)
                except Exception as e:
                    sys.stderr.write(f"[proxy] dump failed: {e}\n")

        def _passthrough_response(
            self, status: int, resp_headers: dict[str, str], resp_body: bytes
        ) -> None:
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
