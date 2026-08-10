"""PROJECT R-5 — the application entry point.

Run it with:

    python3 serve.py

then open http://localhost:8000.

Two jobs, and nothing else:

  * hand out the files in `web/`
  * hold the live sessions, and let the page drive them

The simulation lives entirely on this side (`game/`), which is why it can
be tested without a browser.  The page owns rendering and controls and
never sees an environment or an algorithm — only snapshots.

Only the standard library is used, so the backend adds no dependency any
more than the frontend does.
"""

import json
import mimetypes
import os
import posixpath
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from game import config, rooms
from game.comparison import Comparison
from game.session import IllegalTransition, Session

# WHERE TO LISTEN, LOCALLY AND WHEN HOSTED
#
# A hosting platform hands the process a port in $PORT and expects the server
# to accept connections from outside its container. 127.0.0.1 accepts none of
# them, so a server bound there looks to the platform like a process that
# started and never came up — hence 0.0.0.0, every interface, by default.
#
# $HOST overrides it, for anyone who wants a development server that the rest
# of the local network cannot reach:
#
#     HOST=127.0.0.1 python3 serve.py
#
# The port falls back to 8000, so `python3 serve.py` with nothing set is the
# same http://localhost:8000 the README has always told the reader to open.
#
# Nothing above the socket knows about either. The pages ask for `/api/...`
# relative to whatever host they were loaded from, so the same files serve
# localhost and a public URL with no setting between them.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(ROOT, "web")

# Sessions live for as long as the page holding them does. One process,
# one player, so a plain dictionary behind a lock is enough.
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()

# A session is small, but a page that crashes cannot clean up after itself,
# so the oldest are dropped rather than accumulating forever.
#
# Configurable because the ceiling that matters is memory, and how much there
# is depends on where this is running. A room 5 session carries a 4000-entry
# episode log and two dozen recorded episodes; twelve of those are comfortable
# on a laptop and are most of a small free-tier container. Lower it there
# rather than discovering the limit as a restart mid-training.
SESSIONS_MAX = int(os.environ.get("SESSIONS_MAX") or 12)


def _json_safe(value):
    """Replace anything JSON cannot carry, recursively.

    Infinity and NaN are the ones that turn up in practice: an unmeasured
    convergence delta, or a mean over an empty list.
    """
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


# ----------------------------------------------------------------------
# The API
# ----------------------------------------------------------------------

def _session(identifier):
    with SESSIONS_LOCK:
        if identifier not in SESSIONS:
            raise NotFound("no session %r" % identifier)
        return SESSIONS[identifier]


def _remember(session):
    identifier = uuid.uuid4().hex[:12]
    with SESSIONS_LOCK:
        SESSIONS[identifier] = session
        while len(SESSIONS) > SESSIONS_MAX:
            SESSIONS.pop(next(iter(SESSIONS)))
    return identifier


def api_config():
    return config.as_dict()


def api_rooms():
    return {
        "rooms": [
            {"number": number,
             "name": rooms.room(number)["name"],
             "sector": rooms.room(number)["sector"],
             "built": rooms.is_built(number),
             "algorithmDefault": rooms.room(number)["algorithm_default"]}
            for number in rooms.ROOM_NUMBERS
        ],
    }


def api_create(body):
    number = int(body.get("room", 1))
    if not rooms.is_built(number):
        raise BadRequest("room %d is not built yet" % number)

    try:
        session = Session(number,
                          algorithm_key=body.get("algorithm"),
                          parameters=body.get("parameters"),
                          seed=body.get("seed"))
    except (ValueError, KeyError) as problem:
        raise BadRequest(str(problem))

    identifier = _remember(session)
    return {"session": identifier,
            "describe": session.describe(),
            "snapshot": session.snapshot()}


def api_command(identifier, command, body):
    session = _session(identifier)

    try:
        if command == "play":
            return session.play()
        if command == "pause":
            return session.pause()
        if command == "step":
            return session.step_once()
        if command == "reset":
            return session.reset()
        if command == "replay":
            return session.start_replay()
        if command == "episodes":
            # Not a snapshot: the whole recording, which the room screen is
            # handed once a run has finished and then animates on its own.
            return session.batch()
        if command == "evaluate":
            # The learned policy measured on every layout pool, plus the random
            # baseline. Nothing here updates a weight — see `Session.evaluate`.
            layouts = body.get("layouts")
            return session.run_evaluation(
                None if layouts is None else int(layouts))
        if command == "test-room":
            # Generate an unseen warehouse and fly the frozen policy in it.
            # `seed` is optional; without one a test layout that has not been
            # shown yet is chosen, so "another room" really is another room.
            seed = body.get("seed")
            return session.run_test_room(
                None if seed is None else int(seed))
        if command == "advance":
            budget = body.get("budgetMs")
            if budget is not None:
                budget = min(float(budget), config.TURBO_BUDGET_MS)
                return session.advance(budget_ms=budget)
            return session.advance(steps=int(body.get("steps", 1)))
        if command == "parameters":
            return session.set_parameters(body.get("values") or {})
        if command == "parameter-default":
            return session.reset_parameter(body.get("name"))
        if command == "algorithm":
            return session.set_algorithm(body.get("key"))
    except IllegalTransition as problem:
        raise BadRequest(str(problem))
    except ValueError as problem:
        raise BadRequest(str(problem))

    raise NotFound("no command %r" % command)


def api_compare(body):
    """Start a comparison: one room, several methods, the same seed."""
    number = int(body.get("room", 1))
    if not rooms.is_built(number):
        raise BadRequest("room %d is not built yet" % number)

    try:
        comparison = Comparison(number,
                                algorithm_keys=body.get("algorithms"),
                                parameters=body.get("parameters"),
                                seed=body.get("seed"))
    except (ValueError, KeyError) as problem:
        raise BadRequest(str(problem))

    identifier = _remember(comparison)
    return {"comparison": identifier,
            "describe": comparison.describe(),
            "snapshot": comparison.snapshot()}


def api_compare_advance(identifier, body):
    comparison = _session(identifier)
    budget = body.get("budgetMs")
    if budget is not None:
        budget = min(float(budget), config.TURBO_BUDGET_MS)
    return comparison.advance(budget_ms=budget)


def api_release(identifier):
    """Called on exit, so a session does not outlive the page that made it."""
    with SESSIONS_LOCK:
        SESSIONS.pop(identifier, None)
    return {"released": True}


# ----------------------------------------------------------------------
# The server
# ----------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    server_version = "ProjectR5"
    protocol_version = "HTTP/1.1"

    # -- plumbing ------------------------------------------------------

    def log_message(self, format, *args):
        """Quieter than the default: one line per request, no timestamps."""
        if self.path.startswith("/api/"):
            sys.stderr.write("  %s %s\n" % (self.command, self.path))

    def _send(self, status, body, content_type, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A local tool, and the simulation is not cacheable anyway.
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=200):
        # `allow_nan=False` on purpose. Python is happy to write bare
        # `Infinity` and `NaN` into a document it calls JSON, and its own
        # parser reads them back, so a round trip inside Python looks fine —
        # but no browser will parse either of them, and the whole response
        # is lost rather than one field. Non-finite numbers are converted to
        # null first; this then makes any that were missed fail loudly here
        # instead of silently reaching the page.
        body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error(self, status, message):
        self._send_json({"error": message}, status=status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise BadRequest("the request body is not JSON")
        if not isinstance(parsed, dict):
            raise BadRequest("the request body must be an object")
        return parsed

    # -- static files --------------------------------------------------

    def _resolve(self, path):
        """A path inside `web/`, or None if it tries to escape."""
        clean = posixpath.normpath(path.split("?", 1)[0])
        while clean.startswith("/"):
            clean = clean[1:]
        if not clean or clean.endswith("/"):
            clean = posixpath.join(clean, "index.html")

        full = os.path.normpath(os.path.join(WEB_ROOT, clean))
        # Anything that resolves outside the web root is refused outright.
        if not full.startswith(WEB_ROOT + os.sep) and full != WEB_ROOT:
            return None
        return full

    def _serve_file(self, path):
        full = self._resolve(path)
        if full is None:
            self._send_error(403, "outside the web root")
            return
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            self._send_error(404, "no such file: %s" % path)
            return

        kind, _ = mimetypes.guess_type(full)
        with open(full, "rb") as handle:
            body = handle.read()
        self._send(200, body, kind or "application/octet-stream")

    # -- routing -------------------------------------------------------

    def do_GET(self):
        route = self.path.split("?", 1)[0]

        if route == "/api/config":
            self._send_json(api_config())
            return
        if route == "/api/rooms":
            self._send_json(api_rooms())
            return
        if route.startswith("/api/"):
            self._send_error(404, "no such endpoint")
            return

        if route == "/":
            route = "/index.html"
        self._serve_file(route)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if not route.startswith("/api/"):
            self._send_error(404, "no such endpoint")
            return

        try:
            body = self._read_body()

            if route == "/api/session":
                self._send_json(api_create(body))
                return

            if route == "/api/comparison":
                self._send_json(api_compare(body))
                return

            parts = route.split("/")
            # /api/session/<id>/<command>
            if len(parts) == 5 and parts[2] == "session":
                self._send_json(api_command(parts[3], parts[4], body))
                return
            # /api/comparison/<id>/advance
            if len(parts) == 5 and parts[2] == "comparison" \
                    and parts[4] == "advance":
                self._send_json(api_compare_advance(parts[3], body))
                return

            self._send_error(404, "no such endpoint")

        except BadRequest as problem:
            self._send_error(400, str(problem))
        except NotFound as problem:
            self._send_error(404, str(problem))
        except Exception as problem:                    # noqa: BLE001
            # A simulation bug must not take the server down; the page shows
            # the message and the player can reset.
            self._send_error(500, "%s: %s" % (type(problem).__name__, problem))

    def do_DELETE(self):
        parts = self.path.split("?", 1)[0].split("/")
        if len(parts) == 4 and parts[2] in ("session", "comparison"):
            self._send_json(api_release(parts[3]))
            return
        self._send_error(404, "no such endpoint")


def main():
    if not os.path.isdir(WEB_ROOT):
        sys.stderr.write("no web/ directory beside serve.py\n")
        return 1

    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True

    print("PROJECT R-5")
    print("  serving %s" % WEB_ROOT)
    if HOST == "0.0.0.0":
        # "open http://0.0.0.0:8000" is an instruction that does not work, so
        # print the address that does. On a platform the public address is the
        # one it publishes, and this line is only ever read in its log.
        print("  listening on 0.0.0.0:%d (every interface)" % PORT)
        print("  open http://localhost:%d" % PORT)
    else:
        print("  open http://%s:%d" % (HOST, PORT))
    print("  ctrl-c to stop")
    # Unbuffered enough to reach a platform's log viewer at the moment it
    # happens rather than whenever the pipe fills.
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
