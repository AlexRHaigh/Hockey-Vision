"""Serve the Hockey-Vision web app: the JSON API (api.py) and the page in static/.

    python -m webapp.server                     # http://127.0.0.1:8000
    python -m webapp.server --host 0.0.0.0 --port 8080 --db webapp/hockey_vision.db

Standard library only (http.server + sqlite3), so it runs wherever the pipeline does.
"""

import argparse
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

try:
    from . import api, db
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from webapp import api, db

STATIC = Path(__file__).resolve().parent / "static"
ROUTES = [(method, re.compile(pattern + "$"), fn) for method, pattern, fn in api.ROUTES]
MAX_BODY = 1 << 20


class Handler(BaseHTTPRequestHandler):
    db_path = db.DEFAULT_DB
    server_version = "HockeyVision/1.0"

    def log_message(self, fmt, *args):
        if not getattr(self.server, "quiet", False):
            super().log_message(fmt, *args)

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _api(self, method):
        url = urlsplit(self.path)
        allowed = []
        for route_method, pattern, fn in ROUTES:
            m = pattern.match(url.path)
            if not m:
                continue
            if route_method != method:
                allowed.append(route_method)
                continue
            try:
                body = self._body() if method in ("POST", "PATCH") else {}
                con = db.connect(self.db_path)
                try:
                    result = fn(con, dict(parse_qsl(url.query)), body, *map(int, m.groups()))
                finally:
                    con.close()
                return self._send(HTTPStatus.CREATED if method == "POST" else HTTPStatus.OK, result)
            except api.ApiError as e:
                return self._send(e.status, {"error": str(e)})
        if allowed:
            return self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": f"Use {', '.join(allowed)}"})
        self._send(HTTPStatus.NOT_FOUND, {"error": f"No API route {url.path}"})

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise api.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request too large")
        if not length:
            return {}
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            raise api.ApiError(HTTPStatus.BAD_REQUEST, "Body must be JSON")
        if not isinstance(body, dict):
            raise api.ApiError(HTTPStatus.BAD_REQUEST, "Body must be a JSON object")
        return body

    def _static(self):
        path = urlsplit(self.path).path
        target = (STATIC / path.lstrip("/")).resolve() if path != "/" else STATIC / "index.html"
        if not target.is_relative_to(STATIC) or not target.is_file():
            target = STATIC / "index.html"  # the page routes itself (#/...)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, target.read_bytes(), content_type)

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api("GET")
        self._static()

    do_HEAD = do_GET

    def do_POST(self):
        self._api("POST")

    def do_PATCH(self):
        self._api("PATCH")

    def do_DELETE(self):
        self._api("DELETE")

    def do_PUT(self):
        self._api("PUT")


def make_server(host, port, db_path, quiet=False):
    con = db.connect(db_path)
    db.ensure_players(con)
    con.close()
    handler = type("BoundHandler", (Handler,), {"db_path": db_path})
    server = ThreadingHTTPServer((host, port), handler)
    server.quiet = quiet
    return server


def main():
    parser = argparse.ArgumentParser(description="Serve the Hockey-Vision web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (0.0.0.0 for your network)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB, help="Database file (default: %(default)s)")
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.db)
    print(f"Hockey-Vision web app on http://{args.host}:{args.port} (database {args.db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
