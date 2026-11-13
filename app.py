"""Simple web UI: paste an email, get Phishing / Safe with confidence.

Stdlib only (no Flask needed). Run:  python app.py
Then open http://localhost:8000 in a browser.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import MODEL_PATH, predict_one  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Phishing Email Detector — Thiranex Task 3</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px}
input,textarea{width:100%;padding:8px;margin:6px 0;box-sizing:border-box}
button{padding:10px 18px;font-size:16px;cursor:pointer}
.result{padding:14px;margin-top:16px;border-radius:8px}
.phishing{background:#fde8e8}.safe{background:#e6f4ea}</style></head>
<body><h1>🎣 Phishing Email Detector</h1>
<p>Thiranex Cyber Security — Task 3. Paste an email and the model classifies it.</p>
<form method="post">
<label>Subject</label><input name="subject" required>
<label>Sender</label><input name="sender" placeholder="support@example.com">
<label>Body</label><textarea name="body" rows="6" required></textarea>
<label>Link / URL in the email</label><input name="url" placeholder="https://...">
<button type="submit">Check email</button></form>
{result}</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, html):
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(PAGE.replace("{result}", ""))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        f = parse_qs(body)
        get = lambda k: f.get(k, [""])[0]
        try:
            r = predict_one(get("subject"), get("sender"), get("body"), get("url"))
            cls = "phishing" if r["label"] == "Phishing" else "safe"
            reasons = "".join(f"<li>{x}</li>" for x in r["reasons"])
            result = (f'<div class="result {cls}"><h2>{r["label"]}</h2>'
                      f'<p>Phishing probability: <b>{r["phishing_probability"]:.1%}</b></p>'
                      f'<ul>{reasons}</ul></div>')
        except Exception as e:  # model not trained yet
            result = f'<div class="result phishing"><p>Error: {e}. Run <code>python src/train.py</code> first.</p></div>'
        self._send(PAGE.replace("{result}", result))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("Model not found — run: python src/train.py first")
        raise SystemExit(1)
    print("Open http://localhost:8000")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
