"""Loopback UI for the existing LWM batch pipeline. Start from the project directory."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener

from .audio import RATE
from .models import ASR
from .progress import PREFIX

PORT = 8765
STATIC = Path(__file__).with_name("webui")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
STAGES = {
    "nemotron": "Nemotron으로 화자 구분 중",
    "segment": "발화 구간 찾는 중",
    "diarize": "화자 구분 중",
    "asr": "음성을 텍스트로 변환 중",
    "minutes": "회의록 작성 중",
}
OUTPUTS = {"txt": "transcript.txt", "md": "minutes.md"}


def clock_time(samples):
    minutes, seconds = divmod(samples // RATE, 60)
    return f"{minutes}:{seconds:02}"


def stage_progress(mode, stage, unit, done, total, diarizer):
    stages = (
        ["nemotron", "asr"] if diarizer == "nemotron" else ["segment", "diarize", "asr"]
    )
    if mode == "minutes":
        stages.append("minutes")
    index = stages.index(stage)
    fraction = min(1, done / total) if total else 0
    detail = f"{index + 1}/{len(stages)}단계"
    if unit == "samples":
        detail += f" · 녹음 {clock_time(done)} / {clock_time(total)}"
    elif unit == "regions":
        detail += f" · 화자 구간 {done} / {total}개"
        fraction *= 0.9
    elif unit == "cluster":
        detail += " · 같은 화자의 구간을 연결하는 중"
        fraction = 0.9
    elif unit == "utterances":
        detail += f" · 발언 {done} / {total}개 요약"
        fraction *= 0.8
    elif unit == "groups":
        # Multiple consolidation passes may be necessary. Their count is unknown.
        detail += f" · 회의록 통합 중 · 이번 차수 {done} / {total}묶음"
        fraction = 0.8
    elif unit == "cached":
        detail += " · 저장된 결과 사용"
        fraction = 1
    else:
        detail += " · 모델 준비 중"
    return {
        "percent": min(99, int(100 * (index + fraction) / len(stages))),
        "detail": detail,
    }


class Jobs:
    def __init__(self, root):
        self.root = root.resolve()
        self.output = self.root / "runs" / "web"
        self.lock = threading.Lock()
        self.jobs = {}
        self.latest = None

    def reserve(
        self, name, model, mode, speakers=None, threshold=None, diarizer="nemotron"
    ):
        with self.lock:
            if self.latest and self.jobs[self.latest]["state"] == "busy":
                return None
            job = uuid.uuid4().hex
            self.jobs[job] = {
                "id": job,
                "name": name,
                "model": model,
                "diarizer": diarizer,
                "speakers": speakers,
                "threshold": threshold,
                "mode": mode,
                "state": "busy",
                "message": "파일 받는 중",
                "percent": 0,
                "detail": "",
                "files": {},
            }
            self.latest = job
            return job

    def update(self, job, **fields):
        with self.lock:
            self.jobs[job].update(fields)

    def snapshot(self, job):
        with self.lock:
            item = self.jobs.get(job)
            if item is None:
                return None
            return {**item, "files": list(item["files"])}

    def download(self, job, kind):
        with self.lock:
            item = self.jobs.get(job)
            if item and kind in item["files"]:
                return item["files"][kind], Path(item["name"]).stem + "-" + OUTPUTS[
                    kind
                ]

    def process(self, job, temp, audio):
        try:
            request = self.snapshot(job)
            command = [
                sys.executable,
                "-u",
                "-m",
                "lwm",
                "run",
                str(audio),
                "--stt",
                request["model"],
                "--output",
                str(self.output),
            ]
            if request["mode"] == "txt":
                command.append("--transcript-only")
            command.extend(["--diarizer", request["diarizer"]])
            if request.get("speakers"):
                command.extend(["--speakers", str(request["speakers"])])
            if request["threshold"] is not None:
                command.extend(["--threshold", str(request["threshold"])])
            self.update(job, message="오디오 준비 중")
            files = {}
            last = ""
            with subprocess.Popen(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONUTF8": "1", "LWM_WEB_PROGRESS": "1"},
            ) as proc:
                for line in proc.stdout:
                    last = line.strip() or last
                    if last.startswith(PREFIX):
                        stage, unit, done, total = json.loads(last[len(PREFIX) :])
                        self.update(
                            job,
                            message=STAGES[stage],
                            **stage_progress(
                                request["mode"],
                                stage,
                                unit,
                                done,
                                total,
                                request["diarizer"],
                            ),
                        )
                        continue
                    for stage, label in STAGES.items():
                        if last.startswith(f"[{stage}]"):
                            self.update(
                                job,
                                message=label,
                                **stage_progress(
                                    request["mode"],
                                    stage,
                                    "load",
                                    0,
                                    0,
                                    request["diarizer"],
                                ),
                            )
                    if last.startswith("텍스트 전사본: "):
                        txt = Path(last.removeprefix("텍스트 전사본: ")).resolve()
                        if (
                            txt.is_relative_to(self.output)
                            and txt.name == OUTPUTS["txt"]
                            and txt.is_file()
                        ):
                            files["txt"] = txt
                            self.update(job, files=files.copy())
                if proc.wait():
                    raise RuntimeError(last or "처리를 완료하지 못했습니다.")
            if "txt" not in files:
                raise RuntimeError("녹취록 파일을 찾을 수 없습니다.")
            if request["mode"] == "minutes":
                md = files["txt"].with_name(OUTPUTS["md"])
                if not md.is_file():
                    raise RuntimeError("회의록 파일을 찾을 수 없습니다.")
                files["md"] = md
            self.update(
                job,
                state="done",
                message="처리 완료",
                files=files,
                percent=100,
                detail="",
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            self.update(job, state="error", message=str(exc))
        finally:
            temp.cleanup()


class LocalUI(BaseHTTPRequestHandler):
    def send_bytes(self, data, mime, status=200, **headers):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def reply(self, obj, status=200):
        self.send_bytes(
            json.dumps(obj, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def local_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self):
        if not self.local_host():
            return self.send_error(403)
        path = urlsplit(self.path).path
        jobs = self.server.jobs
        if path in ASSETS:
            name, mime = ASSETS[path]
            return self.send_bytes(
                (STATIC / name).read_bytes(), mime + "; charset=utf-8"
            )
        if path == "/api/state":
            return self.reply(
                {"app": "lwm-local", "latest": jobs.snapshot(jobs.latest)}
            )
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})(?:/(txt|md))?", path)
        if match:
            job, kind = match.groups()
            if not kind and (item := jobs.snapshot(job)):
                return self.reply(item)
            if kind and (file := jobs.download(job, kind)):
                target, name = file
                if target.is_file():
                    return self.send_bytes(
                        target.read_bytes(),
                        "text/plain; charset=utf-8",
                        **{
                            "Content-Disposition": "attachment; filename*=UTF-8''"
                            + quote(name, safe="")
                        },
                    )
        self.send_error(404)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if (
            not self.local_host()
            or self.headers.get("X-LWM-Upload") != "1"
            or origin
            and origin != "http://" + self.headers["Host"]
        ):
            return self.send_error(403)
        if self.path != "/api/jobs":
            return self.send_error(404)
        name = (
            unquote(self.headers.get("X-Filename", ""))
            .replace("\\", "/")
            .split("/")[-1]
        )
        model, mode = self.headers.get("X-Model"), self.headers.get("X-Mode")
        if (
            Path(name).suffix.lower() not in (".wav", ".mp3", ".m4a")
            or model not in ASR
            or mode not in ("txt", "minutes")
        ):
            return self.reply(
                {"error": "WAV, MP3, M4A 파일과 처리 옵션을 확인하세요."}, 400
            )
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 1:
                raise ValueError
        except ValueError:
            return self.reply(
                {"error": "비어 있거나 파일 크기를 확인할 수 없습니다."}, 400
            )
        jobs = self.server.jobs
        count = self.headers.get("X-Speakers", "")
        raw_threshold = self.headers.get("X-Speaker-Threshold")
        tuned = bool(count) or raw_threshold is not None
        diarizer = self.headers.get("X-Diarizer", "community" if tuned else "nemotron")
        if (
            diarizer not in ("nemotron", "community")
            or diarizer == "nemotron"
            and tuned
        ):
            return self.reply({"error": "화자 모델과 화자 설정을 확인하세요."}, 400)
        if count and not re.fullmatch(r"[1-9][0-9]{0,2}", count):
            return self.reply(
                {"error": "발언자 수는 1~999 사이의 정수로 입력하세요."}, 400
            )
        try:
            threshold = float(raw_threshold) if raw_threshold is not None else None
            if threshold is not None and not 0 < threshold < 2:
                raise ValueError
        except ValueError:
            return self.reply(
                {"error": "거리 임계값은 0보다 크고 2보다 작아야 합니다."}, 400
            )
        job = jobs.reserve(
            name, model, mode, int(count) if count else None, threshold, diarizer
        )
        if not job:
            return self.reply(
                {"error": "진행 중인 작업을 마친 후 다시 시도하세요."}, 409
            )
        temp = None
        try:
            jobs.output.mkdir(parents=True, exist_ok=True)
            temp = TemporaryDirectory(prefix=".upload-", dir=jobs.output)
            # Keep the original display name separate from filesystem path construction.
            audio = Path(temp.name) / ("audio" + Path(name).suffix.lower())
            self.connection.settimeout(120)
            with audio.open("wb") as out:
                remaining = length
                while remaining:
                    part = self.rfile.read(min(1024 * 1024, remaining))
                    if not part:
                        raise ValueError("파일 전송이 중단되었습니다.")
                    out.write(part)
                    remaining -= len(part)
            threading.Thread(
                target=jobs.process, args=(job, temp, audio), daemon=True
            ).start()
        except (OSError, ValueError) as exc:
            if temp:
                temp.cleanup()
            jobs.update(job, state="error", message=str(exc))
            return self.reply({"error": str(exc)}, 400)
        self.reply(jobs.snapshot(job), HTTPStatus.ACCEPTED)


def main():
    parser = argparse.ArgumentParser(description="로컬 회의 녹취 웹앱")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--open", action="store_true", help="기본 브라우저 열기")
    args = parser.parse_args()
    address = ("127.0.0.1", args.port)
    url = f"http://127.0.0.1:{args.port}"
    if args.open:
        try:
            with build_opener(ProxyHandler({})).open(
                url + "/api/state", timeout=2
            ) as response:
                if json.load(response).get("app") == "lwm-local":
                    webbrowser.open(url)
                    return
        except (OSError, ValueError):
            pass
    try:
        server = ThreadingHTTPServer(address, LocalUI)
    except OSError as exc:
        raise SystemExit(f"{args.port} 포트를 열 수 없습니다.") from exc
    server.jobs = Jobs(Path.cwd())
    with server:
        print(url, flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
