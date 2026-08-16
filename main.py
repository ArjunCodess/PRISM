from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "apps" / "web"
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "ml" / "artifacts"
PUBLIC = WEB / "public"
REQUIRED_RAW = (RAW / "train_data.zip", RAW / "test_data.csv")
PUBLIC_ARTIFACTS = ("demo_cases.json", "metrics.json", "model_card.json")
PYTHON_MODULES = (
    "fastapi",
    "httpx",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "pydantic",
    "pytest",
    "ruff",
    "shap",
    "sklearn",
    "uvicorn",
    "xgboost",
)


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    printable = " ".join(command)
    print(f"\n[PRISM] {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def downloads_are_verified() -> bool:
    provenance_path = ROOT / "data" / "PROVENANCE.md"
    if not provenance_path.exists():
        return False
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    for path in REQUIRED_RAW:
        record = provenance.get(path.name, {})
        if not path.exists() or path.stat().st_size != record.get("bytes"):
            return False
        if sha256(path) != record.get("sha256"):
            return False
    return True


def download(force: bool) -> None:
    if not force and downloads_are_verified():
        print("[PRISM] ESA archives already exist; keeping the checksummed local copies.")
        return
    run([sys.executable, "ml/src/download.py"])


def ensure_python_dependencies() -> None:
    missing = [name for name in PYTHON_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return
    print(f"[PRISM] Installing missing Python dependencies: {', '.join(missing)}")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def train(source: str, synthetic_events: int) -> None:
    command = [sys.executable, "ml/src/pipeline.py", "--source", source]
    if source == "synthetic":
        command.extend(["--synthetic-events", str(synthetic_events)])
    run(command)


def generate_graphs() -> None:
    run([sys.executable, "ml/src/plots.py"])


def sync_web_artifacts() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in PUBLIC_ARTIFACTS:
        source = ARTIFACTS / name
        if not source.exists():
            raise FileNotFoundError(f"training did not produce {source}")
        shutil.copy2(source, PUBLIC / name)
    print("[PRISM] Synced frozen artifacts into the offline web bundle.")


def ensure_web_dependencies() -> None:
    if (WEB / "node_modules").exists():
        return
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required; install Node.js 20 or newer and try again")
    run([npm, "ci"], cwd=WEB)


def verify() -> None:
    run([sys.executable, "-m", "ruff", "check", "ml", "apps/api", "main.py"])
    run([sys.executable, "-m", "pytest", "ml/tests", "apps/api/tests", "-q"])
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required; install Node.js 20 or newer and try again")
    run([npm, "test"], cwd=WEB)
    run([npm, "run", "lint"], cwd=WEB)
    run([npm, "run", "build"], cwd=WEB)


def wait_for(url: str, process: subprocess.Popen[bytes], timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited early with code {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"service did not become ready within {timeout:.0f}s: {url}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
        process.wait(timeout=5)
        return
    else:
        process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def serve(api_port: int, web_port: int, open_browser: bool) -> None:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required; install Node.js 20 or newer and try again")

    environment = os.environ.copy()
    environment["NEXT_PUBLIC_API_URL"] = f"http://127.0.0.1:{api_port}"
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--app-dir",
            "apps/api",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ],
        cwd=ROOT,
        env=environment,
        creationflags=creation_flags,
    )
    web = subprocess.Popen(
        [npm, "start", "--", "--hostname", "127.0.0.1", "--port", str(web_port)],
        cwd=WEB,
        env=environment,
        creationflags=creation_flags,
    )

    try:
        wait_for(f"http://127.0.0.1:{api_port}/health", api)
        wait_for(f"http://127.0.0.1:{web_port}/", web)
        print(f"\n[PRISM] Web app: http://127.0.0.1:{web_port}")
        print(f"[PRISM] API docs: http://127.0.0.1:{api_port}/docs")
        print("[PRISM] Press Ctrl+C to stop both services.\n")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{web_port}/")
        while api.poll() is None and web.poll() is None:
            time.sleep(0.5)
        failed = "API" if api.poll() is not None else "web app"
        code = api.returncode if api.poll() is not None else web.returncode
        raise RuntimeError(f"{failed} exited unexpectedly with code {code}")
    except KeyboardInterrupt:
        print("\n[PRISM] Stopping services…")
    finally:
        stop_process(web)
        stop_process(api)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, train, evaluate, build, and run the complete PRISM exhibit"
    )
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--synthetic-events", type=int, default=420)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument(
        "--build-only", action="store_true", help="prepare everything without starting services"
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-port", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    ensure_python_dependencies()
    ensure_web_dependencies()
    if not args.skip_download and args.source == "real":
        download(args.force_download)
    if not args.skip_train:
        train(args.source, args.synthetic_events)
    sync_web_artifacts()
    if not args.skip_graphs:
        generate_graphs()
    if not args.skip_checks:
        verify()
    if not args.build_only:
        serve(args.api_port, args.web_port, not args.no_browser)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, TimeoutError) as exc:
        print(f"\n[PRISM] Failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
