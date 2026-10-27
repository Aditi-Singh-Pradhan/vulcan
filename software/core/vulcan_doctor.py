#!/usr/bin/env python3
"""
Vulcan diagnostic script.

Run from the software/ folder:
    python3 vulcan_doctor.py

Writes debug_report_<hostname>.md. Every window test runs in its own
subprocess with a timeout, so a hang is reported as HANG instead of
freezing your terminal. Share the report file with your teammate.
"""

import os
import sys
import socket
import platform
import subprocess
from datetime import datetime
from pathlib import Path

TIMEOUT = 40  # seconds per test (Whisper model load can be slow)
SOFTWARE_DIR = Path(__file__).resolve().parent
ENV = {**os.environ, "QT_QPA_PLATFORM": "xcb"}

# Each window test: do some setup, then try to open a window and pump it.
WINDOW = (
    "import cv2, numpy as np\n"
    "cv2.namedWindow('doctor')\n"
    "img = np.zeros((120,320,3), np.uint8)\n"
    "cv2.putText(img,'doctor',(20,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)\n"
    "cv2.imshow('doctor', img)\n"
    "cv2.waitKey(800)\n"
    "cv2.destroyAllWindows()\n"
    "print('WINDOW_OK')\n"
)

RAG_PATH = (
    f"import sys; sys.path.insert(0, {str(SOFTWARE_DIR)!r}); "
    f"sys.path.insert(0, {str(SOFTWARE_DIR / 'llm' / 'rag')!r})\n"
)

WINDOW_TESTS = [
    ("window alone", ""),
    ("after import sounddevice", "import sounddevice\n"),
    ("after import faster_whisper", "import faster_whisper\n"),
    ("after loading WhisperModel",
     "from faster_whisper import WhisperModel\n"
     "WhisperModel('small', device='cpu', compute_type='int8')\n"),
    ("after import ultralytics", "import ultralytics\n"),
    ("after import mediapipe", "import mediapipe\n"),
    ("after import fer", "import fer\n"),
    ("after import pyorbbecsdk", "import pyorbbecsdk\n"),
    ("after import RAG module", RAG_PATH + "from llm.rag.app_lang2 import response_rag\n"),
]

PACKAGES = [
    "cv2", "numpy", "tensorflow", "protobuf", "mediapipe", "ultralytics",
    "torch", "fer", "faster_whisper", "ctranslate2", "sounddevice",
    "langchain", "langchain_ollama", "pyorbbecsdk", "serial",
]


def run(cmd, timeout=10, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env, shell=isinstance(cmd, str))
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except FileNotFoundError:
        return -1, "command not found"


def py(code, timeout=TIMEOUT):
    return run([sys.executable, "-c", code], timeout=timeout, env=ENV)


def pkg_version(name):
    code = (
        "import importlib\n"
        f"m = importlib.import_module({name!r})\n"
        "print(getattr(m, '__version__', 'installed (no __version__)'))\n"
    )
    if name == "protobuf":
        code = "import google.protobuf as p; print(p.__version__)"
    rc, out = py(code, timeout=60)
    if rc == 0:
        return out.splitlines()[-1]
    return "NOT IMPORTABLE: " + (out.splitlines()[-1] if out else "?")


def main():
    host = socket.gethostname()
    lines = [
        f"# Vulcan debug report — {host}",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## System",
        f"- OS: {platform.platform()}",
        f"- Python: {sys.version.split()[0]} ({sys.executable})",
        f"- Machine: {platform.machine()}",
        f"- XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', '(unset)')}",
        f"- DISPLAY: {os.environ.get('DISPLAY', '(unset)')}",
        f"- WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', '(unset)')}",
        "",
        "## Package versions",
        "| Package | Version |",
        "|---|---|",
    ]

    print("Checking package versions...")
    for name in PACKAGES:
        lines.append(f"| {name} | {pkg_version(name)} |")

    print("Checking OpenCV GUI backend...")
    rc, out = py(
        "import cv2, os\n"
        "info = cv2.getBuildInformation().splitlines()\n"
        "print('\\n'.join(l.strip() for l in info if 'GUI' in l or 'QT:' in l or 'GTK' in l))\n"
        "p = os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins', 'platforms')\n"
        "print('bundled platforms:', os.listdir(p) if os.path.isdir(p) else 'none')\n"
    )
    lines += ["", "## OpenCV GUI", "```", out, "```"]

    print("Checking pip for conflicting OpenCV installs...")
    rc, out = run([sys.executable, "-m", "pip", "list"], timeout=60)
    cv_pkgs = [l for l in out.splitlines() if "opencv" in l.lower()]
    lines += ["", "## OpenCV packages installed (should be exactly one)", "```",
              "\n".join(cv_pkgs) or "none", "```"]
    if len(cv_pkgs) > 1:
        lines.append("⚠️ More than one OpenCV package — they overwrite each other.")

    print("Checking hardware...")
    _, usb = run("lsusb | grep -i orbbec")
    _, tty = run("ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null")
    _, mics = py("import sounddevice as sd; print(sd.query_devices())", timeout=20)
    _, ollama = run("ollama list")
    lines += [
        "", "## Hardware / services",
        f"- Orbbec camera: {usb or 'NOT DETECTED'}",
        f"- Serial ports: {tty or 'none found'}",
        f"- Piper voice present: {Path.home().joinpath('piper-voices/en_US-lessac-medium.onnx').exists()}",
        "", "Ollama models:", "```", ollama, "```",
        "", "Audio devices:", "```", mics, "```",
    ]

    lines += [
        "", "## Window tests (QT_QPA_PLATFORM=xcb, each in its own process)",
        "A small window should flash for each test. HANG = this import breaks the window.",
        "", "| Test | Result |", "|---|---|",
    ]
    for label, setup in WINDOW_TESTS:
        print(f"Window test: {label} ...", flush=True)
        rc, out = py(setup + WINDOW)
        if rc is None:
            result = f"❌ HANG (>{TIMEOUT}s)"
        elif "WINDOW_OK" in out:
            result = "✅ OK"
        else:
            last = out.splitlines()[-1] if out else "no output"
            result = f"❌ ERROR: {last[:120]}"
        print(f"   {result}")
        lines.append(f"| {label} | {result} |")

    report = SOFTWARE_DIR / f"debug_report_{host}.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()