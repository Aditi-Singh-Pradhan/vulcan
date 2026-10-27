# Vulcan: Setup Guide

Follow this top to bottom on a fresh machine. Every command is copy-paste. Don't skip steps, and don't install packages that aren't listed here: most problems so far came from version clashes.

**Tested on:** Ubuntu 22.04 and 24.04, Python 3.10–3.12.
**Time:** ~30–45 min (mostly downloads and one compile).
**You need internet for setup.** After that Vulcan runs fully offline.

---

## 0. What you're installing

| Part | What it does | Needed? |
|---|---|---|
| Python packages | Vision, speech, RAG | Yes |
| PortAudio | Lets Python use the microphone | Yes |
| Ollama + 2 models | The local LLM that answers questions | Yes |
| Piper voice | Vulcan's voice | Yes |
| Orbbec SDK | Depth camera driver | **Only if you have the Gemini camera.** Without it, Vulcan uses your laptop webcam. |

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential cmake \
    libportaudio2 portaudio19-dev libusb-1.0-0-dev alsa-utils
```

`alsa-utils` gives `aplay`, which plays Vulcan's voice. `portaudio` is for the mic.

---

## 2. Get the code

You need access to the repo first.

```bash
cd ~
git clone https://github.com/Aditi-Singh-Pradhan/vulcan.git
cd vulcan
```


---

## 3. Python environment

We use a **virtual environment** (`.venv`) so Vulcan's library versions can't clash with anything else on your system. Do **not** use `--break-system-packages`.

```bash
cd ~/vulcan
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> Every time you open a new terminal to work on Vulcan, run `source ~/vulcan/.venv/bin/activate` first.
> Your prompt will show `(.venv)` when it's active. (`run_vulcan.sh` does this for you automatically.)

### Install the Python packages

These versions are pinned on purpose. Newer TensorFlow/MediaPipe/OpenCV versions break each other.

```bash
pip install "numpy==1.26.4" "tensorflow==2.16.2" "protobuf>=4.25.8,<5" \
    "mediapipe==0.10.14" fer ultralytics \
    faster-whisper sounddevice piper-tts pyserial \
    langchain-core langchain-community langchain-ollama langchain-text-splitters \
    scikit-learn tiktoken pybind11
```

### Fix OpenCV (important, don't skip)

Several packages each install their own copy of OpenCV, and the copies overwrite each other. Remove all of them and install exactly one:

```bash
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install "opencv-contrib-python==4.11.0.86" "numpy==1.26.4"
```

Check:
```bash
pip list 2>/dev/null | grep -i -E "opencv|numpy"
```
You should see exactly **one** opencv line (`opencv-contrib-python 4.11.0.86`) and `numpy 1.26.4`.

---

## 4. Ollama (the LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

Check it's running:
```bash
ollama list
```
You should see both models. If you get "could not connect", start it with `ollama serve` in a separate terminal.

---

## 5. Piper voice

The code expects the voice in `~/piper-voices/`:

```bash
mkdir -p ~/piper-voices
cd ~/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Test it (with the venv active):
```bash
source ~/vulcan/.venv/bin/activate
echo "Hello, I am Vulcan." | piper --model ~/piper-voices/en_US-lessac-medium.onnx --output_file /tmp/test.wav && aplay /tmp/test.wav
```
You should hear the voice.

---

## 6. Orbbec depth camera (only if you have the Gemini)

**No Gemini? Skip this whole step.** Vulcan will automatically use your laptop webcam.

There's no pip package, so it's built from source. Keep the venv active the whole time, so it installs into the right Python.

```bash
source ~/vulcan/.venv/bin/activate
cd ~
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
make install
cd ..
pip install -e .
sudo bash install/lib/pyorbbecsdk/shared/install_udev_rules.sh
```

Plug the camera in (USB 3 port), then test:
```bash
python3 ~/pyorbbecsdk/install/lib/pyorbbecsdk/examples/beginner/01_hello_camera.py
```
It should print `Orbbec Gemini 335L` and its stream settings.

---

## 7. Arduino / ESP32 (only if the robot head is connected)

Find the port with the board plugged in:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```
If it isn't `/dev/ttyUSB0`, change `ARDUINO_PORT` near the top of `software/core/vulcan2.py`.

Give yourself permission to use serial ports (one time, then **log out and back in**):
```bash
sudo usermod -aG dialout $USER
```

Without a board, Vulcan still runs. It just prints `[ARDUINO] Not connected`.

---

## 8. Run Vulcan

```bash
~/vulcan/software/run_vulcan.sh
```

**First run is slow.** It downloads the Whisper speech model (~500 MB) and builds the RAG index. Later runs start in under a minute.

What you should see, in roughly this order:
```
[AUDIO] Loading Whisper...
[AUDIO] Whisper ready.
[RAG] Building vector store ...      (first run only; later: "Loading saved vector store")
[CAMERA] Orbbec color + depth started.   OR   [CAMERA] Laptop webcam (index 0) started.
[VISION] Window created.
[RAG] LLM warmed up.
[AUDIO] Listening...
```

A window opens with the camera feed. Then:
- Say **"Hello Vulcan"** to start. The top-left changes from `IDLE` to `TALKING`.
- Ask anything about ERC, Vulcan or the team.
- Say **"Bye"** to end, or **"Shut down"** to quit.
- Press **`q`** in the video window to quit.

After every answer you'll see a `[TIMING]` line showing how long each step took.

---

## 9. Troubleshooting

Find your error message below. Every one of these has happened to someone on the team.

| Error / symptom | Cause | Fix |
|---|---|---|
| `OSError: PortAudio library not found` | Mic library missing | `sudo apt install libportaudio2 portaudio19-dev` |
| `No module named 'pyorbbecsdk'` | Orbbec SDK not installed | Fine without a Gemini (falls back to webcam). With a Gemini, do step 6 **with the venv active**. |
| `RuntimeError: No device found` | Gemini not plugged in / not detected | Check `lsusb \| grep -i orbbec`. Use a USB 3 port. Re-run the udev rules script from step 6. |
| `No module named 'langchain...'` / `'faster_whisper'` / anything | Package missing, or venv not active | `source ~/vulcan/.venv/bin/activate`, then re-run the install in step 3 |
| `module 'mediapipe' has no attribute 'solutions'` | MediaPipe too new | `pip install "mediapipe==0.10.14"` |
| `cannot import name 'runtime_version' from 'google.protobuf'` | Protobuf/TensorFlow mismatch | `pip install "tensorflow==2.16.2" "protobuf>=4.25.8,<5"` |
| `module 'cv2' has no attribute 'data'` | Wrong OpenCV (e.g. from apt) | `sudo apt remove python3-opencv`, then redo "Fix OpenCV" in step 3 |
| numpy errors / "requires numpy>=2" warnings | Something upgraded numpy | Redo "Fix OpenCV" in step 3 (it re-pins numpy) |
| **No video window appears**, no error | Wayland display | Always start with `run_vulcan.sh` (it sets `QT_QPA_PLATFORM=xcb`). Don't run `python3 -m core.vulcan2` directly. |
| `could not connect to ollama` / RAG errors | Ollama not running | `ollama serve` in another terminal, check `ollama list` shows both models |
| `FileNotFoundError: ... software/data/...txt` | Knowledge file missing | `git pull`. The 3 files `introToERC.txt`, `aboutVulcan.txt`, `team_members.txt` must be in `software/data/` |
| No voice, but answers print | Piper/voice missing | Redo step 5, and check `aplay` works |
| Recording never stops / always 15 s | Room noise above threshold | Lower noise, or raise `SILENCE_THRESHOLD` in `vulcan2.py` (default 0.20) |
| Cuts you off mid-sentence | Threshold too high for your mic | Lower `SILENCE_THRESHOLD` (try 0.15) |
| Hears "Thank you for watching!" from silence | Known Whisper quirk on noise | Harmless. Speak a bit closer to the mic. |
| Wrong webcam opens | Multiple cameras | Change `WEBCAM_INDEX` in `vulcan2.py` (try `1`) |
| `[ARDUINO] Not connected` | No board, or wrong port, or no permission | Step 7 |

Still stuck? Run the diagnostic script and send the report file it creates:
```bash
source ~/vulcan/.venv/bin/activate
cd ~/vulcan/software
python3 core/vulcan_doctor.py
```

---

## 10. Where things live

```
software/
├── run_vulcan.sh            ← start Vulcan with this
├── core/vulcan2.py          ← main program (settings at the top)
├── llm/rag/app_lang2.py     ← RAG: knowledge lookup + LLM
├── data/                    ← knowledge base Vulcan answers from
│   ├── introToERC.txt
│   ├── aboutVulcan.txt
│   └── team_members.txt
├── vision/m_expression.py   ← facial expression model
└── models/                  ← YOLO face model, expression model
Electronics/firmware/servo_control/vulcan_servo2/   ← ESP32 servo firmware
```

**To change what Vulcan knows:** edit the `.txt` files in `software/data/`. The index rebuilds automatically on the next run.

**Useful settings** (top of `software/core/vulcan2.py`):

| Setting | Default | What it does |
|---|---|---|
| `ARDUINO_PORT` | `/dev/ttyUSB0` | ESP32 serial port |
| `WEBCAM_INDEX` | `0` | Laptop camera used when no Gemini |
| `SHOW_LANDMARKS` | `False` | Draw face-mesh dots |
| `SILENCE_THRESHOLD` | `0.20` | How loud counts as speech |
| `SILENCE_LIMIT` | `2.0` | Seconds of quiet before Vulcan answers |
| `WHISPER_HINT` | "Hello Vulcan..." | Words Whisper should expect |