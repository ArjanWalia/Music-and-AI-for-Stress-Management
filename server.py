#!/usr/bin/env python3
"""
Closed-loop biosignal stress monitor — LOCAL app (run on the Mac the Arduino is plugged into).

  Arduino (BPM_RMSSD_EDA.ino)  --USB serial-->  this app  -->  http://127.0.0.1:8000  (clinical dashboard)

Colab CANNOT read the serial port (it's cloud), so run this on your laptop:
    pip install fastapi "uvicorn[standard]" pyserial torch numpy
    python stress_app.py
Then open http://127.0.0.1:8000 in a browser.

Set the three paths + serial port below, and close the Arduino IDE Serial Monitor first (it locks the port).
"""

import os, sys, json, time, math, glob, random, threading, webbrowser
from collections import deque
import numpy as np

# ============================ CONFIG — EDIT THESE ============================
SERIAL_PORT = os.environ.get("STRESS_PORT", "/dev/cu.usbmodem101")  # `ls /dev/cu.*` on macOS to find it
BAUD        = int(os.environ.get("STRESS_BAUD", "115200"))          # must match Serial.begin() in the .ino
MODEL_DIR   = os.environ.get("STRESS_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))        # holds fusionnet.pt, feature_scaler.npz, model_meta.json
MUSIC_DIR   = os.environ.get("STRESS_MUSIC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "suno_ai_songs"))  # optional: subfolders calm/ neutral/ elevated/
PORT = int(os.environ.get("PORT", "8000"))          # cloud hosts (Render, etc.) inject $PORT automatically
# Bind all interfaces when a platform PORT is present (so Render can route to it); loopback locally.
HOST = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")

# --- Spotify (optional; only needed if you use the Spotify source in the dashboard) ---
SPOTIFY_CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI  = os.environ.get("SPOTIFY_REDIRECT_URI", f"http://{HOST}:{PORT}/callback")  # MUST match the dashboard exactly
SPOTIFY_SCOPES        = "streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state"
STRESS_MUSIC_TASTE    = os.environ.get("STRESS_MUSIC_TASTE", "")   # if empty, the saved ~/.stress_music_taste is used

# How the Arduino prints a line. Two formats are auto-detected:
#   labeled : "BPM:72,RMSSD:41.2,EDA:5.34"
#   csv     : "72,41.2,5.34"   (mapped left-to-right using CSV_ORDER below — EDIT to match your sketch)
CSV_ORDER = ["bpm", "hrv", "eda"]          # add "temp","motion" here if your sketch prints them too
ACTIVE_TIMEOUT = 4.0                        # seconds without an update before a channel reads as inactive
EMA_ALPHA      = 0.30                       # smoothing on the stress index (lower = smoother)
CALIB_SEC      = 20                         # seconds of history before the index leaves "calibrating"
SIM_WHEN_NO_SERIAL = True                   # if no Arduino is found, drive the model with a synthetic 5-channel signal
# ===========================================================================

CHANNELS   = ["bpm", "eda", "hrv", "temp", "motion"]

# ---- device profiles: each supported input device declares which channels it can deliver ----
# Channels NOT listed for a device are treated as absent from the start (never "active"), so the
# model scores from only the channels that device provides. `timeout` is the per-device staleness
# window: HealthKit delivers periodic samples (not a 4 Hz stream), so the watch needs a longer one.
DEVICE_PROFILES = {
    "arduino":     {"channels": ["bpm", "eda", "hrv", "temp", "motion"], "timeout": 4.0,  "label": "Arduino kit"},
    "apple_watch": {"channels": ["bpm", "hrv", "motion"],                "timeout": 45.0, "label": "Apple Watch"},
}
DEFAULT_DEVICE = "arduino"

COMMON_HZ, WINDOW_SEC = 4, 60
WIN = COMMON_HZ * WINDOW_SEC               # 240 samples / 60 s window
FEAT_IDX_BY_CH = {"bpm": [0,1,2], "hrv": [3,4,5], "eda": [6,7,8,9], "temp": [10,11], "motion": [12,13]}
KEYMAP = {"BPM":"bpm","HR":"bpm","HEARTRATE":"bpm","BEAT":"bpm",
          "RMSSD":"hrv","HRV":"hrv","SDNN":"hrv",
          "EDA":"eda","GSR":"eda","SCL":"eda",
          "TEMP":"temp","SKINTEMP":"temp","TEMPERATURE":"temp","ST":"temp",
          "MOTION":"motion","ACC":"motion","ACCEL":"motion","MAG":"motion"}

# ---------------------------------------------------------------------------
# Model (architecture must match the saved weights)
# ---------------------------------------------------------------------------
import torch, torch.nn as nn

class RawBranch(nn.Module):
    def __init__(self, n_ch=5, conv=(32,64), k=(7,5), lstm_hidden=64, p=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_ch, conv[0], k[0], padding=k[0]//2), nn.BatchNorm1d(conv[0]), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(conv[0], conv[1], k[1], padding=k[1]//2), nn.BatchNorm1d(conv[1]), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(p))
        self.lstm = nn.LSTM(conv[1], lstm_hidden, batch_first=True, bidirectional=True)
        self.out_dim = 2*lstm_hidden
    def forward(self, x):
        z = self.cnn(x.transpose(1,2)).transpose(1,2)
        _, (h, _) = self.lstm(z)
        return torch.cat([h[-2], h[-1]], dim=1)

class FeatureBranch(nn.Module):
    def __init__(self, n_feat=14, hidden=32, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.out_dim = hidden
    def forward(self, x): return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, n_ch=5, n_feat=14, lstm_hidden=64, mlp_hidden=32, head_hidden=64, p=0.3):
        super().__init__()
        self.raw  = RawBranch(n_ch, lstm_hidden=lstm_hidden, p=p)
        self.feat = FeatureBranch(n_feat, mlp_hidden, p=p)
        self.head = nn.Sequential(nn.Linear(self.raw.out_dim + self.feat.out_dim, head_hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(head_hidden, 1))
    def forward(self, x_cnn, x_mlp):
        z = torch.cat([self.raw(x_cnn), self.feat(x_mlp)], dim=1)
        return self.head(z).squeeze(1)

def load_model():
    """Returns (model, scaler_mean, scaler_scale, meta) or (None, ...) if files are missing."""
    wp = os.path.join(MODEL_DIR, "fusionnet.pt")
    sp = os.path.join(MODEL_DIR, "feature_scaler.npz")
    mp = os.path.join(MODEL_DIR, "model_meta.json")
    if not (os.path.isfile(wp) and os.path.isfile(sp) and os.path.isfile(mp)):
        print(f"[model] missing files in {MODEL_DIR} — running in signal-only mode (no stress index).")
        return None, None, None, {"temperature": 1.0, "decision_threshold": 0.5}
    meta = json.load(open(mp))
    md = meta.get("model", {})
    net = FusionNet(md.get("n_ch",5), md.get("n_feat",14),
                    md.get("lstm_hidden",64), md.get("mlp_hidden",32), md.get("head_hidden",64))
    net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
    z = np.load(sp)
    print(f"[model] loaded {wp}  (T={meta.get('temperature',1.0):.3f})")
    return net, z["mean"].astype(np.float32), z["scale"].astype(np.float32), meta

MODEL, SC_MEAN, SC_SCALE, META = load_model()
TEMP   = float(META.get("temperature", 1.0))
EDA_FEAT_IDX = META.get("eda_feature_indices", FEAT_IDX_BY_CH["eda"])

def compute_features(W, has_eda):
    """Mirror of the training-time feature extractor (same 14-feature order)."""
    bpm, eda, hrv, temp, motion = (W[:, i] for i in range(5))
    slope = lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0
    if has_eda:
        dd = np.diff(eda)
        pk = int(np.sum((dd[:-1] > 0) & (dd[1:] <= 0) & (eda[1:-1] > eda.mean())))
        ef = [eda.mean(), eda.std(), slope(eda), pk]
    else:
        ef = [0.0, 0.0, 0.0, 0.0]
    return np.array([bpm.mean(), bpm.std(), slope(bpm), hrv.mean(), hrv.std(), hrv.min(),
                     *ef, temp.mean(), slope(temp), motion.mean(), motion.std()], dtype=np.float32)

# ---------------------------------------------------------------------------
# Shared state hub (filled by serial + sampler threads, read by the web layer)
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest  = {c: None for c in CHANNELS}
        self.last_ts = {c: 0.0  for c in CHANNELS}
        self.ring    = {c: deque([np.nan]*WIN, maxlen=WIN) for c in CHANNELS}
        self.n   = {c: 0   for c in CHANNELS}      # Welford running stats for per-session z-score
        self.mean= {c: 0.0 for c in CHANNELS}
        self.M2  = {c: 0.0 for c in CHANNELS}
        self.stress = None; self.stress_smooth = None; self.calibrating = True
        self.inf_seq = 0                            # increments on every inference (UI pulse)
        self.last_hrv = None
        self.serial_ok = False; self.last_line = ""
        self.band = "calm"; self.band_since = time.time()
        self.device = DEFAULT_DEVICE                # selected input device (gates which channels are eligible)
        self.last_post_ts = 0.0                     # last time a watch/external POST landed (sim stands down when fresh)
        self.start = time.time()

HUB = Hub()

def parse_line(line):
    line = line.strip()
    if not line:
        return {}
    out = {}
    if ":" in line:                                 # labeled key:value pairs
        for tok in line.replace(";", ",").split(","):
            if ":" in tok:
                k, v = tok.split(":", 1)
                key = KEYMAP.get(k.strip().upper())
                if key:
                    try: out[key] = float(v.strip())
                    except ValueError: pass
    else:                                           # bare CSV of numbers
        parts = [p for p in line.replace(";", ",").split(",") if p.strip() != ""]
        try: nums = [float(p) for p in parts]
        except ValueError: return {}
        for key, val in zip(CSV_ORDER, nums):
            out[key] = val
    return out

def serial_loop():
    try:
        import serial
    except ImportError:
        print("[serial] pyserial not installed — `pip install pyserial`"); return
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
            with HUB.lock: HUB.serial_ok = True
            print(f"[serial] connected {SERIAL_PORT} @ {BAUD}")
            while True:
                raw = ser.readline().decode("utf-8", "ignore")
                if not raw:
                    continue
                d = parse_line(raw)
                if not d:
                    continue
                now = time.time(); new_hrv = False
                with HUB.lock:
                    HUB.last_line = raw.strip()
                    for k, v in d.items():
                        HUB.latest[k] = v; HUB.last_ts[k] = now
                    if "hrv" in d and d["hrv"] != HUB.last_hrv:
                        new_hrv = True; HUB.last_hrv = d["hrv"]
                if new_hrv:                          # update the index every new HRV (~every 2 beats)
                    run_inference()
        except Exception as e:
            with HUB.lock: HUB.serial_ok = False
            print(f"[serial] {e}  — retrying in 2 s (is the Serial Monitor closed?)")
            time.sleep(2)

def sampler_loop():
    """Resample irregular serial onto a fixed 4 Hz grid and keep running per-channel stats."""
    dt = 1.0 / COMMON_HZ
    while True:
        now = time.time()
        with HUB.lock:
            for c in CHANNELS:
                active = (HUB.latest[c] is not None) and ((now - HUB.last_ts[c]) < ACTIVE_TIMEOUT)
                val = float(HUB.latest[c]) if active else np.nan
                HUB.ring[c].append(val)
                if active and not math.isnan(val):   # Welford update
                    HUB.n[c] += 1
                    delta = val - HUB.mean[c]; HUB.mean[c] += delta / HUB.n[c]
                    HUB.M2[c] += delta * (val - HUB.mean[c])
        time.sleep(dt)

def simulate_loop():
    """Synthetic biosignal source — runs ONLY while no real input is arriving. It is device-aware:
       in Apple Watch mode it fabricates only the channels a watch provides (BPM + HRV), so the demo
       reflects the true partial-input path. A slow latent 'arousal' (0..1) drives the channels in a
       physiologically coherent way (HR up, HRV down, EDA up, skin temp down). It beats in real time."""
    rng = np.random.default_rng(); a = 0.25; beat = 0
    while True:
        # stand down if real hardware is connected, sim disabled, or fresh watch POSTs are arriving
        watch_live = (HUB.device == "apple_watch") and ((time.time() - HUB.last_post_ts) < 60.0)
        if HUB.serial_ok or not SIM_WHEN_NO_SERIAL or watch_live:
            time.sleep(0.5); continue
        allowed = set(DEVICE_PROFILES.get(HUB.device, DEVICE_PROFILES[DEFAULT_DEVICE])["channels"])
        a += (0.30 - a) * 0.02 + rng.normal(0, 0.04)                # mean-reverting random walk...
        if rng.random() < 0.004: a += rng.uniform(0.20, 0.50)       # ...with occasional stress episodes
        a = float(np.clip(a, 0.02, 0.98))
        bpm    = 68 + 46*a + rng.normal(0, 1.5)                     # HR rises with arousal
        rmssd  = max(12.0, 78 - 52*a + rng.normal(0, 4.0))          # HRV falls with arousal
        eda    = max(0.3, 3.0 + 9*a + rng.normal(0, 0.25)          # tonic EDA + phasic bursts
                     + (rng.uniform(0, 2.5) if rng.random() < 0.08 else 0.0))
        temp   = 34.8 - 1.6*a + rng.normal(0, 0.05)                 # skin temp drops (vasoconstriction)
        motion = max(0.0, 1.0 + rng.normal(0, 0.03)               # ~1 g rest baseline + occasional movement
                     + (rng.uniform(0, 0.6) if rng.random() < 0.05 else 0.0))
        now = time.time(); new_hrv = (beat % 2 == 0)               # new RMSSD every two beats
        vals = {"bpm": bpm, "eda": eda, "temp": temp, "motion": motion}
        with HUB.lock:
            for c in ("bpm", "eda", "temp", "motion"):
                if c in allowed:                                    # only fabricate channels this device provides
                    HUB.latest[c] = vals[c]; HUB.last_ts[c] = now
            if new_hrv and "hrv" in allowed:
                HUB.latest["hrv"]=rmssd; HUB.last_ts["hrv"]=now; HUB.last_hrv=rmssd
            shown = " ".join(f"{c}={vals.get(c, rmssd if c=='hrv' else 0):.0f}" for c in allowed)
            HUB.last_line = f"[SIM:{HUB.device}] {shown}"
        if new_hrv:
            run_inference()
        beat += 1
        time.sleep(60.0 / max(bpm, 40.0))                          # one beat interval

def _score_from_window(W, stats, active):
    """Shared inference core: given a (WIN,5) window, per-channel (mean,std,n) stats, and an
    `active` map, run the model and return a 0-100 stress score. Inactive channels are zeroed
    in BOTH the raw tensor and the feature vector, so the model scores from only what's present.
    This is the SAME logic the Arduino path and the watch path both use."""
    Xc  = np.zeros((WIN, 5), np.float32)
    raw = np.zeros((WIN, 5), np.float32)
    for i, c in enumerate(CHANNELS):
        mu, sd, n = stats[c]; col = W[:, i]
        filled = np.where(np.isnan(col), (mu if (active[c] and n > 0) else 0.0), col)
        raw[:, i] = filled
        if active[c] and n > 1 and sd > 1e-6:
            Xc[:, i] = (filled - mu) / sd            # per-session z-score
        # inactive channel -> left as zeros (graceful degrade)

    has_eda = active["eda"]
    feat = compute_features(raw, has_eda)
    feat = (feat - SC_MEAN) / SC_SCALE
    for c in CHANNELS:                               # zero features of any inactive channel
        if not active[c]:
            for j in FEAT_IDX_BY_CH[c]:
                feat[j] = 0.0
    for j in EDA_FEAT_IDX:                            # belt-and-suspenders for EDA
        if not has_eda: feat[j] = 0.0

    with torch.no_grad():
        xc = torch.tensor(Xc[None], dtype=torch.float32)
        xm = torch.tensor(feat[None], dtype=torch.float32)
        prob = float(torch.sigmoid(MODEL(xc, xm) / TEMP))
    return prob * 100.0

def _active_map(now):
    """Which channels currently count as active, gated by the selected device profile:
    a channel the chosen device can't provide is never active (so it's scored as absent)."""
    prof = DEVICE_PROFILES.get(HUB.device, DEVICE_PROFILES[DEFAULT_DEVICE])
    allowed = set(prof["channels"]); to = float(prof["timeout"])
    return {c: (c in allowed) and (HUB.latest[c] is not None) and ((now - HUB.last_ts[c]) < to)
            for c in CHANNELS}

def run_inference():
    if MODEL is None:
        return
    now = time.time()
    with HUB.lock:
        active = _active_map(now)
        W = np.stack([np.array(HUB.ring[c], dtype=np.float32) for c in CHANNELS], axis=1)  # (WIN,5)
        stats = {c: (HUB.mean[c], math.sqrt(HUB.M2[c]/HUB.n[c]) if HUB.n[c] > 1 else 0.0, HUB.n[c]) for c in CHANNELS}
    # calibration baseline forms from only the channels this device actually streams
    allowed = set(DEVICE_PROFILES.get(HUB.device, DEVICE_PROFILES[DEFAULT_DEVICE])["channels"])
    total = max((stats[c][2] for c in CHANNELS if c in allowed), default=0)
    calibrating = total < COMMON_HZ * CALIB_SEC

    val = _score_from_window(W, stats, active)

    with HUB.lock:
        HUB.stress = val
        HUB.stress_smooth = val if HUB.stress_smooth is None else (EMA_ALPHA*val + (1-EMA_ALPHA)*HUB.stress_smooth)
        HUB.calibrating = calibrating
        HUB.inf_seq += 1
        _update_band_locked(HUB.stress_smooth)

# music band selection with hysteresis + minimum dwell (prevents song-flapping)
BANDS = [("calm", 0, 38), ("neutral", 38, 68), ("elevated", 68, 101)]
BAND_MARGIN, BAND_MIN_DWELL = 5.0, 12.0
def _band_of(v):
    for name, lo, hi in BANDS:
        if lo <= v < hi: return name
    return "elevated"
def _update_band_locked(v):
    cur = HUB.band; target = _band_of(v)
    if target == cur: return
    lo = next(b[1] for b in BANDS if b[0] == cur); hi = next(b[2] for b in BANDS if b[0] == cur)
    crossed = (v >= hi + BAND_MARGIN) or (v < lo - BAND_MARGIN)
    if crossed and (time.time() - HUB.band_since) > BAND_MIN_DWELL:
        HUB.band = target; HUB.band_since = time.time()

def snapshot():
    now = time.time()
    with HUB.lock:
        prof = DEVICE_PROFILES.get(HUB.device, DEVICE_PROFILES[DEFAULT_DEVICE])
        allowed = set(prof["channels"]); to = float(prof["timeout"])
        active   = {c: (c in allowed) and (HUB.latest[c] is not None) and ((now - HUB.last_ts[c]) < to) for c in CHANNELS}
        readings = {c: (round(float(HUB.latest[c]), 2) if HUB.latest[c] is not None else None) for c in CHANNELS}
        return {
            "readings": readings, "active": active,
            "stress": None if HUB.stress_smooth is None else round(HUB.stress_smooth, 1),
            "stress_raw": None if HUB.stress is None else round(HUB.stress, 1),
            "calibrating": HUB.calibrating, "inf_seq": HUB.inf_seq,
            "band": HUB.band, "serial_ok": HUB.serial_ok,
            "source": ("live" if HUB.serial_ok else ("sim" if SIM_WHEN_NO_SERIAL else "none")),
            "model_loaded": MODEL is not None,
            "device": HUB.device, "device_channels": prof["channels"],   # which channels THIS device provides (dashboard hides the rest)
            "n_active": sum(active.values()), "n_total": len(allowed),    # totals are per-device (e.g. 2 for the watch)
            "session_sec": int(now - HUB.start), "last_line": HUB.last_line,
        }

# ---------------------------------------------------------------------------
# Web layer
# ---------------------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn, asyncio

app = FastAPI(title="Biosignal Stress Monitor")
if os.path.isdir(MUSIC_DIR):
    app.mount("/music", StaticFiles(directory=MUSIC_DIR), name="music")

# served-track selection with a per-band "played bin": a song won't repeat until the whole
# band has been played, then the bin resets and the cycle starts again.
PLAYED = {}                         # band -> set(filenames already served this cycle)
PLAYED_LOCK = threading.Lock()
def list_tracks(band):
    folder = os.path.join(MUSIC_DIR, band); out = []
    if os.path.isdir(folder):
        for ext in ("*.mp3", "*.wav", "*.ogg", "*.m4a", "*.flac"):
            out += [os.path.basename(p) for p in glob.glob(os.path.join(folder, ext))]
    return sorted(out)

@app.get("/api/track")
def api_track(band: str = "calm"):
    files = list_tracks(band)
    if not files:
        return JSONResponse({"url": None, "name": None, "band": band, "remaining": 0})
    with PLAYED_LOCK:
        played = PLAYED.setdefault(band, set()) & set(files)     # drop files no longer on disk
        pool = [f for f in files if f not in played]
        if not pool:                                             # whole bin played -> fresh cycle
            played = set(); pool = files[:]
        name = random.choice(pool)                               # skip + auto-advance both draw from the UNUSED bin
        played.add(name); PLAYED[band] = played
        remaining = len(files) - len(played)
    return JSONResponse({"url": f"/music/{band}/{name}", "name": name, "band": band, "remaining": remaining})

# ---------------------------------------------------------------------------
# Input-device selection + external-device (Apple Watch) score ingest.
#   The dashboard asks which device is in use; that gates which channels are eligible.
#   A phone app reads HealthKit (BPM, HRV) and POSTs here; the same model scores it and
#   the result flows into the SAME band/music loop as the Arduino path.
# ---------------------------------------------------------------------------
@app.get("/api/device")
def get_device():
    prof = DEVICE_PROFILES.get(HUB.device, DEVICE_PROFILES[DEFAULT_DEVICE])
    return JSONResponse({"device": HUB.device, "channels": prof["channels"], "label": prof["label"],
                         "devices": {k: {"label": v["label"], "channels": v["channels"]} for k, v in DEVICE_PROFILES.items()}})

@app.post("/api/device")
async def set_device(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    dev = (body.get("device") or "").strip().lower()
    if dev not in DEVICE_PROFILES:
        return JSONResponse({"ok": False, "error": f"unknown device; choose one of {list(DEVICE_PROFILES)}"}, status_code=400)
    with HUB.lock:
        if dev != HUB.device:                          # switching device -> reset running stats so calibration is clean
            HUB.device = dev
            for c in CHANNELS:
                HUB.latest[c] = None; HUB.last_ts[c] = 0.0
                HUB.n[c] = 0; HUB.mean[c] = 0.0; HUB.M2[c] = 0.0
                HUB.ring[c] = deque([np.nan]*WIN, maxlen=WIN)
            HUB.stress = None; HUB.stress_smooth = None; HUB.calibrating = True
            HUB.start = time.time()
    prof = DEVICE_PROFILES[dev]
    return JSONResponse({"ok": True, "device": dev, "channels": prof["channels"], "label": prof["label"]})

def _preprocess_watch_payload(body):
    """Convert raw Apple Watch data into the model's channels (server-side preprocessing).

    Accepts whatever the iOS app sends, in any combination:
      • rr_intervals / ibi / ibi_ms : beat-to-beat intervals from HKHeartbeatSeriesSample
            -> BPM  = 60000 / mean(interval_ms)
            -> HRV  = RMSSD = sqrt(mean(successive-difference^2))  (ms)
      • accel / motion_samples      : [{x,y,z}, ...] from CMMotionManager
            -> motion = mean(sqrt(x^2 + y^2 + z^2))   (per-session z-scoring handles scale)
      • bpm / hrv / motion / ...     : already-computed values (used directly / to fill gaps)

    Returns a dict of channel -> float for whatever could be derived."""
    out = {}
    # --- heart: derive BPM + RMSSD from inter-beat intervals (HKHeartbeatSeriesSample) ---
    rr = body.get("rr_intervals") or body.get("ibi_ms") or body.get("ibi")
    if isinstance(rr, list):
        vals = [float(x) for x in rr if isinstance(x, (int, float)) and x > 0]
        rr_ms = [(x * 1000.0 if x < 10 else x) for x in vals]   # HKHeartbeatSeries is in seconds; ms if already large
        if len(rr_ms) >= 2:
            out["bpm"] = 60000.0 / (sum(rr_ms) / len(rr_ms))
            diffs = [rr_ms[i + 1] - rr_ms[i] for i in range(len(rr_ms) - 1)]
            out["hrv"] = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
    # --- motion: magnitude of the acceleration vectors (CMMotionManager) ---
    acc = body.get("accel") or body.get("motion_samples")
    if isinstance(acc, list) and acc:
        mags = []
        for s in acc:
            if isinstance(s, dict):
                x, y, z = float(s.get("x", 0) or 0), float(s.get("y", 0) or 0), float(s.get("z", 0) or 0)
                mags.append((x * x + y * y + z * z) ** 0.5)
            elif isinstance(s, (int, float)):
                mags.append(abs(float(s)))
        if mags:
            out["motion"] = sum(mags) / len(mags)
    # --- direct, already-computed values win / fill any gaps (covers a simpler app that does its own math) ---
    for c in CHANNELS:
        v = body.get(c)
        if isinstance(v, (int, float)):
            out[c] = float(v)
    return out

@app.post("/api/score")
async def api_score(request: Request):
    """External device (e.g. iPhone reading Apple Watch HealthKit + CoreMotion) POSTs raw or computed
    signals. They're preprocessed into BPM/HRV/motion, then only channels valid for the SELECTED device
    are kept; the rest stay absent and the model scores from what's present — same logic as the Arduino path."""
    """External device (e.g. iPhone reading Apple Watch HealthKit) POSTs the channels it has.
    Only channels valid for the SELECTED device are accepted; the rest stay absent and the
    model scores from what's present — identical logic to the Arduino path."""
    if MODEL is None:
        return JSONResponse({"error": "model not loaded on server"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    allowed = set(DEVICE_PROFILES.get(HUB.device, DEVICE_PROFILES[DEFAULT_DEVICE])["channels"])
    derived = _preprocess_watch_payload(body)        # raw RR intervals / accel -> bpm, hrv, motion (+ any direct values)
    incoming = {c: derived[c] for c in CHANNELS
                if c in allowed and c in derived and isinstance(derived[c], (int, float))}
    if not incoming:
        return JSONResponse({"error": f"no usable channels for device '{HUB.device}'; expected any of {sorted(allowed)}"}, status_code=400)
    now = time.time()
    new_hrv = False
    with HUB.lock:
        HUB.last_post_ts = now
        for c, v in incoming.items():
            HUB.latest[c] = v; HUB.last_ts[c] = now
            HUB.n[c] += 1                              # Welford update so per-session z-score works for POSTed data too
            delta = v - HUB.mean[c]; HUB.mean[c] += delta / HUB.n[c]; HUB.M2[c] += delta * (v - HUB.mean[c])
            HUB.ring[c].append(v)
        if "hrv" in incoming and incoming["hrv"] != HUB.last_hrv:
            new_hrv = True; HUB.last_hrv = incoming["hrv"]
        HUB.last_line = f"[POST:{HUB.device}] " + " ".join(f"{c}={incoming[c]:.1f}" for c in incoming)
    # HealthKit is periodic; score on each new HRV, or whenever HRV isn't among the channels
    if new_hrv or "hrv" not in allowed:
        run_inference()
    with HUB.lock:
        out = {"stress": None if HUB.stress_smooth is None else round(HUB.stress_smooth, 1),
               "band": HUB.band, "calibrating": HUB.calibrating,
               "channels_used": sorted(incoming), "device": HUB.device}
    return JSONResponse(out)

# ---------------------------------------------------------------------------
# Per-user sessions (cookie 'sid'). Each browser gets its OWN Anthropic key,
# music taste, Spotify token, and played-bin — so the app is multi-user.
# The developer's Spotify client id/secret stay server-side; users only OAuth.
# ---------------------------------------------------------------------------
import secrets as _secrets, urllib.parse as _urlparse, base64 as _b64
SESSIONS = {}                       # sid -> per-user state (in memory only; nothing persisted to disk)
SESS_LOCK = threading.Lock()
_sp_lock = threading.Lock()

# ---- owner gate: Spotify is restricted to whoever controls THIS machine's environment ----
# The key comes from the env (or is generated and printed to the server terminal). Either way it is
# visible ONLY to the OS account that runs the server — i.e. exactly "the user who can see the env vars".
OWNER_KEY = os.environ.get("STRESS_OWNER_KEY", "") or _secrets.token_urlsafe(9)

def _is_owner(request, s):
    """Owner = this session unlocked with the key, or the request comes from this very machine."""
    if s.get("owner"):
        return True
    host = request.client.host if request.client else ""
    if host in ("127.0.0.1", "::1"):                 # loopback = physically the host machine
        s["owner"] = True
        return True
    return False

def _owner_denied(sid):
    return _with_cookie(JSONResponse({"ok": False, "error": "Spotify is restricted to the app owner on this server."},
                                     status_code=403), sid)

def _new_session():
    return {"key": "", "model": None, "taste": "", "pref_type": "taste", "pref_value": "", "pref_artist": "",
            "owner": False,
            "access": None, "refresh": None, "exp": 0.0, "state": None, "store": {}}

def _session(request):
    sid = request.cookies.get("sid")
    with SESS_LOCK:
        s = SESSIONS.get(sid)
        if not sid or s is None:
            sid = _secrets.token_urlsafe(24)
            s = _new_session(); SESSIONS[sid] = s
        return sid, s

def _with_cookie(resp, sid):
    resp.set_cookie("sid", sid, httponly=True, samesite="lax", max_age=7 * 24 * 3600, path="/")
    return resp

def _sp_basic():
    return _b64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()

def _sp_valid_token(s):
    """Return a non-expired user access token for this session, refreshing if needed."""
    with _sp_lock:
        if not s.get("access"):
            return None
        if time.time() < s["exp"] - 30:
            return s["access"]
        if not s.get("refresh"):
            return s["access"]
        try:
            import requests
            r = requests.post("https://accounts.spotify.com/api/token",
                data={"grant_type": "refresh_token", "refresh_token": s["refresh"]},
                headers={"Authorization": f"Basic {_sp_basic()}"}, timeout=15)
            if r.ok:
                j = r.json()
                s["access"] = j["access_token"]; s["exp"] = time.time() + j.get("expires_in", 3600)
                if j.get("refresh_token"): s["refresh"] = j["refresh_token"]
        except Exception as e:
            print("[spotify] refresh failed:", e)
        return s["access"]

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    sid, _ = _session(request)
    resp = HTMLResponse(HTML, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
    return _with_cookie(resp, sid)

@app.post("/api/owner")
async def api_owner(request: Request):
    """Unlock the Spotify feature by proving you can see this server's environment (the owner key)."""
    sid, s = _session(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if _secrets.compare_digest((body.get("key") or "").strip(), OWNER_KEY):
        s["owner"] = True
        return _with_cookie(JSONResponse({"ok": True, "owner": True}), sid)
    return _with_cookie(JSONResponse({"ok": False, "owner": False, "error": "wrong owner key"}, status_code=403), sid)

@app.post("/api/config")
async def api_config(request: Request):
    """User submits their own Anthropic key. Stored in their session only. Owner-gated."""
    sid, s = _session(request)
    if not _is_owner(request, s):
        return _owner_denied(sid)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if "anthropic_key" in body: s["key"]   = (body.get("anthropic_key") or "").strip()
    if body.get("model"):       s["model"] = (body.get("model") or "").strip() or None
    return _with_cookie(JSONResponse({"ok": True, "key_set": bool(s["key"])}), sid)

@app.post("/api/pref")
async def api_pref(request: Request):
    """Set/replace the music preference at any time. type = taste | artist | song.
    Clears this session's queue so the change takes effect on the very next request.
    For a 'song' preference we also resolve and remember its ARTIST, so that when the
    stress state changes we can play a different song BY THAT ARTIST that fits the new state."""
    sid, s = _session(request)
    if not _is_owner(request, s):
        return _owner_denied(sid)
    try:
        body = await request.json()
    except Exception:
        body = {}
    t = (body.get("type") or "taste").strip().lower()
    if t not in ("taste", "artist", "song"):
        t = "taste"
    s["pref_type"]   = t
    s["pref_value"]  = (body.get("value") or "").strip()
    s["pref_artist"] = ""
    if t == "song" and s["pref_value"]:
        try:
            import fetch_spotify
            tr = fetch_spotify.spotify_search_query(s["pref_value"])
            if tr:
                s["pref_artist"] = (tr.get("artist", "") or "").split(",")[0].strip()   # the song's primary artist
        except Exception as e:
            print("[spotify] could not resolve preferred song's artist:", e)
    s["store"]["queue"] = {}          # drop any pre-fetched songs so the new preference applies immediately
    return _with_cookie(JSONResponse({"ok": True, "pref_type": s["pref_type"],
                                      "pref_value": s["pref_value"], "pref_artist": s["pref_artist"]}), sid)

@app.get("/login")
def sp_login(request: Request):
    sid, s = _session(request)
    if not _is_owner(request, s):
        return HTMLResponse("<p>Spotify is restricted to the app owner on this server. You can close this window.</p>", status_code=403)
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        return HTMLResponse("<p>Server is missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET. <a href='/'>back</a></p>")
    st = _secrets.token_urlsafe(16); s["state"] = st
    q = _urlparse.urlencode({"client_id": SPOTIFY_CLIENT_ID, "response_type": "code",
                             "redirect_uri": SPOTIFY_REDIRECT_URI, "scope": SPOTIFY_SCOPES, "state": st})
    return _with_cookie(RedirectResponse("https://accounts.spotify.com/authorize?" + q), sid)

@app.get("/callback", response_class=HTMLResponse)
def sp_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    sid, s = _session(request)
    if error:
        return HTMLResponse(f"<p>Spotify auth failed: {error}. <a href='/login'>retry</a></p>")
    if not code or state != s.get("state"):
        return HTMLResponse("<p>State mismatch or missing code. <a href='/login'>retry</a></p>")
    try:
        import requests
        r = requests.post("https://accounts.spotify.com/api/token",
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI},
            headers={"Authorization": f"Basic {_sp_basic()}"}, timeout=15)
        if not r.ok:
            return HTMLResponse(f"<p>Token exchange failed ({r.status_code}): {r.text}. <a href='/login'>retry</a></p>")
        j = r.json()
        with _sp_lock:
            s["access"] = j["access_token"]; s["refresh"] = j.get("refresh_token")
            s["exp"] = time.time() + j.get("expires_in", 3600)
    except Exception as e:
        return HTMLResponse(f"<p>Token exchange error: {e}. You can close this window and retry.</p>")
    # finish inside the popup: tell the dashboard we're connected, then close. (The dashboard never reloads.)
    return HTMLResponse("""<!doctype html><html><body style="background:#0a0e13;color:#cdd9e5;font-family:monospace">
<p style="padding:20px">Spotify connected. You can close this window.</p>
<script>
  try{ if(window.opener){ window.opener.postMessage("spotify-connected", "*"); } }catch(e){}
  setTimeout(function(){ window.close(); }, 400);
  if(!window.opener){ location.href = "/"; }   // direct (non-popup) visit: fall back to the dashboard
</script></body></html>""")

@app.get("/api/spotify/status")
def sp_status(request: Request):
    sid, s = _session(request)
    return _with_cookie(JSONResponse({
        "connected": bool(s["access"]),
        "configured": bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET),
        "key_set": bool(s["key"]),
        "owner": _is_owner(request, s),
    }), sid)

@app.get("/api/spotify/token")
def sp_token(request: Request):
    sid, s = _session(request)
    if not _is_owner(request, s):
        return _owner_denied(sid)
    return _with_cookie(JSONResponse({"access_token": _sp_valid_token(s)}), sid)

@app.get("/api/spotify/track")
def sp_track(request: Request, band: str = "calm", exact: int = 0):
    sid, s = _session(request)
    if not _is_owner(request, s):
        return _owner_denied(sid)
    if not s["key"]:
        return _with_cookie(JSONResponse({"uri": None, "name": None, "band": band, "error": "no Anthropic key set"}), sid)
    try:
        import fetch_spotify
        pt, pv = s.get("pref_type", "taste"), s.get("pref_value", "")
        pa = s.get("pref_artist", "")
        if pt == "song" and pv:
            if exact:                                  # explicit set/replay -> the EXACT song
                tr = fetch_spotify.fetch_for_state(band, api_key=s["key"], model=s["model"], store=s["store"], song=pv)
            elif pa:                                   # stress state changed/skip -> a song BY THAT ARTIST fitting the state
                tr = fetch_spotify.fetch_for_state(band, api_key=s["key"], model=s["model"], store=s["store"], artist=pa)
            else:                                      # artist unknown -> fall back to the exact song
                tr = fetch_spotify.fetch_for_state(band, api_key=s["key"], model=s["model"], store=s["store"], song=pv)
        elif pt == "artist" and pv:
            tr = fetch_spotify.fetch_for_state(band, api_key=s["key"], model=s["model"], store=s["store"], artist=pv)
        else:
            taste = pv or "calm, melodic, lo-fi and acoustic"
            tr = fetch_spotify.fetch_for_state(band, taste, api_key=s["key"], model=s["model"], store=s["store"])
        if not tr:
            return _with_cookie(JSONResponse({"uri": None, "name": None, "band": band}), sid)
        return _with_cookie(JSONResponse({"uri": tr["uri"], "name": tr["name"], "artist": tr.get("artist"),
                                          "image": tr.get("image"), "band": band}), sid)
    except Exception as e:
        return _with_cookie(JSONResponse({"uri": None, "name": None, "band": band, "error": str(e)}), sid)

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            await sock.send_text(json.dumps(snapshot()))
            await asyncio.sleep(0.1)                 # 10 Hz to the browser
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Clinical dashboard (single page). Plain string — JS/CSS braces are safe.
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Biosignal Stress Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script>
// Spotify SDK calls this when it finishes loading; maybeInitSpotify is defined in the main script below.
window.onSpotifyWebPlaybackSDKReady = function(){ window._sdkLoaded = true; if (typeof maybeInitSpotify === "function") maybeInitSpotify(); };
</script>
<script src="https://sdk.scdn.co/spotify-player.js"></script>
<style>
  :root{
    --bg:#0a0e13; --panel:#0f161e; --panel2:#0c121a; --line:#1b2531; --grid:#121b25;
    --ink:#cdd9e5; --muted:#5f7183; --faint:#3a4a59;
    --bpm:#3ddc97; --hrv:#46c9e6; --eda:#f2c14e; --temp:#ff8c5a; --motion:#b79ced;
    --calm:#37d293; --neutral:#f2c14e; --elevated:#ff5d5d; --alarm:#ff4d4f;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;
       font-size:14px;-webkit-font-smoothing:antialiased;min-height:100%}
  .mono{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
  /* ---- top status rail ---- */
  header{position:sticky;top:0;z-index:9;display:flex;align-items:center;gap:18px;padding:10px 18px;border-bottom:1px solid var(--line);
         background:linear-gradient(180deg,#0d141c,#0a0e13)}
  .brand{display:flex;align-items:baseline;gap:10px}
  .brand b{font-weight:600;letter-spacing:.14em;font-size:13px}
  .brand span{color:var(--muted);font-size:11px;letter-spacing:.18em}
  .pill{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.06em;color:var(--muted);
        border:1px solid var(--line);border-radius:3px;padding:3px 8px;display:flex;align-items:center;gap:7px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--faint)}
  .dot.ok{background:var(--calm);box-shadow:0 0 8px var(--calm)}
  .dot.bad{background:var(--alarm);box-shadow:0 0 8px var(--alarm)}
  .spacer{flex:1}
  /* ---- layout (page scrolls; gauge/music stay pinned) ---- */
  main{display:grid;grid-template-columns:1fr 360px;gap:16px;padding:16px;align-items:start}
  .traces{display:flex;flex-direction:column;gap:14px}
  .side{position:sticky;top:78px;display:flex;flex-direction:column;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:7px;position:relative}
  /* ---- vital card: header (name + reading) / body (y-axis + plot) / x-axis ---- */
  .vital{display:flex;flex-direction:column;padding:18px 22px 14px;min-height:215px}
  .vhead{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:16px}
  .vtitle{display:flex;align-items:center;gap:11px;min-width:0}
  .vdot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
  .vname{font-size:15px;font-weight:600;letter-spacing:.03em;white-space:nowrap}
  .vlead{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.16em;color:var(--faint);
         border:1px solid var(--line);border-radius:3px;padding:2px 7px}
  .vread{display:flex;align-items:baseline;gap:8px;flex:0 0 auto}
  .val{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:34px;line-height:1}
  .unit{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);letter-spacing:.04em}
  .vbody{flex:1;display:flex;gap:16px;min-height:112px}
  .yaxis{display:flex;flex-direction:column;justify-content:space-between;width:46px;text-align:right;
         font-family:"IBM Plex Mono",monospace;font-size:9.5px;color:var(--muted);padding:2px 0}
  .plot{position:relative;flex:1;min-width:0}
  .plot canvas{display:block;width:100%;height:100%}
  .nosig{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
         font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;color:var(--faint)}
  .xaxis{display:flex;justify-content:space-between;padding-left:62px;margin-top:9px;
         font-family:"IBM Plex Mono",monospace;font-size:9.5px;color:var(--muted)}
  .val.tick{animation:tickflash .5s ease}
  @keyframes tickflash{0%{filter:brightness(1.9)}100%{filter:brightness(1)}}
  .vital.off .val,.vital.off .vname{color:var(--faint)}
  /* ---- stress gauge ---- */
  .gauge{display:flex;flex-direction:column;padding:16px 16px 14px;min-height:330px}
  .gauge h2{margin:0;font-size:11px;letter-spacing:.18em;color:var(--muted);font-weight:600}
  .gauge .sub{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--faint);letter-spacing:.1em;margin-top:3px}
  .gwrap{flex:1;display:grid;grid-template-columns:1fr 64px;gap:14px;align-items:stretch;margin-top:12px;min-height:0}
  .greadout{display:flex;flex-direction:column;justify-content:center;gap:6px}
  .gnum{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:72px;line-height:.9}
  .gband{font-size:13px;letter-spacing:.2em;font-weight:600}
  .gnote{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted)}
  .gcol{position:relative;border:1px solid var(--line);border-radius:4px;background:var(--panel2);overflow:hidden}
  .gfill{position:absolute;left:0;right:0;bottom:0;transition:height .25s ease, background .25s ease}
  .gtick{position:absolute;left:0;right:0;height:1px;background:var(--line)}
  .gtick span{position:absolute;right:4px;top:-7px;font-family:"IBM Plex Mono",monospace;font-size:9px;color:var(--faint)}
  .pulse{position:absolute;top:12px;right:14px;width:8px;height:8px;border-radius:50%;background:var(--faint);transition:all .12s}
  /* ---- music ---- */
  .music{padding:13px 14px}
  .music h2{margin:0 0 10px;font-size:11px;letter-spacing:.18em;color:var(--muted);font-weight:600}
  .trow{display:flex;align-items:center;gap:12px}
  .play{width:42px;height:42px;border-radius:50%;border:1px solid var(--line);background:var(--panel2);
        color:var(--ink);cursor:pointer;font-size:15px;display:flex;align-items:center;justify-content:center;flex:0 0 auto}
  .play:hover{border-color:var(--muted)}
  .tinfo{flex:1;min-width:0}
  .tstate{font-size:12px;font-weight:500}
  .ttrack{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .vol{display:flex;align-items:center;gap:9px;margin-top:12px}
  .vol label{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.12em;color:var(--muted)}
  input[type=range]{-webkit-appearance:none;appearance:none;flex:1;height:3px;border-radius:2px;background:var(--line);outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--ink);cursor:pointer}
  .badge{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;padding:2px 7px;border-radius:3px;border:1px solid var(--line);color:var(--muted)}
  /* button hover + interactive states (override inline styles with !important) */
  #srcLocal,#srcSpotify,#spSave,#prefSet,#spConnect,#prefType,.play{transition:border-color .15s ease,color .15s ease,box-shadow .15s ease,background .15s ease,filter .15s ease}
  #srcLocal:hover,#srcSpotify:hover{border-color:var(--ink) !important;color:var(--ink) !important}
  #spSave:hover,#prefSet:hover{border-color:var(--ink) !important;color:var(--ink) !important;background:rgba(255,255,255,.05) !important}
  #spConnect:hover{filter:brightness(1.3)}
  #play:hover,#skip:hover{border-color:var(--ink);color:#fff}
  button:disabled,#prefSet:disabled{opacity:.45;cursor:not-allowed}
  .lit{border-color:#3ddc97 !important;color:#3ddc97 !important;box-shadow:0 0 11px rgba(61,220,151,.40);animation:litpulse 1.8s ease-in-out infinite}
  @keyframes litpulse{0%,100%{box-shadow:0 0 6px rgba(61,220,151,.28)}50%{box-shadow:0 0 14px rgba(61,220,151,.55)}}
  .spin{display:none;width:13px;height:13px;border-radius:50%;border:2px solid var(--line);border-top-color:var(--hrv);animation:spin .7s linear infinite;flex:0 0 auto;margin-right:6px}
  .spin.on{display:inline-block}
  @keyframes spin{to{transform:rotate(360deg)}}
  .ttrackrow{display:flex;align-items:center;min-width:0}
  @media (max-width:880px){ main{grid-template-columns:1fr} .side{position:static} }
</style>
</head>
<body>
<div id="devModal" style="position:fixed;inset:0;z-index:50;background:rgba(6,10,14,.86);backdrop-filter:blur(3px);display:flex;align-items:center;justify-content:center">
  <div style="background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:26px 26px 22px;max-width:420px;width:90%">
    <div style="font-size:13px;letter-spacing:.16em;color:var(--muted);font-weight:600;margin-bottom:4px">SELECT INPUT DEVICE</div>
    <div class="mono" style="font-size:11px;color:var(--faint);margin-bottom:18px">Choose what's measuring your biosignals. This sets which channels the model expects.</div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <button class="devpick" data-dev="arduino" style="text-align:left;padding:13px 15px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);cursor:pointer">
        <div style="font-weight:600;font-size:14px">Arduino kit</div>
        <div class="mono" style="font-size:10px;color:var(--muted);margin-top:3px">All 5 channels · BPM, HRV, EDA, skin temp, motion</div>
      </button>
      <button class="devpick" data-dev="apple_watch" style="text-align:left;padding:13px 15px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);cursor:pointer">
        <div style="font-weight:600;font-size:14px">Apple Watch</div>
        <div class="mono" style="font-size:10px;color:var(--muted);margin-top:3px">3 channels · BPM, HRV, motion (HealthKit + CoreMotion → POST)</div>
      </button>
    </div>
    <div class="mono" style="font-size:10px;color:var(--faint);margin-top:16px">You can switch devices later from the header.</div>
  </div>
</div>
<header>
  <div class="brand"><b>BIOSIGNAL STRESS MONITOR</b><span>CLOSED-LOOP · v1.0</span></div>
  <div class="pill" id="simbadge" style="display:none;color:var(--motion);border-color:var(--motion)">SIMULATION MODE</div>
  <div class="pill" id="devpill" style="cursor:pointer;color:var(--hrv);border-color:var(--hrv)" title="Switch input device">DEVICE</div>
  <div class="pill"><span id="leadcount">5/5 LEADS</span></div>
  <div class="pill"><span class="dot" id="dmodel"></span><span id="modeltxt">MODEL</span></div>
  <div class="spacer"></div>
  <div class="pill mono" id="clock">00:00:00</div>
  <div class="pill"><span class="dot" id="dser"></span><span id="sertxt">SERIAL</span></div>
</header>

<main>
  <section class="traces" id="traces"></section>

  <aside class="side">
    <div class="card gauge">
      <div class="pulse" id="pulse"></div>
      <h2>STRESS INDEX</h2>
      <div class="sub">0–100 · updates every new HRV</div>
      <div class="gwrap">
        <div class="greadout">
          <div class="gnum mono" id="gnum">--</div>
          <div class="gband" id="gband">—</div>
          <div class="gnote" id="gnote">awaiting signal</div>
        </div>
        <div class="gcol" id="gcol">
          <div class="gfill" id="gfill" style="height:0%"></div>
          <div class="gtick" style="bottom:68%"><span>68</span></div>
          <div class="gtick" style="bottom:38%"><span>38</span></div>
        </div>
      </div>
    </div>

    <div class="card music">
      <h2>ADAPTIVE THERAPY</h2>
      <div class="trow">
        <button class="play" id="play" title="Play / pause">►</button>
        <button class="play" id="skip" title="Skip (new song)" style="font-size:15px">⏭</button>
        <div class="tinfo">
          <div class="tstate" id="mstate">Calm program</div>
          <div class="ttrackrow"><span class="spin" id="mspin"></span><div class="ttrack" id="mtrack">no track loaded</div></div>
        </div>
        <span class="badge" id="mband">CALM</span>
      </div>
      <div class="vol">
        <label>VOL</label>
        <input type="range" id="vol" min="0" max="100" value="60">
        <span class="mono" id="voltxt" style="font-size:11px;color:var(--muted);width:30px;text-align:right">60</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:10px;flex-wrap:wrap">
        <button id="srcLocal"   class="srcbtn on" style="font:600 10px/1 var(--mono);letter-spacing:1px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer">LOCAL</button>
        <button id="srcSpotify" class="srcbtn"    style="font:600 10px/1 var(--mono);letter-spacing:1px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--muted);cursor:pointer">SPOTIFY</button>
        <span id="spStatus" class="mono" style="font-size:10px;color:var(--muted)"></span>
      </div>
      <div id="spSetup" style="display:none;flex-direction:column;gap:6px;margin-top:8px">
        <div id="ownRow" style="display:none;gap:8px;align-items:center">
          <input id="ownKey" type="password" autocomplete="off" placeholder="Owner key (printed in the server terminal)"
                 style="flex:1;min-width:0;font:400 12px/1.2 var(--mono);padding:7px 9px;border-radius:6px;border:1px solid var(--line);background:rgba(255,255,255,.03);color:var(--fg);outline:none">
          <button id="ownUnlock" style="font:600 10px/1 var(--mono);letter-spacing:1px;padding:7px 11px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer;white-space:nowrap">UNLOCK</button>
        </div>
        <div id="spControls" style="display:flex;flex-direction:column;gap:6px">
        <div style="display:flex;gap:8px;align-items:center">
          <input id="spKey" type="password" autocomplete="off" placeholder="Your Anthropic API key (sk-ant-...)"
                 style="flex:1;min-width:0;font:400 12px/1.2 var(--mono);padding:7px 9px;border-radius:6px;border:1px solid var(--line);background:rgba(255,255,255,.03);color:var(--fg);outline:none">
          <button id="spSave" style="font:600 10px/1 var(--mono);letter-spacing:1px;padding:7px 11px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer;white-space:nowrap">SAVE KEY</button>
          <a id="spConnect" href="/login" style="display:none;font:600 10px/1 var(--mono);letter-spacing:1px;padding:7px 11px;border-radius:6px;border:1px solid #1db954;color:#1db954;text-decoration:none;white-space:nowrap">CONNECT</a>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <select id="prefType" style="font:500 11px/1 var(--mono);padding:7px 6px;border-radius:6px;border:1px solid var(--line);background:var(--panel2);color:var(--fg);outline:none;cursor:pointer">
            <option value="taste">Vibe</option>
            <option value="artist">Artist</option>
            <option value="song">Song</option>
          </select>
          <input id="prefValue" type="text" placeholder="lo-fi, acoustic, ambient"
                 style="flex:1;min-width:0;font:400 12px/1.2 var(--mono);padding:7px 9px;border-radius:6px;border:1px solid var(--line);background:rgba(255,255,255,.03);color:var(--fg);outline:none">
          <button id="prefSet" style="font:600 10px/1 var(--mono);letter-spacing:1px;padding:7px 11px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer;white-space:nowrap">SET</button>
        </div>
        <div id="prefHint" class="mono" style="font-size:10px;color:var(--muted)">Spotify picks adapt to your stress state. Change this anytime.</div>
        </div>
      </div>
      <audio id="audioA" preload="auto"></audio>
      <audio id="audioB" preload="auto"></audio>
    </div>
  </aside>
</main>

<script>
const VITALS = [
  {key:"bpm",   name:"HEART RATE",       short:"HR",   lead:"PPG",  unit:"bpm", color:getCSS("--bpm"),    dec:0, ymin:40, ymax:160},
  {key:"hrv",   name:"HRV · RMSSD",      short:"HRV",  lead:"PRV",  unit:"ms",  color:getCSS("--hrv"),    dec:0, ymin:0,  ymax:120},
  {key:"eda",   name:"ELECTRODERMAL",    short:"EDA",  lead:"SCL",  unit:"µS",  color:getCSS("--eda"),    dec:2, ymin:0,  ymax:20},
  {key:"temp",  name:"SKIN TEMPERATURE", short:"TEMP", lead:"IR",   unit:"°C",  color:getCSS("--temp"),   dec:1, ymin:30, ymax:38},
  {key:"motion",name:"MOTION",           short:"MOT",  lead:"ACC",  unit:"g",   color:getCSS("--motion"), dec:2, ymin:0,  ymax:3},
];
function getCSS(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
const N = 240;                         // points held per trace (~24 s at the 10 Hz stream rate)
const traces = {};                     // key -> {canvas,ctx,buf,active}
const tickStr = x => String(Math.round(x*10)/10);

const wrap = document.getElementById("traces");
const SPAN = (N*0.1);                   // seconds of history shown
for(const v of VITALS){
  const card = document.createElement("div"); card.className="card vital"; card.id="row-"+v.key;
  card.innerHTML =
    `<div class="vhead">`+
      `<div class="vtitle"><span class="vdot" style="background:${v.color}"></span>`+
        `<span class="vname">${v.name}</span><span class="vlead">${v.lead}</span></div>`+
      `<div class="vread"><span class="val mono" id="val-${v.key}" style="color:${v.color}">--</span>`+
        `<span class="unit">${v.unit}</span></div>`+
    `</div>`+
    `<div class="vbody">`+
      `<div class="yaxis"><span>${tickStr(v.ymax)}</span><span>${tickStr((v.ymax+v.ymin)/2)}</span><span>${tickStr(v.ymin)}</span></div>`+
      `<div class="plot"><canvas></canvas><div class="nosig" id="ns-${v.key}" style="display:none">NO SIGNAL</div></div>`+
    `</div>`+
    `<div class="xaxis"><span>-${SPAN.toFixed(0)}s</span><span>-${(SPAN/2).toFixed(0)}s</span><span>0s · now</span></div>`;
  wrap.appendChild(card);
  const cv = card.querySelector("canvas");
  traces[v.key] = {cv, ctx:cv.getContext("2d"), buf:new Array(N).fill(null), active:true, meta:v, lastVal:null};
}

// ---- device handling: show ONLY the channels the selected device provides ----
let deviceChannels = VITALS.map(v=>v.key);     // default: all five (Arduino)
function applyDeviceChannels(chs){
  if(chs && chs.length) deviceChannels = chs;
  for(const v of VITALS){
    const row=document.getElementById("row-"+v.key);
    if(row) row.style.display = deviceChannels.includes(v.key) ? "" : "none";   // absent channels hidden entirely
  }
}
async function chooseDevice(dev){
  try{ await fetch("/api/device", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({device:dev})}); }catch(e){}
  for(const k in traces){ traces[k].buf=new Array(N).fill(null); traces[k].lastVal=null; }   // clear traces on switch
  document.getElementById("devModal").style.display="none";
}
document.querySelectorAll(".devpick").forEach(b=> b.onclick=()=>chooseDevice(b.dataset.dev));
document.getElementById("devpill").onclick=()=>{ document.getElementById("devModal").style.display="flex"; };
// the modal blocks until a device is picked on first load; the header pill reopens it later

function ensureSize(t){                  // keep the canvas backing store matched to its CSS box (no overlap, crisp lines)
  const r=t.cv.getBoundingClientRect(); if(!r.width||!r.height) return false;
  const dpr=window.devicePixelRatio||1, W=Math.round(r.width*dpr), H=Math.round(r.height*dpr);
  if(t.cv.width!==W||t.cv.height!==H){ t.cv.width=W; t.cv.height=H; t.ctx.setTransform(dpr,0,0,dpr,0,0); }
  return true;
}

function drawTrace(t){
  if(!ensureSize(t)) return;
  const ctx=t.ctx, w=t.cv.clientWidth, h=t.cv.clientHeight, v=t.meta, lo=v.ymin, hi=v.ymax;
  ctx.clearRect(0,0,w,h);
  // grid (aligns with the HTML y-axis ticks at 0/50/100%)
  ctx.strokeStyle=getCSS("--grid"); ctx.lineWidth=1; ctx.beginPath();
  for(let gx=0; gx<=w+1; gx+=Math.max(26,w/8)){ctx.moveTo(gx,0);ctx.lineTo(gx,h);}
  for(let i=0;i<=4;i++){const gy=Math.round(h*i/4)+0.5;ctx.moveTo(0,gy);ctx.lineTo(w,gy);} ctx.stroke();
  ctx.strokeStyle=getCSS("--line"); ctx.strokeRect(0.5,0.5,w-1,h-1);
  if(!t.active){                                   // dashed flatline for an inactive lead
    ctx.strokeStyle=getCSS("--faint"); ctx.lineWidth=1.4; ctx.setLineDash([5,5]);
    ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke(); ctx.setLineDash([]); return;
  }
  ctx.strokeStyle=v.color; ctx.lineWidth=1.7; ctx.lineJoin="round"; ctx.shadowColor=v.color; ctx.shadowBlur=6;
  ctx.beginPath(); let started=false;
  for(let i=0;i<N;i++){const val=t.buf[i]; if(val===null)continue;
    const cl=Math.max(lo,Math.min(hi,val));
    const x=(i/(N-1))*w, y=h-((cl-lo)/(hi-lo))*h;
    if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);}
  ctx.stroke(); ctx.shadowBlur=0;
}
function frame(){ for(const k in traces) drawTrace(traces[k]); requestAnimationFrame(frame); } frame();

// ---- stress gauge ----
function bandColor(b){return getCSS(b==="elevated"?"--elevated":b==="neutral"?"--neutral":"--calm");}
function bandLabel(b){return b==="elevated"?"ELEVATED":b==="neutral"?"ELEVATED·MILD":"CALM";}
let lastInf=-1;
function setGauge(s){
  const num=document.getElementById("gnum"), fill=document.getElementById("gfill"),
        band=document.getElementById("gband"), note=document.getElementById("gnote");
  if(s.stress===null || !s.model_loaded){
    num.textContent="--"; fill.style.height="0%"; band.textContent="—";
    note.textContent = s.model_loaded? "awaiting signal":"model not loaded"; return;
  }
  const col=bandColor(s.band);
  num.textContent = Math.round(s.stress); num.style.color=col;
  fill.style.height = s.stress+"%";
  fill.style.background = "linear-gradient(180deg,"+col+"33,"+col+")";
  band.textContent = s.calibrating? "CALIBRATING" : bandLabel(s.band);
  band.style.color = s.calibrating? getCSS("--muted"):col;
  note.textContent = s.calibrating ? "building baseline…"
                   : (s.n_active<s.n_total ? ("degraded · "+s.n_active+"/"+s.n_total+" leads") : "all leads nominal");
  if(s.inf_seq!==lastInf){                          // pulse on each new HRV-driven update
    lastInf=s.inf_seq; const p=document.getElementById("pulse");
    p.style.background=col; p.style.boxShadow="0 0 12px "+col; p.style.transform="scale(1.7)";
    setTimeout(()=>{p.style.transform="scale(1)";p.style.boxShadow="none";p.style.background=getCSS("--faint");},150);
  }
}

// ---- music: two-element crossfade player + server-side played-song bin ----
const players=[document.getElementById("audioA"), document.getElementById("audioB")];
const playBtn=document.getElementById("play"), vol=document.getElementById("vol"), voltxt=document.getElementById("voltxt");
let active=0, curBand=null, playing=false, targetVol=0.6, switchSeq=0;
let playedBand=null, fetching=false;   // band the audible song was fetched for + request-in-flight flag (state-match watchdog)
const CROSSFADE_SEC=2.5, PROGRAM={calm:"Calm program",neutral:"Steadying program",elevated:"Down-regulation program"};
players.forEach(p=>{ p.volume=0; });

function setTrackName(name,band){ document.getElementById("mtrack").textContent = name || ("no tracks in "+(band||curBand||"?")+"/"); }
let localName=null, spName=null;
function setLoading(on){ const s=document.getElementById("mspin"); if(s) s.classList.toggle("on", !!on); }   // spinner while a song is changing
function localDisplayName(){ const el=players[active]; try{ const f=decodeURIComponent((el.src||"").split("/").pop()); return f || null; }catch(e){ return null; } }
function setLocal(name){ localName=name; if(!spotifyActive()) setTrackName(name, curBand); }   // show ONLY when local is the audible source
function setSp(name){ spName=name; if(spotifyActive()) setTrackName(name); }                    // show ONLY when Spotify is the audible source
function showActiveName(){ if(spotifyActive()) setTrackName(spName); else setTrackName(localName!==null?localName:localDisplayName(), curBand); }
async function fetchTrack(band){ try{ const r=await fetch("/api/track?band="+encodeURIComponent(band)); return await r.json(); }catch(e){ return null; } }
function cancelFade(el){ if(el._fade){ cancelAnimationFrame(el._fade); el._fade=null; } }
function fade(el, to, ms, done){            // animate one element's volume to a target
  cancelFade(el); const from=el.volume, t0=performance.now();
  const step=t=>{ const k=ms<=0?1:Math.min(1,(t-t0)/ms); el.volume=Math.max(0,Math.min(1,from+(to-from)*k));
    if(k<1) el._fade=requestAnimationFrame(step); else { el._fade=null; if(done) done(); } };
  el._fade=requestAnimationFrame(step);
}

// Switch to `url` with a crossfade. Serialized via switchSeq so a newer switch supersedes a stale one,
// guaranteed to fire (canplay/readyState/timeout), and an enforcer guarantees only the active track is audible.
function enforce(seq){                                // hard final state: active plays at volume; the other is paused + silent
  if(seq!==switchSeq) return;
  players.forEach((p,i)=>{
    if(i===active){ cancelFade(p); p.volume=targetVol; if(playing && p.paused) p.play().catch(()=>{}); }
    else { cancelFade(p); try{ if(!p.paused) p.pause(); }catch(e){} p.volume=0; }
  });
}
function switchTo(url, name, forBand){
  if(!url){ setTrackName(null); fetching=false; return; }
  const seq = ++switchSeq;
  const cur = players[active], nxt = players[active^1];
  console.log("[music] switch", seq, "->", name, "(for band", forBand + ")");
  cancelFade(nxt); try{ nxt.pause(); }catch(e){} nxt.volume=0;
  let begun=false;
  const begin=()=>{
    if(begun || seq!==switchSeq) return;          // run once, and only if still the latest switch
    begun=true;
    nxt.play().then(()=>{
      if(seq!==switchSeq) return;                 // superseded while loading -> let the newer one win
      setLocal(name); setLoading(false);
      playedBand = forBand || curBand; fetching=false;   // record which band this song was fetched FOR (watchdog uses it)
      active ^= 1;
      console.log("[music] playing", name, "on element", active);
      fade(players[active], targetVol, CROSSFADE_SEC*1000);
      fade(cur, 0, CROSSFADE_SEC*1000, ()=>{ try{cur.pause();}catch(e){} });
      setTimeout(()=>enforce(seq), CROSSFADE_SEC*1000 + 250);   // <-- kill any lingering old track, force new one audible
    }).catch(err=>{                                // playback blocked -> keep the CURRENT track audible (never mute)
      console.warn("[music] play blocked:", err && err.name); setLoading(false); fetching=false;
      try{ cur.volume=targetVol; if(cur.paused) cur.play(); }catch(e){}
    });
  };
  nxt.oncanplay = begin;                            // primary trigger
  try{ nxt.src = url; nxt.load(); }catch(e){}
  if(nxt.readyState >= 3) begin();                  // already buffered
  setTimeout(begin, 1500);                          // fallback if canplay never fires (the old silent-switch bug)
}
async function changeTrack(){
  const band = curBand; setLoading(true); fetching=true;
  const j = await fetchTrack(band);
  if(band !== curBand){ setLoading(false); fetching=false; return; }   // state changed during the fetch -> newer change wins (watchdog re-checks anyway)
  if(j&&j.url) switchTo(j.url, j.name, band); else { setLoading(false); fetching=false; setLocal(null); playedBand=band; }
}
async function preloadFor(band){                    // stage a track silently so the first Play is instant (unlocks audio)
  const j=await fetchTrack(band); const el=players[active]; cancelFade(el); try{ el.pause(); }catch(e){}
  if(j&&j.url){ try{ el.src=j.url; }catch(e){} el.volume=0; setLocal(j.name); playedBand=band; } else setLocal(null);
}

players.forEach(p=>{
  p.addEventListener("play", ()=>{ p._adv=false; });           // reset the per-track advance guard
  p.addEventListener("timeupdate", ()=>{                        // start the next song just before this one ends
    if(p!==players[active] || !playing || p._adv) return;
    if(isFinite(p.duration) && p.duration>0 && p.currentTime >= p.duration-CROSSFADE_SEC){ p._adv=true; changeTrack(); }
  });
  p.addEventListener("ended", ()=>{ if(p===players[active] && playing){ p._adv=true; changeTrack(); } });
});

let primed=false;
playBtn.onclick=async ()=>{
  playing=!playing;
  if(playing){
    playBtn.textContent="❚❚";
    if(spotifyActive()){
      if(spHasTrack && spPlayer){ spPlayer.resume().catch(()=>{}); }   // resume the SAME song — no new request
      else { await maybeInitSpotify(); await spPlayBand(curBand); }    // first play only: load a song
      return;
    }
    const el=players[active], other=players[active^1];
    if(!primed){ primed=true;                       // unlock the idle element inside this gesture so later crossfades may play (Safari)
      if(!other.src && el.src){ try{ other.src=el.src; }catch(e){} }   // needs a src to actually start (and unlock)
      const pr=other.play(); if(pr&&pr.then) pr.then(()=>{ try{other.pause();}catch(e){} }).catch(()=>{}); }
    if(!el.src){ setLoading(true); const j=await fetchTrack(curBand); if(j&&j.url){ try{el.src=j.url;}catch(e){} setLocal(j.name); } setLoading(false); }
    cancelFade(el); el.volume=0;                     // fade IN instead of a hard start
    try{ await el.play(); fade(el, targetVol, CROSSFADE_SEC*1000); setLocal(localDisplayName()); }
    catch(e){ el.volume=targetVol; setTrackName("tap ► again to start audio"); }
  } else {
    playBtn.textContent="►";
    if(spotifyActive()){ if(spPlayer) spPlayer.pause().catch(()=>{}); return; }   // pause Spotify, keep its song loaded
    players.forEach(p=>{ cancelFade(p); p.pause(); });                            // pause local, keep its src loaded
  }
};
vol.oninput=()=>{ targetVol=vol.value/100; voltxt.textContent=vol.value;
  if(spotifyActive()){ if(spPlayer) spPlayer.setVolume(targetVol).catch(()=>{}); return; }
  const a=players[active]; if(a&&!a._fade&&playing) a.volume=targetVol; };

document.getElementById("skip").onclick = async ()=>{   // SKIP = explicitly request a NEW song for the current band
  if(!playing){ playing=true; playBtn.textContent="❚❚"; }
  if(spotifyActive()){ await maybeInitSpotify(); await spPlayBand(curBand); }   // new Spotify request (crossfades)
  else { await changeTrack(); }                                                 // new local request (crossfades)
};

function setMusic(s){
  document.getElementById("mstate").textContent=PROGRAM[s.band]||"—";
  const mb=document.getElementById("mband"); mb.textContent=s.band.toUpperCase(); mb.style.color=bandColor(s.band); mb.style.borderColor=bandColor(s.band);
  if(s.band!==curBand){ curBand=s.band;                          // band changed = stress state changed
    if(spotifyActive()){ if(playing) spPlayBand(curBand); }
    else if(playing) changeTrack(); else preloadFor(curBand);
  }
  // WATCHDOG (10 Hz): the song that's audible must match the CURRENT state. If any race (e.g. a
  // skip landing right as the state flips) leaves a mismatched song playing, re-request now.
  if(fetching){ if(!window._ft) window._ft=Date.now(); else if(Date.now()-window._ft>8000){ fetching=false; window._ft=0; } }
  else window._ft=0;                                  // failsafe: a hung request can't pin `fetching` forever
  if(playing && !fetching && playedBand!==null && playedBand!==curBand){
    console.log("[music] watchdog: playing", playedBand, "but state is", curBand, "-> correcting");
    fetching=true;                                   // claim immediately so the 10 Hz tick can't double-fire
    if(spotifyActive()) spPlayBand(curBand); else changeTrack();
  }
  // SOURCE WATCHDOG (10 Hz): audio may only come from the selected source.
  if(spotifyActive()){                                // Spotify mode -> every local element must be silent
    players.forEach(p=>{ if(!p.paused){ cancelFade(p); try{p.pause();}catch(e){} p.volume=0; console.log("[music] source watchdog: silenced local"); } });
  } else if(spPlayer && spIsPlaying){                 // local mode (or Spotify unusable) -> the Spotify stream must be paused
    spIsPlaying=false; spPlayer.pause().catch(()=>{}); console.log("[music] source watchdog: paused Spotify");
  }
}

// ---- Spotify Web Playback SDK source (optional; toggled in the UI) ----
let musicSource="local", spPlayer=null, spDeviceId=null, spToken=null, spConnected=false, spKeySet=false, spHasTrack=false, spIsPlaying=false, spOwner=false;
function spotifyActive(){ return musicSource==="spotify" && spConnected; }   // Spotify drives audio ONLY when connected; otherwise local does
async function spGetToken(){ try{ const r=await fetch("/api/spotify/token"); const j=await r.json(); spToken=j.access_token; return spToken; }catch(e){ return null; } }
async function refreshSpStatus(){
  try{ const r=await fetch("/api/spotify/status"); const j=await r.json();
    spConnected=!!j.connected; spKeySet=!!j.key_set; spOwner=!!j.owner;
    document.getElementById("ownRow").style.display     = spOwner ? "none" : "flex";   // non-owner -> only the lock row
    document.getElementById("spControls").style.display = spOwner ? "flex" : "none";
    document.getElementById("spConnect").style.display = (spOwner && j.configured && !j.connected) ? "inline-block" : "none";
    let msg;
    if(!spOwner)           msg = "Spotify is owner-only on this server";
    else if(!j.configured) msg = "server: Spotify keys not set";
    else if(!j.key_set)    msg = "enter your Anthropic key →";
    else if(!j.connected)  msg = "key saved · connect Spotify →";
    else                   msg = "● ready";
    const st=document.getElementById("spStatus"); st.textContent=msg; st.style.color = j.connected ? "#1db954" : "var(--muted)";
    document.getElementById("spSave").classList.toggle("lit", spOwner && j.configured && !j.key_set);   // SAVE KEY glows when a key is still needed
    document.getElementById("spConnect").classList.toggle("lit", spOwner && j.configured && j.key_set && !j.connected);
    refreshPrefLit();
    if(spConnected) maybeInitSpotify();
  }catch(e){}
}
document.getElementById("ownUnlock").onclick = async ()=>{
  const key=document.getElementById("ownKey").value.trim(); if(!key) return;
  try{
    const r=await fetch("/api/owner", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({key:key})});
    const j=await r.json();
    if(!j.owner) flashStatus("wrong owner key");
  }catch(e){}
  document.getElementById("ownKey").value="";
  refreshSpStatus();
};
document.getElementById("ownKey").addEventListener("keydown", e=>{ if(e.key==="Enter") document.getElementById("ownUnlock").click(); });
function refreshPrefLit(){   // SET glows only when it can do something: a value is present AND Spotify is usable
  const hasVal = document.getElementById("prefValue").value.trim().length>0;
  const ready  = spOwner && spKeySet && spConnected;
  const set = document.getElementById("prefSet");
  set.classList.toggle("lit", hasVal && ready);
  set.disabled = !hasVal;
}
document.getElementById("spSave").onclick = async ()=>{
  const key=document.getElementById("spKey").value.trim();
  try{ await fetch("/api/config", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({anthropic_key:key})}); }catch(e){}
  document.getElementById("spKey").value="";        // don't keep the key sitting in the input
  refreshSpStatus();
};
const PREF_PH = { taste:"lo-fi, acoustic, ambient", artist:"artist name, e.g. Bon Iver", song:"song — artist, e.g. Holocene Bon Iver" };
const PREF_HINT = { taste:"Spotify picks adapt to your stress state. Change this anytime.",
                    artist:"Only this artist's songs, chosen to fit your stress state.",
                    song:"Plays this exact song." };
document.getElementById("prefType").onchange = ()=>{
  const t=document.getElementById("prefType").value;
  document.getElementById("prefValue").placeholder = PREF_PH[t]||"";
  document.getElementById("prefHint").textContent = PREF_HINT[t]||"";
};
async function applyPref(){
  const type=document.getElementById("prefType").value, value=document.getElementById("prefValue").value.trim();
  if(!value) return;
  setLoading(true);                                  // show the spinner while we switch (no tab reload)
  try{ await fetch("/api/pref", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({type:type, value:value})}); }catch(e){}
  refreshPrefLit();
  if(spotifyActive()){                               // apply immediately + smoothly if Spotify is the source
    if(!playing){ playing=true; playBtn.textContent="❚❚"; }
    await spPlayBand(curBand, {exact:true});         // a freshly-set song plays EXACTLY; vibe/artist just refresh
  } else { setLoading(false); }                       // not on Spotify yet -> stored for when you connect/switch
}
document.getElementById("prefSet").onclick = applyPref;
document.getElementById("prefValue").addEventListener("input", refreshPrefLit);     // light SET as soon as there's something to set
document.getElementById("prefValue").addEventListener("keydown", e=>{ if(e.key==="Enter") applyPref(); });
async function maybeInitSpotify(){
  if(spPlayer || !window._sdkLoaded || !spConnected) return;
  await spGetToken();
  spPlayer = new Spotify.Player({ name:"Stress Dashboard", volume: targetVol,
    getOAuthToken: cb => { spGetToken().then(t=> t && cb(t)); } });
  spPlayer.addListener("ready", ({device_id})=>{ spDeviceId=device_id; console.log("[spotify] device", device_id); });
  spPlayer.addListener("not_ready", ()=>{ spDeviceId=null; });
  spPlayer.addListener("player_state_changed", st=>{ spIsPlaying = !!(st && !st.paused); });   // source watchdog reads this
  spPlayer.addListener("authentication_error", ({message})=>{ console.warn("[spotify] auth:", message); });
  spPlayer.addListener("account_error", ({message})=>{ console.warn("[spotify] account (Premium required):", message); document.getElementById("spStatus").textContent="needs Premium"; });
  spPlayer.addListener("initialization_error", ({message})=>{ console.warn("[spotify] init:", message); });
  spPlayer.connect();
}
const SP_FADE_MS = 1200;
async function spFadeVolume(from, to, ms){       // ramp the single Spotify stream's volume
  if(!spPlayer) return;
  const steps=16, dt=Math.max(15, ms/steps);
  for(let i=1;i<=steps;i++){
    const v = from + (to-from)*(i/steps);
    try{ await spPlayer.setVolume(Math.max(0, Math.min(1, v))); }catch(e){}
    await new Promise(r=>setTimeout(r, dt));
  }
}
function flashStatus(msg){                            // transient note in the status line; restores itself
  const st=document.getElementById("spStatus"); if(!st) return;
  if(st._t) clearTimeout(st._t);
  if(st._orig===undefined) st._orig=st.textContent;
  st.textContent=msg; st.style.color="var(--neutral, #f2c14e)";
  st._t=setTimeout(()=>{ st.textContent=st._orig; st._orig=undefined; st.style.color=spConnected?"#1db954":"var(--muted)"; refreshSpStatus(); }, 4000);
}
async function spPlayBand(band, opts){
  opts = opts || {};
  if(!spOwner){ flashStatus("Spotify is owner-only on this server"); fetching=false; playedBand=band; return; }
  if(!spConnected){ setTrackName("connect Spotify first"); fetching=false; return; }
  setLoading(true); fetching=true;
  if(!spDeviceId){ await maybeInitSpotify(); await new Promise(r=>setTimeout(r,600)); }   // give the device a moment to register
  if(!spDeviceId){ setLoading(false); fetching=false; setTrackName("Spotify player not ready"); return; }
  const url = "/api/spotify/track?band="+encodeURIComponent(band) + (opts.exact ? "&exact=1" : "");
  let j; try{ j=await (await fetch(url)).json(); }catch(e){ setLoading(false); fetching=false; return; }
  if(band !== curBand){ setLoading(false); fetching=false; return; }   // state changed during the fetch -> newer change wins (watchdog re-checks anyway)
  if(!j || !j.uri){                                   // lookup found nothing -> KEEP the current song + text; just note it and move on
    setLoading(false); fetching=false; playedBand=band;               // mark satisfied so the watchdog doesn't retry-loop
    console.warn("[spotify] no track found:", (j&&j.error)||"(empty result)");
    flashStatus("couldn't find that — keeping current song");
    return;
  }
  const tok = await spGetToken();
  try{
    if(spHasTrack && spPlayer){ await spFadeVolume(targetVol, 0, SP_FADE_MS); }   // fade the current track down first
    await fetch("https://api.spotify.com/v1/me/player/play?device_id="+spDeviceId, {
      method:"PUT", headers:{ "Authorization":"Bearer "+tok, "Content-Type":"application/json" },
      body: JSON.stringify({ uris:[j.uri] }) });
    setSp(j.artist ? (j.artist+" — "+j.name) : j.name);
    spHasTrack=true; spIsPlaying=true; setLoading(false);
    playedBand=band; fetching=false;                  // record which band this song serves (watchdog checks it at 10 Hz)
    if(spPlayer){ try{ await spPlayer.setVolume(0); }catch(e){} await spFadeVolume(0, targetVol, SP_FADE_MS); }   // fade the new track up
    console.log("[spotify] playing", j.uri);
  }catch(e){ setLoading(false); fetching=false; console.warn("[spotify] play failed:", e); }
}
async function resumeLocal(){                         // (re)start local playback on the active element (used when returning to LOCAL)
  const el=players[active], other=players[active^1];
  if(!primed){ primed=true; if(!other.src && el.src){ try{ other.src=el.src; }catch(e){} }
    const pr=other.play(); if(pr&&pr.then) pr.then(()=>{ try{other.pause();}catch(e){} }).catch(()=>{}); }
  cancelFade(el);
  if(!el.src){ setLoading(true); const j=await fetchTrack(curBand); if(j&&j.url){ try{el.src=j.url;}catch(e){} } setLoading(false); }
  el.volume=0;                                         // fade IN on return to local instead of a hard start
  try{ await el.play(); fade(el, targetVol, CROSSFADE_SEC*1000); setLocal(localDisplayName()); }
  catch(e){ el.volume=targetVol; setTrackName("tap ► to start audio"); }
}
function setSource(src){
  musicSource = src;
  document.getElementById("srcLocal").classList.toggle("on", src==="local");
  document.getElementById("srcSpotify").classList.toggle("on", src==="spotify");
  document.getElementById("srcLocal").style.color   = src==="local"   ? "var(--fg)" : "var(--muted)";
  document.getElementById("srcSpotify").style.color = src==="spotify" ? "var(--fg)" : "var(--muted)";
  document.getElementById("spSetup").style.display = src==="spotify" ? "flex" : "none";
  if(src==="spotify"){
    refreshSpStatus();
    if(spConnected){                                  // Spotify ready -> it takes over the audio
      players.forEach(p=>{ cancelFade(p); p.pause(); });
      maybeInitSpotify();
      if(playing){ if(spHasTrack && spPlayer){ spPlayer.resume().catch(()=>{}); showActiveName(); } else spPlayBand(curBand); }
      else showActiveName();                          // paused: still show the last Spotify track
    } else { showActiveName(); }                      // not connected -> local keeps playing; show the LOCAL track
  } else {                                            // back to LOCAL
    if(spPlayer) spPlayer.pause().catch(()=>{});      // stop Spotify (its track stays loaded for later)
    showActiveName();                                 // <-- update the text to the local song (fixes wrong track text on tab switch)
    if(playing) resumeLocal();                        // resume the local library so sound continues
  }
}
document.getElementById("srcLocal").onclick   = ()=> setSource("local");
document.getElementById("srcSpotify").onclick = ()=> setSource("spotify");
// CONNECT opens the OAuth flow in a POPUP so the dashboard itself never navigates or reloads.
document.getElementById("spConnect").onclick = (e)=>{
  e.preventDefault();
  window.open("/login", "spotify_auth", "width=520,height=720,menubar=no,toolbar=no");
};
window.addEventListener("message", (e)=>{           // the popup announces success, then closes itself
  if(e.data === "spotify-connected"){ refreshSpStatus(); }
});
refreshSpStatus();

// ---- websocket ----
function fmt(v,d){ return v===null||v===undefined? "--" : Number(v).toFixed(d); }
function flashRead(key){ const el=document.getElementById("val-"+key); el.classList.remove("tick"); void el.offsetWidth; el.classList.add("tick"); }
function setStatus(s){
  const dser=document.getElementById("dser"), dm=document.getElementById("dmodel"), sertxt=document.getElementById("sertxt");
  document.getElementById("simbadge").style.display = (s.source==="sim") ? "flex" : "none";
  if(s.source==="sim"){
    dser.className="dot"; dser.style.background=getCSS("--motion"); dser.style.boxShadow="0 0 8px "+getCSS("--motion");
    sertxt.textContent="SIMULATED INPUT";
  } else {
    dser.style.background=""; dser.style.boxShadow="";
    dser.className="dot "+(s.serial_ok?"ok":"bad"); sertxt.textContent=s.serial_ok?"SERIAL OK":"NO SERIAL";
  }
  dm.className="dot "+(s.model_loaded?"ok":"bad");  document.getElementById("modeltxt").textContent=s.model_loaded?"MODEL OK":"NO MODEL";
  document.getElementById("leadcount").textContent=s.n_active+"/"+s.n_total+" LEADS";
  if(s.device){ const DL={arduino:"ARDUINO", apple_watch:"APPLE WATCH"};
    document.getElementById("devpill").textContent = DL[s.device] || s.device.toUpperCase(); }
  if(s.device_channels) applyDeviceChannels(s.device_channels);   // keep visible traces in sync with the device
  const t=s.session_sec, hh=String(Math.floor(t/3600)).padStart(2,"0"),
        mm=String(Math.floor(t%3600/60)).padStart(2,"0"), ss=String(t%60).padStart(2,"0");
  document.getElementById("clock").textContent=hh+":"+mm+":"+ss;
}
function connect(){
  const ws=new WebSocket((location.protocol==="https:"?"wss://":"ws://")+location.host+"/ws");
  ws.onmessage=ev=>{ const s=JSON.parse(ev.data);
    for(const v of VITALS){ const t=traces[v.key];
      if(!deviceChannels.includes(v.key)) continue;    // channel hidden for this device -> skip its trace updates
      const act=s.active[v.key]; const r=s.readings[v.key];
      t.active=act; t.buf.push(act? r : null); if(t.buf.length>N)t.buf.shift();
      const el=document.getElementById("val-"+v.key);
      if(act && r!==t.lastVal){ el.textContent=fmt(r,v.dec); flashRead(v.key); t.lastVal=r; }   // new reading => update + flash (per beat; HRV per 2 beats)
      else if(!act){ el.textContent="--"; t.lastVal=null; }
      document.getElementById("ns-"+v.key).style.display = act? "none":"block";
      document.getElementById("row-"+v.key).classList.toggle("off", !act);
    }
    setStatus(s); setGauge(s); setMusic(s);
  };
  ws.onclose=()=>setTimeout(connect,1000);
}
connect();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=serial_loop,  daemon=True).start()
    threading.Thread(target=sampler_loop, daemon=True).start()
    threading.Thread(target=simulate_loop, daemon=True).start()
    url = f"http://{HOST}:{PORT}"
    print(f"[app] dashboard at {url}   (serial: {SERIAL_PORT} @ {BAUD})")
    print(f"[owner] Spotify owner key: {OWNER_KEY}")
    print( "[owner] Spotify is restricted to the app owner. Local browsers unlock automatically;")
    print( "[owner] a remote browser must enter this key (it is only visible here, in YOUR terminal).")
    try: threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    except Exception: pass
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
