# Music & AI for Stress Management

A closed-loop biofeedback system. Custom Arduino hardware streams **heart rate (BPM), heart-rate
variability (HRV / RMSSD), and electrodermal activity (EDA)** over serial; a **CNN+LSTM fusion model**
classifies the wearer's stress level in real time; and the app responds with **adaptive music** that
crossfades between calm, neutral, and elevated programs as the stress index moves — actively steering
the listener back toward a calm state.

Built for the Congressional App Challenge.

---

## What it does

- **Reads live biosignals** from an Arduino over USB serial (BPM, RMSSD, EDA).
- **Scores stress 0–100** every new HRV reading using a two-branch fusion model
  (CNN+LSTM over the raw signal + an MLP over hand-crafted features), with per-session
  z-scoring, temperature calibration, and EMA smoothing.
- **Maps the score to a band** — calm (<38), neutral (38–68), elevated (>68) — with hysteresis
  and a minimum dwell time so the music doesn't flap on small fluctuations.
- **Plays adaptive music** that crossfades on every band change. Two selectable sources:
  - **Local** — your own library (e.g. Suno-generated piano), organized into `calm/`, `neutral/`,
    `elevated/` folders. No account required.
  - **Spotify** (optional) — Claude (Anthropic API) recommends real songs matching the current
    stress state *and* your preference (a vibe, a single artist, or a specific song); the Spotify
    Web Playback SDK streams them in-browser (Spotify Premium required).
- **Clinical-style dashboard** with live vital traces, a stress gauge, and the music controls.

**No hardware?** The app automatically runs a physiologically coherent **simulation mode**, so the
entire closed loop can be demoed without an Arduino.

---

## Quick start

```bash
git clone https://github.com/ArjanWalia/Music-and-AI-for-Stress-Management.git
cd Music-and-AI-for-Stress-Management
pip install -r requirements.txt
python3 server.py
```

Then open **http://127.0.0.1:8000**.

- No model files present → dashboard runs in signal-only mode (no stress index).
- No Arduino on the serial port → simulation mode drives the signals.

> If you don't have a `requirements.txt`, install directly:
> `pip install fastapi "uvicorn[standard]" pyserial torch numpy requests anthropic`

---

## Project layout

```
server.py            # FastAPI app: serial reader, model inference, dashboard, music engine
fetch_spotify.py     # Claude -> Spotify Search pipeline (state + preference -> playable track)
outputs/             # model artifacts: fusionnet.pt, feature_scaler.npz, model_meta.json
suno_ai_songs/       # local music library
  calm/  neutral/  elevated/
arduino/             # the sketch (prints e.g. "BPM:72,RMSSD:41.2,EDA:5.34" over serial @115200)
```

---

## Configuration

Everything is optional for local-only use and is read from environment variables.

| Variable | Purpose | Default |
|---|---|---|
| `STRESS_PORT` | Arduino serial port (`ls /dev/cu.*` on macOS to find it) | `/dev/cu.usbmodem101` |
| `STRESS_BAUD` | serial baud rate (match your sketch) | `115200` |
| `STRESS_MODEL` | model artifacts directory | `./outputs` |
| `STRESS_MUSIC` | local music directory | `./suno_ai_songs` |
| `SPOTIFY_CLIENT_ID` | Spotify app client ID (Spotify source only) | — |
| `SPOTIFY_CLIENT_SECRET` | Spotify app client secret | — |
| `STRESS_OWNER_KEY` | fixed key for the Spotify owner gate (else generated per run) | random |

Each user supplies their **own Anthropic API key** in the dashboard UI — it is kept in memory for
their session only and never written to disk.

Set variables in your shell, e.g.:

```bash
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret
```

---

## Spotify setup (optional)

1. Create an app in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   and enable **Web API** + **Web Playback SDK**.
2. Add this redirect URI **exactly**:
   ```
   http://127.0.0.1:8000/callback
   ```
   `127.0.0.1` is the loopback address — it means "this machine" everywhere — so this single
   redirect URI works on any computer that runs the server locally. **Run the server on the same
   machine as the browser.**
3. Export `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`, then restart `server.py`.
4. In the dashboard: switch to **SPOTIFY** → save your Anthropic key → **CONNECT** (Premium account)
   → choose a **Vibe / Artist / Song** preference.

**Owner gate.** Because Spotify apps run in development mode with a small allowlist, the Spotify
feature is restricted to whoever controls this server's environment: browsers on the host machine
(loopback) unlock automatically; anyone else must enter the owner key printed in the server terminal
at startup.

---

## How the adaptive loop works

1. A serial thread parses each Arduino line; every new HRV reading triggers an inference.
2. The fusion model outputs a 0–100 stress index (z-scored per session, temperature-calibrated,
   EMA-smoothed).
3. The index maps to a band with hysteresis + minimum dwell.
4. On each band change the music crossfades to a fitting track; **skip** draws from the band's
   unplayed bin; watchdogs guarantee the audible song always matches both the current stress state
   and the selected source (local vs. Spotify).

---

## Security notes

- **Never commit secrets.** Keep a `.env` (gitignored); the Spotify client secret and owner key live
  only in the server's environment.
- Users' Anthropic keys are per-session and in-memory only.
- The repository should contain no `.env`, no model weights, and no audio files committed in the open.

---

## Acknowledgements

- Stress classification trained on the **WESAD** and **VitaStress** datasets (plus PhysioNet wearable
  stress/exercise recordings).
- Local adaptive music generated with **Suno**; recommendations via the **Anthropic API**; streaming
  via the **Spotify Web Playback SDK**.
