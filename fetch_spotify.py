#!/usr/bin/env python3
"""
fetch_spotify.py — pick a Spotify song that matches the classified stress state + the user's music taste.

PIPELINE
    stress state  +  user's described taste
          │  (Claude = the "music AI" / recommender)
          ▼
    song candidates  [{artist, title}, ...]
          │  (Spotify Search API — /v1/search)
          ▼
    real Spotify tracks  ->  one playable track  ->  handed to the web app

WHY THIS SHAPE (and not Spotify's own recommender):
    On 2024-11-27 Spotify deprecated, for any app registered on/after that date:
    Recommendations, Audio Features, Audio Analysis, Related Artists, and the 30s preview_url in
    multi-get responses. (They cited concern about people training AI on those signals.) The /v1/search
    endpoint is still open. So we generate candidates with an LLM and resolve them through Search.

REQUIREMENTS (the "assume we qualify" part)
    Spotify app:    export SPOTIFY_CLIENT_ID=...   export SPOTIFY_CLIENT_SECRET=...
                    (Client-Credentials flow — covers Search; no user login needed just to FIND tracks.)
    Anthropic key:  export ANTHROPIC_API_KEY=...
    Optional:       export ANTHROPIC_MODEL=claude-sonnet-4-6   export SPOTIFY_MARKET=US
    Install:        pip install requests anthropic

PLAYBACK (read this — it's the honest part)
    fetch_for_state() returns every handle the web app might need:
        uri          spotify:track:...      -> for the Web Playback SDK or to open in the Spotify app
        external_url https://open.spotify.. -> "open in Spotify"
        preview_url  30s mp3 OR None        -> playable in a plain <audio> tag when present
    For NEW apps Spotify usually returns preview_url = None, so FULL in-browser playback needs the
    Spotify Web Playback SDK (Premium + user OAuth, browser-side JS — not pure Python). Where a
    preview_url IS present, the existing crossfade <audio> player in stress_app.py can play it as-is.

CLI
    python3 fetch_spotify.py                 # asks taste, prints a pick for calm / neutral / elevated
    python3 fetch_spotify.py --state elevated
"""

import os
import sys
import json
import time
import base64
import random
import threading
import requests

# ----------------------------- config -----------------------------
SPOTIFY_CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
SPOTIFY_MARKET        = os.environ.get("SPOTIFY_MARKET", "US")
CANDIDATES_PER_STATE  = 8          # how many songs to ask the LLM for per refill
SERVE_BATCH           = 5          # how many resolved tracks to keep queued per state
TASTE_PATH            = os.path.expanduser("~/.stress_music_taste")

# This is a stress-REDUCTION app, so each band has a therapeutic job, not just a vibe.
# (Matches the dashboard's calm / steadying / down-regulation "programs".)
STATE_INTENT = {
    "calm":     "a peaceful, calm song — gentle, relaxed, and soothing",
    "neutral":  "a chill song that still has some energy — easygoing and pleasant, not sleepy",
    "elevated": "a peaceful, soothing song that actively calms the listener down and brings their stress level down",
}
# accept whatever the classifier emits
STATE_ALIASES = {
    "stressed": "elevated", "high": "elevated", "high_stress": "elevated",
    "medium": "neutral", "mid": "neutral", "moderate": "neutral",
    "low": "calm", "relaxed": "calm", "rest": "calm",
}


# ------------------- Spotify auth (client credentials) -------------------
_token = {"value": None, "exp": 0.0}
_token_lock = threading.Lock()

def _spotify_token():
    """Client-credentials token. Sufficient for /v1/search. Cached until ~expiry."""
    with _token_lock:
        if _token["value"] and time.time() < _token["exp"] - 30:
            return _token["value"]
        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
            raise RuntimeError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        basic = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {basic}"},
            timeout=15,
        )
        r.raise_for_status()
        j = r.json()
        _token["value"] = j["access_token"]
        _token["exp"] = time.time() + float(j.get("expires_in", 3600))
        return _token["value"]


def _search(params):
    """GET /v1/search with one automatic 401 retry (token refresh)."""
    url = "https://api.spotify.com/v1/search"
    for attempt in (1, 2):
        r = requests.get(url, headers={"Authorization": f"Bearer {_spotify_token()}"},
                         params=params, timeout=15)
        if r.status_code == 401 and attempt == 1:
            _token["value"] = None        # force refresh, retry once
            continue
        r.raise_for_status()
        return r.json()
    return {}


def _track_dict(t):
    """Normalize a Spotify track object to our compact dict."""
    imgs = t.get("album", {}).get("images", [])
    return {
        "id": t["id"],
        "uri": t["uri"],
        "name": t["name"],
        "artist": ", ".join(a["name"] for a in t.get("artists", [])),
        "album": t.get("album", {}).get("name", ""),
        "image": imgs[0]["url"] if imgs else None,
        "preview_url": t.get("preview_url"),              # often None for new apps
        "external_url": t.get("external_urls", {}).get("spotify"),
        "duration_ms": t.get("duration_ms"),
    }


def _artist_matches(track, artist):
    """True if `artist` is (part of) the track's artist string, case-insensitive."""
    return artist.strip().lower() in (track.get("artist", "") or "").lower()


def spotify_search_track(artist, title, market=None):
    """Resolve an (artist, title) suggestion to a real Spotify track dict, or None."""
    market = market or SPOTIFY_MARKET
    try:
        for q in (f'track:"{title}" artist:"{artist}"', f"{title} {artist}"):   # precise, then loosened
            items = _search({"q": q, "type": "track", "limit": 5, "market": market}).get("tracks", {}).get("items", [])
            if items:
                return _track_dict(items[0])
        return None
    except Exception as e:
        print(f"[spotify] search failed for {artist!r} - {title!r}: {e}", file=sys.stderr)
        return None


def spotify_search_query(query, market=None):
    """Free-text search (used for exact-song mode). Returns the top track dict or None."""
    market = market or SPOTIFY_MARKET
    try:
        items = _search({"q": query, "type": "track", "limit": 5, "market": market}).get("tracks", {}).get("items", [])
        return _track_dict(items[0]) if items else None
    except Exception as e:
        print(f"[spotify] query search failed for {query!r}: {e}", file=sys.stderr)
        return None


def spotify_artist_tracks(artist, limit=8, market=None):
    """Search for tracks by a specific artist (used as a fallback in artist mode)."""
    market = market or SPOTIFY_MARKET
    try:
        items = _search({"q": f'artist:"{artist}"', "type": "track", "limit": limit, "market": market}).get("tracks", {}).get("items", [])
        out = [_track_dict(t) for t in items]
        return [t for t in out if _artist_matches(t, artist)]
    except Exception as e:
        print(f"[spotify] artist search failed for {artist!r}: {e}", file=sys.stderr)
        return []


# ------------------- Claude: (state, taste) -> song candidates -------------------
_SYS = (
    "You are a music supervisor for a real-time stress-management app that plays music to regulate the "
    "listener's physiological state. Given the listener's current stress state and their described music "
    "taste, choose real, existing songs that are available on Spotify and that fit BOTH the taste AND the "
    "therapeutic goal for that state. Favor well-known tracks (so they resolve on Spotify), and vary the "
    "artists. Respond with ONLY a JSON array of objects like "
    '[{"artist": "...", "title": "..."}] and absolutely nothing else.'
)

def _anthropic_complete(system, user, max_tokens=700, api_key=None, model=None):
    """Call Claude via the SDK if installed, else a raw HTTP POST. Returns the text body.
    api_key/model override the env defaults so each user can bring their own key."""
    key = api_key or ANTHROPIC_API_KEY
    mdl = model or ANTHROPIC_MODEL
    if not key:
        raise RuntimeError("No Anthropic API key provided")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=mdl, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except ImportError:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": mdl, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=30,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", [])
                       if b.get("type") == "text")


def _parse_song_json(text):
    """Pull a [{artist,title}] list out of the model's reply, tolerating code fences / stray prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j == -1 or j < i:
        print("[ai] no JSON array found in reply", file=sys.stderr)
        return []
    try:
        data = json.loads(text[i:j + 1])
    except Exception as e:
        print(f"[ai] JSON parse failed: {e}", file=sys.stderr)
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        a = str(d.get("artist") or d.get("artist_name") or "").strip()
        t = str(d.get("title") or d.get("track") or d.get("song") or d.get("name") or "").strip()
        if a and t:
            out.append({"artist": a, "title": t})
    return out


def suggest_songs(state, taste, n=CANDIDATES_PER_STATE, api_key=None, model=None):
    """Ask Claude for n real songs that fit the taste and the state's therapeutic goal."""
    intent = STATE_INTENT.get(state, STATE_INTENT["neutral"])
    user = (
        f"Stress state: {state}.\n"
        f"Therapeutic goal for this state: {intent}.\n"
        f"Listener's music taste: {taste}\n\n"
        f"Give {n} songs that clearly fit the taste while serving the goal. JSON array only."
    )
    return _parse_song_json(_anthropic_complete(_SYS, user, api_key=api_key, model=model))


def suggest_artist_songs(state, artist, n=CANDIDATES_PER_STATE, api_key=None, model=None):
    """Ask Claude for songs BY a specific artist that fit the state's goal (artist mode)."""
    intent = STATE_INTENT.get(state, STATE_INTENT["neutral"])
    user = (
        f"Stress state: {state}.\n"
        f"Therapeutic goal for this state: {intent}.\n"
        f'Recommend {n} real songs BY THE ARTIST "{artist}" ONLY (no other artists) that best serve the goal. '
        f'Every object must have "artist" exactly "{artist}". If the artist has few fitting songs, list their '
        f"calmest/most fitting tracks anyway. JSON array of {{\"artist\",\"title\"}} only."
    )
    return _parse_song_json(_anthropic_complete(_SYS, user, api_key=api_key, model=model))


# ------------------- orchestration + per-state played bin -------------------
_lock = threading.Lock()
_queue = {}     # state -> [resolved track dicts not yet served]
_played = {}    # state -> set(track id) already served this cycle (avoid repeats until refill)

def _refill(state, taste, want=SERVE_BATCH, played=None, api_key=None, model=None):
    """Top up the queue for a state: LLM -> Search -> de-duped list of playable tracks."""
    if played is None:
        played = _played.setdefault(state, set())
    cands = suggest_songs(state, taste, api_key=api_key, model=model)
    random.shuffle(cands)
    out, seen = [], set()
    for c in cands:
        tr = spotify_search_track(c["artist"], c["title"])
        if tr and tr["id"] not in seen and tr["id"] not in played:
            out.append(tr); seen.add(tr["id"])
        if len(out) >= want:
            break
    if not out and cands:                 # everything matched was already played -> reset the cycle
        played.clear()
        for c in cands:
            tr = spotify_search_track(c["artist"], c["title"])
            if tr and tr["id"] not in seen:
                out.append(tr); seen.add(tr["id"])
            if len(out) >= want:
                break
    return out


def _refill_artist(state, artist, want=SERVE_BATCH, played=None, api_key=None, model=None):
    """Top up the queue with songs by ONE artist that fit the state (artist mode)."""
    if played is None:
        played = _played.setdefault(state, set())
    out, seen = [], set()
    for c in suggest_artist_songs(state, artist, api_key=api_key, model=model):
        tr = spotify_search_track(artist, c["title"])
        if tr and _artist_matches(tr, artist) and tr["id"] not in seen and tr["id"] not in played:
            out.append(tr); seen.add(tr["id"])
        if len(out) >= want:
            break
    if len(out) < 2:                       # LLM thin / mismatch -> fall back to a direct artist search
        for tr in spotify_artist_tracks(artist, limit=want * 2):
            if tr["id"] not in seen and tr["id"] not in played:
                out.append(tr); seen.add(tr["id"])
            if len(out) >= want:
                break
    if not out:                            # last resort: ignore the played-bin so something plays
        out = spotify_artist_tracks(artist, limit=want)
    return out


def fetch_for_state(state, taste="", api_key=None, model=None, store=None, artist=None, song=None):
    """
    Return ONE playable Spotify track dict for the given classified state, or None.

    Preference modes (mutually exclusive):
      song   : return that exact song (Claude not used; state ignored).
      artist : return songs BY that artist that fit the state.
      taste  : (default) songs that fit `taste` + the state's goal.

    api_key / model : per-user Anthropic credentials (fall back to env if omitted).
    store           : a per-session dict for queue + played-bin isolation across users.
    """
    state = STATE_ALIASES.get((state or "").lower(), (state or "").lower())
    if state not in STATE_INTENT:
        state = "neutral"

    if song and song.strip():                          # exact-song mode -> just resolve it
        return spotify_search_query(song.strip())

    queue_map = store.setdefault("queue", {}) if store is not None else _queue
    played_map = store.setdefault("played", {}) if store is not None else _played
    with _lock:
        q = queue_map.setdefault(state, [])
        played = played_map.setdefault(state, set())
        if not q:
            if artist and artist.strip():
                q.extend(_refill_artist(state, artist.strip(), played=played, api_key=api_key, model=model))
            else:
                q.extend(_refill(state, taste, played=played, api_key=api_key, model=model))
        if not q:
            return None
        track = q.pop(0)
        played.add(track["id"])
        return track


# ------------------- taste capture (startup) -------------------
def get_taste(path=TASTE_PATH, force_ask=False):
    """Ask the user to describe their taste once; reuse it on later runs (delete the file to change)."""
    if not force_ask and os.path.isfile(path):
        saved = open(path).read().strip()
        if saved:
            print(f"Using saved music taste: {saved!r}  (delete {path} to change)")
            return saved
    try:
        taste = input("Describe your music taste (genres, artists, mood): ").strip()
    except EOFError:
        taste = ""
    taste = taste or "calm, melodic, lo-fi and acoustic"
    try:
        open(path, "w").write(taste)
    except Exception:
        pass
    return taste


# ------------------- CLI demo -------------------
def _print_track(state, tr):
    print(f"\n=== {state.upper()} ===")
    if not tr:
        print("  (no track found — check API keys / quota)")
        return
    print(f"  {tr['artist']} — {tr['name']}   [{tr.get('album','')}]")
    print(f"  uri         : {tr['uri']}")
    print(f"  open        : {tr['external_url']}")
    print(f"  preview_url : {tr['preview_url'] or '(none — full playback needs the Web Playback SDK)'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch a Spotify song for a stress state + music taste.")
    ap.add_argument("--state", default=None, help="calm | neutral | elevated (default: show all three)")
    ap.add_argument("--taste", default=None, help="music taste string (default: ask / use saved)")
    ap.add_argument("--reask", action="store_true", help="re-ask for taste even if one is saved")
    args = ap.parse_args()

    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        print("!! Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET first.", file=sys.stderr)
    if not ANTHROPIC_API_KEY:
        print("!! Set ANTHROPIC_API_KEY first.", file=sys.stderr)

    taste = args.taste or get_taste(force_ask=args.reask)
    states = [args.state] if args.state else ["calm", "neutral", "elevated"]
    for st in states:
        st = STATE_ALIASES.get(st.lower(), st.lower())
        _print_track(st, fetch_for_state(st, taste))