<div align="center">

# AVTR-1

[![Project Page](https://img.shields.io/badge/Project%20Page-AVTR--1-8A2BE2)](https://avaturn-live.github.io/avtr-1-projectpage/)
[![Managed API](https://img.shields.io/badge/Managed%20API-avaturn.live-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAADZc7J/AAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAd0SU1FB+oFFA8fAzOfxeUAAAOsSURBVEjHfZVPSFxXFMa/+wYpJGqofxCrhBKcpBMTozGRdFNCSDYtpRFCFu0mWYRgNor54za7JILgspa6SALZl9JCVgndFGqqVWmpmkKjUWGCCiGOzrx376+L++Y5o5Oe1TDvfueec+53vs+oJDAKjJVo1ef6TCf0keokbehfzeg3/WCWJVJyBlUKAkmig+/ZoFJkGaN75+RueEpiP8PkAbBEOJzDgcMSYQHYYoQaf3ovPMMkAKFHRcnVMdoRAjBFelcKUhI9LHmwgwS8RS5JAg7rbAgs0lOSgkCijRUoRDuHs9ylnQCRYSjO7aMQASsciWeBIaCaaY+z5AiBpzQgTnKDPk4hGnkKhORwxRtm2EeA8eWPAAXLAw4hWrlKLZ1MJDOYoYMa+jmISDNKwZczLBEIQyc57KbrRZzhJhcQ4jqL5LE4tnnFFYQ4Rz9diF62HZYCGT+DMSAaRdwD4DHiBEIcIMMxqlH8zyMA7iHu+zbGJUOL+yOot0qbek0IvdVhNep3LehHzWpVTq06ri+U1mm90Z+ql/SpVvWSlOFtcFT0AXYTMUAEDCKeVSTic8QdIOIW4h1Y4Jp4CEQRTZwF5qjiYkydEIvDxb8scAnDHHCeBkLfxBMxCaGF24gXfEOKl7iYt6VhccxTxWUmEQNA6IBZse5ryVJHK2KwhIflEQFDiCaaWcajWNPOx28Rn5DDebJUrGGDDOK7kktKFjOQZJSXVHndjaQPhCRXtodrvphlmmlCDP1vC4OIFurIFltYF7N+HAOISS5Txfx7h/gPKb7mBeI2fvRMiSdAFNLAeWAOwyXA7nnGCLhIFX8DZ2ki8iU9FNcA+w5xiwi4g3hesYVnyRsNIDZ9D32GFveXqbW0mWb9KmlN7WrUhBb0k2b1WoGadVxfKq1uvdG8amV0WmtaICW3FnRKYhyI7vsVAR4ly1TNMTIcSJbpcbJMo76BMf8OGQrYbdeL6KKfcwhxhVds47DkWeQ6QlzgJmcQvWw6LDm6MF7QhoGwwChpxEH6qaGDmaT7CTqp5SqtiEM8wEIBGJFIeUnbx4x/aJdIWiPiFH3c4CSiIZE0W6TENNUEmKKoHmF1R1RDYIkhMoiAdu6SLaFXIQJWaSsxmFjWF8GGtkzWc2yVyXrsDUtlsp6kOMxUXIAjpk4R7GDHWCbJvM+bahiJr/RmVrQ2l1hbnmH2V4CXmGs3Y2QrUnGDcTp2m6upaO8t+ko96tDH+lDSulY0rV/0s3m9197/A6bOU5Zxdgd9AAAAAElFTkSuQmCC)](https://avaturn.live)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange?logo=huggingface)](https://huggingface.co/avaturn-live/avtr-1)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAADZc7J/AAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAd0SU1FB+oFFA8fAzOfxeUAAAOsSURBVEjHfZVPSFxXFMa/+wYpJGqofxCrhBKcpBMTozGRdFNCSDYtpRFCFu0mWYRgNor54za7JILgspa6SALZl9JCVgndFGqqVWmpmkKjUWGCCiGOzrx376+L++Y5o5Oe1TDvfueec+53vs+oJDAKjJVo1ef6TCf0keokbehfzeg3/WCWJVJyBlUKAkmig+/ZoFJkGaN75+RueEpiP8PkAbBEOJzDgcMSYQHYYoQaf3ovPMMkAKFHRcnVMdoRAjBFelcKUhI9LHmwgwS8RS5JAg7rbAgs0lOSgkCijRUoRDuHs9ylnQCRYSjO7aMQASsciWeBIaCaaY+z5AiBpzQgTnKDPk4hGnkKhORwxRtm2EeA8eWPAAXLAw4hWrlKLZ1MJDOYoYMa+jmISDNKwZczLBEIQyc57KbrRZzhJhcQ4jqL5LE4tnnFFYQ4Rz9diF62HZYCGT+DMSAaRdwD4DHiBEIcIMMxqlH8zyMA7iHu+zbGJUOL+yOot0qbek0IvdVhNep3LehHzWpVTq06ri+U1mm90Z+ql/SpVvWSlOFtcFT0AXYTMUAEDCKeVSTic8QdIOIW4h1Y4Jp4CEQRTZwF5qjiYkydEIvDxb8scAnDHHCeBkLfxBMxCaGF24gXfEOKl7iYt6VhccxTxWUmEQNA6IBZse5ryVJHK2KwhIflEQFDiCaaWcajWNPOx28Rn5DDebJUrGGDDOK7kktKFjOQZJSXVHndjaQPhCRXtodrvphlmmlCDP1vC4OIFurIFltYF7N+HAOISS5Txfx7h/gPKb7mBeI2fvRMiSdAFNLAeWAOwyXA7nnGCLhIFX8DZ2ki8iU9FNcA+w5xiwi4g3hesYVnyRsNIDZ9D32GFveXqbW0mWb9KmlN7WrUhBb0k2b1WoGadVxfKq1uvdG8amV0WmtaICW3FnRKYhyI7vsVAR4ly1TNMTIcSJbpcbJMo76BMf8OGQrYbdeL6KKfcwhxhVds47DkWeQ6QlzgJmcQvWw6LDm6MF7QhoGwwChpxEH6qaGDmaT7CTqp5SqtiEM8wEIBGJFIeUnbx4x/aJdIWiPiFH3c4CSiIZE0W6TENNUEmKKoHmF1R1RDYIkhMoiAdu6SLaFXIQJWaSsxmFjWF8GGtkzWc2yVyXrsDUtlsp6kOMxUXIAjpk4R7GDHWCbJvM+bahiJr/RmVrQ2l1hbnmH2V4CXmGs3Y2QrUnGDcTp2m6upaO8t+ko96tDH+lDSulY0rV/0s3m9197/A6bOU5Zxdgd9AAAAAElFTkSuQmCC)](https://avaturn.live/demo)

</div>

**AVTR-1** is a flow-matching-based autoregressive model for live dialogue. Given a portrait image and dual-stream audio, it renders lip-synced speech and active listening at 25 fps on a single GPU. Built for production deployment: model weights, TensorRT-accelerated inference, and the live-session backend - available as an API or fully self-hosted

<div align="center">
  <video src="https://github.com/user-attachments/assets/5e4f96af-973a-4aa4-a8be-e0ce6f44a5d1" muted=false width="50%"></video>
</div>

---

## 📑 What's included

- [x] Model weights
- [x] Inference code
- [x] Interactive streaming demo
- [ ] Technical report (Coming soon)
- [ ] Production-ready back-end (Coming soon)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Performance](#2-performance)
3. [Troubleshooting](#troubleshooting)

---

## 1. 🚀 Quick Start

### Prerequisites

- Linux
- NVIDIA GPU (Ampere or later recommended)
- CUDA 12.x + TensorRT 10.x
- [pixi](https://prefix.dev/) — `curl -fsSL https://pixi.sh/install.sh | sh`

### Install

```bash
git clone https://github.com/avaturn-live/avtr-1.git
cd avtr-1
pixi install
```

### Set storage path (Optional)

```bash
export AVTR1_LOCAL_STORAGE=/path/to/avtr1_storage
```

All downloaded weights and built engines go here. Defaults to `<project_root>/artifacts/` (the repo checkout, not the caller's working directory) when unset.

### Download weights

```bash
pixi run download
```

First run will prompt for a HuggingFace login via `hf auth login`
(automatically invoked as a dependency of `download`).

### Build TRT engines

Weights are pulled from two public HF repos by the previous step:
AVTR-1 weights from [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1)
and LivePortrait weights repackaged as ONNX graphs from
[digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
TRT engines are compute-capability specific and built locally — run the scripts
below once per machine; outputs land under `$AVTR1_LOCAL_STORAGE`.

```bash
# Build everything at once
pixi run build-trt-engines

# Or individually
pixi run build-trt-engines-avtr1
pixi run build-trt-engines-renderer
pixi run build-trt-engines-hubert
```

### Run interactive demo
```bash
pixi run interactive-demo
```


### Run offline generation

**Single speaker.** Avatar lip-syncs the given audio track.

```bash
pixi run generate_offline --speech example/speaker_1.ogg

# with a custom avatar and background:
pixi run generate_offline --speech example/speaker_1.ogg --avatar maria --bg minimal_office
```

**Two-speaker dialogue.** Avatar voices `--speech` and reacts (active listening) to the peer audio on `--listen`. Run twice with the tracks swapped to render both sides of the conversation.

```bash
# avatar = speaker 1 (elena)
pixi run generate_offline --speech example/speaker_1.ogg --listen example/speaker_2.ogg --avatar elena  --out elena.mp4
# avatar = speaker 2 (marcus)
pixi run generate_offline --speech example/speaker_2.ogg --listen example/speaker_1.ogg --avatar marcus --out marcus.mp4

# stitch both sides into a single side-by-side video:
ffmpeg -i elena.mp4 -i marcus.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2[v];[0:a][1:a]amix=inputs=2[a]" \
  -map "[v]" -map "[a]" dialogue.mp4
```

**Silence / idle motion.** No audio — renders idle micro-motion for the given duration.

```bash
pixi run generate_offline --duration 10
```

Available avatars are the filenames (without `.png`) inside
`$AVTR1_LOCAL_STORAGE/v1/avatars_artifacts/reference_frames/` after downloading.
---

## 2. Performance

### Per-chunk latency

AVTR-1 generates motion in 5-frame chunks end-to-end. At 25 fps that's 200 ms
of output per chunk, so any GPU under that line runs in real-time.

| GPU         | Latency / 5-frame chunk | Real-time factor |
| ----------- | ----------------------- | ---------------- |
| L40         | 84 ms                   | 2.4×             |
| A100        | 91 ms                   | 2.2×             |
| RTX 4060 Ti | 166 ms                  | 1.2×             |
| RTX 3070    | 181 ms                  | 1.1×             |
| L4          | 202 ms                  | 0.99×            |
| RTX 3060 Ti | 206 ms                  | 0.97×            |
| RTX 4060    | 232 ms                  | 0.86×            |

Real-time factor = 200 ms / latency. ≥ 1.0× means the GPU keeps up with 25 fps.

---

## Troubleshooting

<details>
<summary><b>TURN server setup</b> (optional)</summary>

ICE tries direct UDP first (host candidates + STUN-reflexive
candidates from a public STUN server) and only needs a TURN relay when the
network in between can't pass UDP between browser and streamer — typical
when the streamer lives on a cloud VM whose security group blocks inbound
UDP, or when one peer is behind symmetric NAT.

If direct UDP works for your setup you can skip this section entirely. The
browser's connectivity card after the engine dropdown tells you which path
ICE actually picked, and the same UI links back here when the verdict is
"only TURN works" or "nothing worked".

The project is wired for **Cloudflare's Realtime TURN**. The free tier is
generous enough for development; no credit card required.

**1. Create a TURN application on Cloudflare**

- Sign in to [dash.cloudflare.com](https://dash.cloudflare.com).
- Navigate to **Realtime → TURN Server**.
- Click **Create TURN App**, give it a name (e.g. `avtr1-dev`), and submit.

**2. Copy the two credential values**

On the application's detail page you'll see:

- **Turn Key ID** — short identifier (looks like a UUID without dashes).
- **API Token** — long secret shown only once at creation. Save it before
  navigating away.

**3. Put them in `.env`**

```dotenv
CLOUDFLARE_TURN_KEY_ID="<Turn Key ID>"
CLOUDFLARE_TURN_KEY_TOKEN="<API Token>"
```

That's it. On the next `/ice-servers` request the streamer mints a fresh,
short-lived TURN credential per session via Cloudflare's
`/v1/turn/keys/{kid}/credentials/generate` endpoint — the long-lived API
token never leaves the server. You can verify it picked up the keys by
watching the streamer log for `ice: using Cloudflare TURN` on the first
browser request.

The browser-side connectivity probe (the small status card under the
controls) tells you which ICE path actually wins:

- ✓ **host** — the browser saw its own local interface; always present.
- ✓ **server-reflexive via STUN** — the browser learned its public IP via
  STUN; doesn't prove the streamer is reachable on UDP from the browser.
- ✓ **relay via TURN** — the browser successfully allocated a Cloudflare
  TURN relay; required when direct UDP can't traverse the network in
  between.

If the relay check fails while TURN is configured the most likely cause is
wrong credentials — re-check that you copied the full **API Token** (not
the Key ID twice) into `CLOUDFLARE_TURN_KEY_TOKEN`.

**Alternatives.** Anything that speaks the standard TURN protocol works.
Set `TURN_URL` (and optionally `TURN_USERNAME` / `TURN_CREDENTIAL`) instead
of the Cloudflare variables and `resolve_ice_servers()` will use it
verbatim — e.g. a self-hosted [coturn](https://github.com/coturn/coturn) on
a small VM. STUN-only also works *if* you can open the appropriate UDP
port range inbound on whatever firewall sits in front of the streamer.

</details>

---

## License

This repository contains three separately licensed components:

- **`scripts/`** — build and demo tooling, released under the **AVTR-1
  Community License** ([LICENSE-MODEL.md](LICENSE-MODEL.md)). Permits
  commercial use by entities under USD 10M annual revenue; entities at or
  above that threshold need a commercial agreement. The same license governs
  the AVTR-1 weights distributed at
  [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1).
- **`src/avtr1_renderer/`** — Avaturn Renderer (inference pipeline), released
  under the **PolyForm Noncommercial License 1.0.0** with a Required Notice
  ([LICENSE-RENDERER.md](LICENSE-RENDERER.md)). **Noncommercial use only**,
  regardless of revenue; any commercial use needs a separate Renderer
  Commercial License.
- **`src/avaturn_live_streamer/`** — Avaturn Streamer (orchestration
  backend), released under the **PolyForm Noncommercial License 1.0.0** with
  a Required Notice and patent reservation
  ([LICENSE-STREAMER.md](LICENSE-STREAMER.md),
  [PATENTS.md](PATENTS.md)). **Noncommercial use only**, regardless of
  revenue; any commercial use needs a separate Streamer Commercial License.

See [LICENSE.md](LICENSE.md) for the full component map and the consequences
of the multi-license structure. In any conflict between this summary and the
underlying license files, the license files control.

### Non-commercial dependency

The pipeline uses InsightFace's pretrained SCRFD detector and 2D106 landmark
model, which are licensed for **non-commercial research use only**. To use
AVTR-1 commercially you must either obtain a commercial license from
InsightFace (deepinsight@gmail.com) or replace these models with
permissively-licensed alternatives (e.g., MediaPipe). See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full picture.

**Commercial inquiries:** hello@avaturn.me
