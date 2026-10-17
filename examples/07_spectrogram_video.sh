#!/usr/bin/env bash
#
# Workflow 7: spectrogram and waveform videos
# -------------------------------------------
# Turn a recording into videos you can play, share, or drop into a talk: the
# spectrogram (or waveform) moves in sync with the audio, which is muxed in.
# Renders the same recording in each of the available styles and in the
# frame shapes the usual destinations want (16:9, 9:16, 1:1).
#
# Needs: a recording (set RECORDING) and the ffmpeg command on PATH (the
# encoder; `brew install ffmpeg` / `apt install ffmpeg`). Rendering is fast —
# roughly real time at 1080p on a laptop — and runs on the CPU.
#
# Every `bioamla audio video` flag used here:
#   -t   mel | stft | waveform      what to draw (default mel)
#   -m   scroll | page              picture moves past a fixed playhead, or the
#                                   playhead sweeps across a held page (default scroll)
#   -w   seconds across the frame; 0 = the whole recording on one page
#   -s/-e  start/end seconds, to render just a stretch of a long file
#   --size WxH, --cmap, --playhead 0..1, --no-waveform, --no-axes, --no-audio

set -euo pipefail

OUT=./out/07_spectrogram_video
mkdir -p "$OUT"

RECORDING="${RECORDING:-$OUT/recording.wav}"   # <-- point at your file
# Optional: only render this stretch (seconds). Leave empty for the whole file.
START="${START:-}"
END="${END:-}"

if [[ ! -f "$RECORDING" ]]; then
  echo "Set RECORDING to a real audio file (WAV, FLAC, MP3, ...)." >&2
  exit 1
fi
command -v ffmpeg >/dev/null || { echo "ffmpeg is required; install it and retry." >&2; exit 1; }

CLIP=()
[[ -n "$START" ]] && CLIP+=(-s "$START")
[[ -n "$END" ]] && CLIP+=(-e "$END")

# 1. The default: mel spectrogram scrolling past a centred playhead, with a
#    waveform strip underneath. Four seconds of audio span the frame.
bioamla audio video "$RECORDING" "${CLIP[@]}" -o "$OUT/1_mel_scroll.mp4"

# 2. Linear-frequency STFT, held still in 8-second pages while the playhead
#    sweeps across. Easier to read fine harmonic detail than a scrolling view.
bioamla audio video "$RECORDING" "${CLIP[@]}" -o "$OUT/2_stft_pages.mp4" \
  -t stft -m page -w 8 --cmap viridis

# 3. The whole recording on one page: the classic "player" view, good for short
#    clips. (-w 0 means the page is the entire recording.)
bioamla audio video "$RECORDING" "${CLIP[@]}" -o "$OUT/3_mel_whole.mp4" -m page -w 0

# 4. Waveform only, whole recording. No spectrogram is computed at all.
bioamla audio video "$RECORDING" "${CLIP[@]}" -o "$OUT/4_waveform_whole.mp4" \
  -t waveform -m page -w 0

# 5. Portrait 9:16 for phones (Reels, Shorts, TikTok). A tall frame holds fewer
#    seconds comfortably, so narrow the window to 3 s. Playhead at the right
#    edge shows only what has already played.
bioamla audio video "$RECORDING" "${CLIP[@]}" -o "$OUT/5_portrait.mp4" \
  --size 1080x1920 -w 3 --playhead 1

# 6. Square 1:1 for feed posts, borderless: no axes, no waveform strip.
bioamla audio video "$RECORDING" "${CLIP[@]}" -o "$OUT/6_square_clean.mp4" \
  --size 1080x1080 --no-axes --no-waveform --cmap inferno

echo "Done. Videos are under $OUT/."
