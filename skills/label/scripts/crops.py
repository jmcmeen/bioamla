"""Cut a short clip and a spectrogram image around each labeling candidate.

Reads ``candidates.json`` from a run folder and, for each candidate, has bioamla trim
the recording to the event plus some padding (``audio trim``) and draw the clip's
spectrogram (``audio visualize``) with the event's box outlined on it, both into
``<run_dir>/crops/``. Writes ``crops.json``: each candidate's clip, image, where the clip
starts in the recording, and the event's box in the clip's own time (seconds) and
frequency (Hz), so a person reviewing it knows where to look.

Usage (from the workspace root):
    python scripts/crops.py runs/label/<timestamp> --pad 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import BioamlaCommandError, read_json, run, write_json  # noqa: E402


def box_option(box: dict, label: str | None = None) -> str:
    """``box`` as the value of ``audio visualize --box``: ``START,END[,LOW,HIGH][,LABEL]``
    (the band only when both edges are known; a label with a comma is left off)."""
    parts = [f"{box['start']:g}", f"{box['end']:g}"]
    if box.get("low_freq") is not None and box.get("high_freq") is not None:
        parts += [f"{box['low_freq']:g}", f"{box['high_freq']:g}"]
    if label and "," not in str(label):
        parts.append(str(label))
    return ",".join(parts)


def crop(run_folder: str, pad: float = 0.5, kind: str = "stft", root: str = ".") -> dict:
    """Write each candidate's clip and image, and ``crops.json``; returns the summary."""
    folder = Path(run_folder)
    candidates = read_json(Path(root) / folder / "candidates.json")["candidates"]
    crops = []
    for c in candidates:
        start = max(0.0, c["start_time"] - pad)
        end = c["end_time"] + pad
        clip = (folder / "crops" / f"{c['id']}.wav").as_posix()
        image = (folder / "crops" / f"{c['id']}.png").as_posix()
        run("audio", "trim", c["file"], clip, "--start", start, "--end", end, cwd=root)
        box = {
            "start": round(c["start_time"] - start, 4),
            "end": round(c["end_time"] - start, 4),
            "low_freq": c["low_freq"],
            "high_freq": c["high_freq"],
        }
        run("audio", "visualize", clip, "--type", kind, "--output", image,
            "--box", box_option(box, c.get("label")), cwd=root)  # fmt: skip
        crops.append(
            {
                "id": c["id"],
                "clip": clip,
                "image": image,
                "clip_start": round(start, 4),
                "box": box,
            }  # fmt: skip
        )
    write_json(Path(root) / folder / "crops.json", {"pad": pad, "type": kind, "crops": crops})
    return {"run_dir": run_folder, "crops": len(crops), "folder": (folder / "crops").as_posix()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder from candidates.py")
    p.add_argument("--pad", type=float, default=0.5, help="Seconds of context each side")
    p.add_argument("--type", dest="kind", choices=["stft", "mel"], default="stft",
                   help="Spectrogram type")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    args = p.parse_args(argv)
    try:
        summary = crop(args.run_dir, args.pad, args.kind, args.root)
    except BioamlaCommandError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    except FileNotFoundError as e:
        print(json.dumps({"error": f"No candidates in {args.run_dir}: {e}"}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
