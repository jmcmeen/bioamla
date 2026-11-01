"""Cut a short clip and a spectrogram image around each labeling candidate.

Reads ``candidates.json`` from a run folder and, for each candidate, has bioamla trim
the recording to the event plus some padding (``audio trim``) and draw the clip's
spectrogram (``audio visualize``) with the event's box outlined on it, both into
``<run_dir>/crops/``. Writes ``crops.json``: each candidate's clip, image, where the clip
starts in the recording, and the event's box in the clip's own time (seconds) and
frequency (Hz), so a person reviewing it knows where to look. A box with only one band
edge is drawn full height; one with no length or width isn't drawn (``box_drawn``).

Usage (from the workspace root):
    python scripts/crops.py runs/label/<timestamp> --pad 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import BioamlaCommandError, add_remote_option, read_json, run, save_json, use_remote  # noqa: E402


def _num(value: float) -> str:
    return repr(float(value))  # full precision: the drawn box matches crops.json


def drawable(box: dict) -> bool:
    """Whether ``audio visualize`` can outline ``box``: it has length, and a band (when
    both edges are known) has width."""
    if not box["end"] > box["start"]:
        return False
    low, high = box.get("low_freq"), box.get("high_freq")
    return low is None or high is None or high > low


def box_option(box: dict, label: str | None = None) -> str:
    """``box`` as the value of ``audio visualize --box``: ``START,END,LOW,HIGH[,LABEL]``,
    or ``START,END`` / ``START,END,,,LABEL`` without both band edges (then it spans the
    full height). A label may contain commas: all after the fourth comma is the label."""
    low, high = box.get("low_freq"), box.get("high_freq")
    parts = [_num(box["start"]), _num(box["end"])]
    if low is not None and high is not None:
        parts += [_num(low), _num(high)]
    elif label:
        parts += ["", ""]
    if label:
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
        # A box with no length or no width can't be outlined; the image goes without it.
        boxed = drawable(box)
        outline = ["--box", box_option(box, c.get("label"))] if boxed else []
        run("audio", "visualize", clip, "--type", kind, "--output", image, *outline, cwd=root)
        crops.append(
            {
                "id": c["id"],
                "clip": clip,
                "image": image,
                "clip_start": round(start, 4),
                "box": box,
                "box_drawn": boxed,
            }
        )
    save_json((folder / "crops.json").as_posix(), {"pad": pad, "type": kind, "crops": crops}, root)
    return {"run_dir": run_folder, "crops": len(crops), "folder": (folder / "crops").as_posix()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder from candidates.py")
    p.add_argument("--pad", type=float, default=0.5, help="Seconds of context each side")
    p.add_argument("--type", dest="kind", choices=["stft", "mel"], default="stft",
                   help="Spectrogram type")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    args = p.parse_args(argv)
    use_remote(args.remote)
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
