# Recorder metadata

Field recorders write deployment details into the WAV container alongside the audio:
when the file started, where the unit was, which unit it was, and how it was set.
bioamla reads these with [riffy](https://github.com/jmcmeen/riffy), a pure-Python
RIFF parser, so no FFmpeg, libsndfile or torch is needed and nothing is decoded.

| Standard | Written by | Where it lives |
| --- | --- | --- |
| GUANO | Wildlife Acoustics (Song Meter), bat detectors, many others | `guan` chunk |
| AudioMoth comment | Open Acoustic Devices AudioMoth | RIFF `INFO` `ICMT`/`IART` |
| WAMD | older Wildlife Acoustics firmware | `wamd` chunk |
| `bext` | broadcast and studio tools | `bext` chunk |
| RIFF `INFO` | general tagging | `LIST`/`INFO` |

Only WAV files carry this; FLAC, MP3 and friends are skipped everywhere below.

## The flat view

`bioamla.audio.get_recording_metadata` merges whatever standards a file carries into
one `RecordingMetadata`:

```python
from bioamla.audio import get_recording_metadata

meta = get_recording_metadata("FROGNN-1_20230306_182817.wav")
meta.recorded_at    # datetime(2023, 3, 6, 18, 28, 17, tzinfo=UTC-05:00)
meta.latitude, meta.longitude   # 36.55876, -82.21265
meta.device_make    # "Wildlife Acoustics, Inc."
meta.device_model   # "Song Meter Micro"
meta.device_id      # "SMM08024"
meta.gain           # "12"
meta.sources        # ("guano", "wamd")
meta.is_empty       # False
meta.to_dict()      # CSV-ready row, ISO timestamp, sources joined with "+"
```

Fields: `recorded_at`, `latitude`, `longitude`, `device_make`, `device_model`,
`device_id`, `gain`, `temperature_c`, `battery_voltage`, `sources`. Each is `None`
when no standard provides it. When several do, GUANO wins, then AudioMoth, then
`bext`, then WAMD. Gain also comes from Wildlife Acoustics' `WA|Song Meter|Audio
settings` GUANO field.

If no standard carries a timestamp but the file is named `PREFIX_YYYYMMDD_HHMMSS.wav`
(Song Meter and AudioMoth both do this), `recorded_at` is taken from the name as a
naive datetime and `"filename"` is added to `sources`.

Need every field, not just the flat ones? `dump_recording_metadata(path)` returns a
JSON-serializable dict with one entry per standard, including custom GUANO fields,
`bext` coding history and iXML.

## From the command line

```bash
bioamla audio info recording.wav --metadata     # header, then the flat view if any
bioamla audio metadata recording.wav            # flat view
bioamla audio metadata recording.wav --json     # everything, per standard
```

For chunk-level inspection or editing (fixing a wrong `Loc Position`, say) use the
`riffy` command that ships with riffy: `riffy inspect`, `riffy diff`, `riffy set`,
`riffy chunk`. bioamla reads; it does not write recorder metadata.

## Whole deployments

`bioamla batch audio validate` is a pre-flight step: it checks that every WAV under a
directory is well-formed PCM from the header alone and exits 1 if any is not, writing
`validation_report.csv` (`file_name`, `valid`, `error`) when `--output-dir` is given.
Non-WAV files are reported as `skipped`.

`bioamla batch audio metadata` writes `recorder_metadata.csv` with the flat fields per
WAV, or, with `--input-file meta.csv`, merges those columns into an existing manifest:

```bash
bioamla batch audio validate --input-dir ./raw --recursive
bioamla batch audio metadata --input-dir ./raw --recursive --output-dir ./out
bioamla batch audio metadata --input-file ./dataset/metadata.csv
```

## In datasets

`bioamla.datasets.add_recording_metadata(dataset_dir)` fills the recorder columns of a
dataset's `metadata.csv` in place (rows that already have `metadata_sources` are kept
unless `overwrite=True`). Once those columns exist, `bioamla dataset manifest` records
the recording span, the distinct recorders and the distinct sites in `dataset.json`,
and the dataset card gains a "Recording" section:

```json
"recorded_from": "2023-03-06T18:27:37-05:00",
"recorded_to": "2023-03-06T18:28:17-05:00",
"devices": [{"device_model": "Song Meter Micro", "device_id": "SMM08024", "files": 2}],
"sites": [{"latitude": 36.56, "longitude": -82.21, "files": 2}]
```

Site coordinates in the manifest are rounded to 0.01 degrees (about 1 km) so a public
manifest does not pinpoint a recorder. The per-file coordinates in `metadata.csv` are
not rounded; strip or round those columns yourself before publishing a dataset whose
exact recorder locations are sensitive.

See [`examples/08_raw_deployment_metadata.sh`](https://github.com/jmcmeen/bioamla/blob/main/examples/08_raw_deployment_metadata.sh)
for the validate, extract and manifest steps wired together.
