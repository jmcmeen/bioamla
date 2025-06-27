I'll create a Python script that plays WAV files from a directory with options for verbose output and recursive searching. This script will use the `pygame` library for audio playback.I've created a comprehensive WAV file player script with all the requested features. Here's what the script provides:

## Key Features:

**Playback Options:**
- Plays WAV files sequentially from a specified directory
- **Verbose mode** (`-v`): Shows filename, duration, and file path
- **Recursive mode** (`-r`): Searches subdirectories for WAV files
- Configurable delay between files

**User Controls:**
- Press Ctrl+C once to skip current file
- Press Ctrl+C twice quickly to exit the program
- Automatic progression through all files

**File Information:**
- In verbose mode, displays duration in MM:SS.mmm format
- Shows relative file paths when in recursive mode
- Provides playback progress (current file number / total files)

## Usage Examples:

**Basic usage:**
```bash
python wav_player.py /path/to/audio/files
```

**Verbose mode (show details):**
```bash
python wav_player.py /path/to/audio/files -v
```

**Recursive search in subdirectories:**
```bash
python wav_player.py /path/to/audio/files -r
```

**Both verbose and recursive:**
```bash
python wav_player.py /path/to/audio/files -v -r
```

**With custom delay between files:**
```bash
python wav_player.py /path/to/audio/files --delay 2.0
```

## Sample Output:

**Normal mode:**
```
Playing: audio_sample.wav
```

**Verbose mode:**
```
🎵 Playing: audio_sample.wav
   Duration: 02:35.420
   Path: /path/to/files/music/audio_sample.wav
   ✅ Playback completed
```

## Requirements:
Install pygame for audio playback:
```bash
pip install pygame
```

## Features:

**Smart File Detection:**
- Finds both `.wav` and `.wave` files
- Case-insensitive file extension matching
- Sorted file list for consistent playback order

**Error Handling:**
- Graceful handling of corrupted audio files
- Clear error messages for various failure modes
- Continues playing remaining files even if some fail

**User Experience:**
- Progress indicator showing current file number
- Clean summary at the end showing statistics
- Intuitive keyboard controls
- Proper cleanup of audio resources

**Playback Control:**
- Immediate response to skip requests
- Clean audio channel management
- Automatic progression through playlist

The script is robust and handles various edge cases like missing files, corrupted audio, and user interruptions gracefully. It provides a complete audio playback experience with professional-grade error handling and user controls.