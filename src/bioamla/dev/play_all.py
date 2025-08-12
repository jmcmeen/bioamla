#!/usr/bin/env python3
"""
WAV File Player
Plays WAV files from a directory with options for verbose output
and recursive directory searching.
"""

import sys
import wave
import time
import argparse
from pathlib import Path

try:
    import pygame
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
except ImportError:
    print("Error: pygame is required for audio playback.")
    print("Install it with: pip install pygame")
    sys.exit(1)

class WAVPlayer:
    def __init__(self):
        self.current_sound = None
        self.is_playing = False
        self.stop_requested = False
        
    def get_wav_duration(self, wav_path):
        """
        Get the duration of a WAV file in seconds.
        
        Args:
            wav_path (Path): Path to the WAV file
            
        Returns:
            float: Duration in seconds, or 0 if error
        """
        try:
            with wave.open(str(wav_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                if sample_rate > 0:
                    return frames / sample_rate
        except Exception:
            pass
        return 0.0
    
    def format_duration(self, seconds):
        """
        Format duration in MM:SS.mmm format.
        
        Args:
            seconds (float): Duration in seconds
            
        Returns:
            str: Formatted duration string
        """
        if seconds <= 0:
            return "00:00.000"
        
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"
    
    def play_wav_file(self, wav_path, verbose=False):
        """
        Play a single WAV file.
        
        Args:
            wav_path (Path): Path to the WAV file
            verbose (bool): Whether to display verbose information
            
        Returns:
            bool: True if played successfully, False otherwise
        """
        try:
            # Get file duration for verbose output
            duration = 0.0
            if verbose:
                duration = self.get_wav_duration(wav_path)
            
            # Display file information
            if verbose:
                duration_str = self.format_duration(duration)
                print(f"🎵 Playing: {wav_path.name}")
                print(f"   Duration: {duration_str}")
                print(f"   Path: {wav_path}")
            else:
                print(f"Playing: {wav_path.name}")
            
            # Load and play the sound
            self.current_sound = pygame.mixer.Sound(str(wav_path))
            self.is_playing = True
            self.stop_requested = False
            
            # Start playback
            channel = self.current_sound.play()
            
            # Wait for playback to complete or stop request
            while channel.get_busy() and not self.stop_requested:
                time.sleep(0.1)
                
                # Handle keyboard interrupt gracefully
                try:
                    pass
                except KeyboardInterrupt:
                    self.stop_requested = True
                    break
            
            # Stop the sound if still playing
            if channel.get_busy():
                channel.stop()
            
            self.is_playing = False
            
            if self.stop_requested:
                return False
            
            if verbose:
                print("   ✅ Playback completed\n")
            
            return True
            
        except pygame.error as e:
            print(f"   ❌ Pygame error: {str(e)}")
            return False
        except Exception as e:
            print(f"   ❌ Error playing {wav_path.name}: {str(e)}")
            return False
    
    def stop_playback(self):
        """Stop current playback."""
        self.stop_requested = True
        if self.current_sound and self.is_playing:
            pygame.mixer.stop()

def find_wav_files(input_dir, recursive=False):
    """
    Find WAV files in the input directory.
    
    Args:
        input_dir (Path): Input directory path
        recursive (bool): Whether to search recursively
        
    Returns:
        list: Sorted list of Path objects for WAV files
    """
    wav_files = []
    
    if recursive:
        # Recursive search
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.wave']:
                wav_files.append(file_path)
    else:
        # Non-recursive search (current directory only)
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.wave']:
                wav_files.append(file_path)
    
    return sorted(wav_files)

def print_controls():
    """Print playback control instructions."""
    print("\n" + "="*60)
    print("PLAYBACK CONTROLS:")
    print("="*60)
    print("• Press Ctrl+C to skip current file")
    print("• Press Ctrl+C twice quickly to exit program")
    print("• Each file plays automatically after the previous one")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Play WAV files from a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wav_player.py /path/to/audio/files
  python wav_player.py /path/to/audio/files -v
  python wav_player.py /path/to/audio/files -r -v
  python wav_player.py /path/to/audio/files --recursive --verbose
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory containing WAV files"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (show filename and duration)"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search subdirectories"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between files in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    # Find WAV files
    print(f"Searching for WAV files in: {input_dir}")
    if args.recursive:
        print("(Recursive search enabled)")
    
    wav_files = find_wav_files(input_dir, args.recursive)
    
    if not wav_files:
        search_type = "recursively" if args.recursive else "in directory"
        print(f"No WAV files found {search_type}: {input_dir}")
        return
    
    # Display found files
    print(f"Found {len(wav_files)} WAV file{'s' if len(wav_files) != 1 else ''}")
    
    if args.verbose:
        print("\nFiles to be played:")
        for i, wav_file in enumerate(wav_files, 1):
            rel_path = wav_file.relative_to(input_dir) if args.recursive else wav_file.name
            print(f"  {i:2d}. {rel_path}")
    
    # Show controls
    print_controls()
    
    # Initialize player
    player = WAVPlayer()
    
    # Play files
    played_count = 0
    skipped_count = 0
    error_count = 0
    consecutive_interrupts = 0
    
    try:
        for i, wav_file in enumerate(wav_files, 1):
            print(f"[{i}/{len(wav_files)}] ", end="")
            
            try:
                success = player.play_wav_file(wav_file, args.verbose)
                
                if success:
                    played_count += 1
                    consecutive_interrupts = 0
                    
                    # Add delay between files if specified
                    if args.delay > 0 and i < len(wav_files):
                        time.sleep(args.delay)
                else:
                    skipped_count += 1
                    print("   ⏭️  Skipped\n")
                    
            except KeyboardInterrupt:
                consecutive_interrupts += 1
                skipped_count += 1
                print("   ⏭️  Skipped (Ctrl+C)\n")
                
                # Exit if two consecutive interrupts
                if consecutive_interrupts >= 2:
                    print("Exiting due to consecutive interrupts...")
                    break
                
                # Reset stop flag for next file
                player.stop_requested = False
                
            except Exception as e:
                error_count += 1
                print(f"   ❌ Unexpected error: {str(e)}\n")
    
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    
    finally:
        # Cleanup
        player.stop_playback()
        pygame.mixer.quit()
        
        # Final summary
        print("\n" + "="*60)
        print("PLAYBACK SUMMARY:")
        print("="*60)
        print(f"Total files found: {len(wav_files)}")
        print(f"Successfully played: {played_count}")
        if skipped_count > 0:
            print(f"Skipped: {skipped_count}")
        if error_count > 0:
            print(f"Errors: {error_count}")
        print("="*60)

if __name__ == "__main__":
    main()