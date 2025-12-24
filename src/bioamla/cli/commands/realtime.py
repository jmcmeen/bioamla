"""Real-time audio processing commands."""

import click


@click.group()
def realtime():
    """Real-time audio processing commands."""
    pass


@realtime.command("devices")
def realtime_devices():
    """List available audio input devices."""
    from bioamla.core.realtime import list_audio_devices

    devices = list_audio_devices()

    click.echo("Available Audio Input Devices:")
    for device in devices:
        click.echo(f"  [{device['index']}] {device['name']}")
        click.echo(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']}")


@realtime.command("test")
@click.option("--duration", "-d", type=float, default=3.0, help="Recording duration in seconds")
@click.option("--device", type=int, help="Device index")
@click.option("--output", "-o", help="Output file to save recording")
def realtime_test(duration: float, device: int, output: str):
    """Test audio recording from microphone."""
    from bioamla.core.realtime import test_recording

    click.echo(f"Recording for {duration} seconds...")
    audio = test_recording(duration=duration, device=device)

    click.echo(f"Recorded {len(audio)} samples")
    click.echo(f"Max amplitude: {audio.max():.4f}")
    click.echo(f"RMS: {(audio**2).mean() ** 0.5:.4f}")

    if output:
        import soundfile as sf

        sf.write(output, audio, 16000)
        click.echo(f"Saved recording to: {output}")
