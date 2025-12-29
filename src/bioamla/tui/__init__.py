"""Terminal User Interface (TUI) components for bioamla."""

from bioamla.tui.magpy import MagpyLite, run_magpy


def main() -> None:
    """Main entry point for magpy-lite TUI application."""
    import sys

    # Get starting directory from command line args if provided
    start_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_magpy(start_dir=start_dir)


__all__ = ["MagpyLite", "run_magpy", "main"]
