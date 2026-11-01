"""Rich-based console output for the bioamla CLI.

This is the single output path for the command layer: progress bars, status
spinners, tables/panels, and the line/value helpers below. Routing everything
through the shared :data:`console` keeps colour consistent **and** safe — Rich
auto-detects non-TTY output and ``NO_COLOR`` and strips styling, so piped or
redirected output stays plain text.

Two rules keep that guarantee intact:

* **Never wrap data lines.** Rich word-wraps at the (defaulted) console width
  when output is captured/redirected, which would split long paths and break
  substring matching. The line helpers here pass ``soft_wrap=True`` so a path
  or message is emitted verbatim on one line.
* **Machine-readable output bypasses Rich.** JSON / CSV dumps meant for piping
  are written with plain ``click.echo`` in the command bodies, not through this
  module — Rich highlighting/markup must never touch structured output.

Semantic palette (apply via the helpers, don't hardcode colour elsewhere):
success = green ✓, error = bold-red ``Error:`` (matches Click), warning = yellow !,
info = blue ℹ, file path = cyan, header = bold, label/value = bold value,
dim = secondary. Errors and warnings go to stderr; everything else to stdout.
"""

import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Global console instances. ``console`` is stdout (normal output); ``err_console``
# is stderr — diagnostics (errors/warnings) go there by convention so they never
# pollute piped stdout (e.g. a JSON dump on stdout stays clean).
console = Console()
err_console = Console(stderr=True)

# Semantic palette — the one place colour names live.
PATH_STYLE = "cyan"
HEADER_STYLE = "bold"
VALUE_STYLE = "bold"
DIM_STYLE = "dim"

# Type variable for generic iterables
T = TypeVar("T")


PROGRESS_ENV = "BIOAMLA_PROGRESS"
_JSONL_INTERVAL = 0.1  # seconds between progress events (the last one is always sent)


def machine_progress() -> bool:
    """True when ``BIOAMLA_PROGRESS=jsonl``: progress is JSON lines, not a drawn bar."""
    return os.environ.get(PROGRESS_ENV, "").strip().lower() == "jsonl"


class ProgressBar:
    """
    Rich-based progress bar for batch operations.

    This class provides a convenient wrapper around Rich's Progress
    with sensible defaults for file processing operations.

    With ``BIOAMLA_PROGRESS=jsonl`` nothing is drawn; instead one JSON object per
    line goes to stderr, for a server or script to follow::

        {"event": "start", "description": "Computing features", "done": 0, "total": 120}
        {"event": "progress", "description": "Computing features", "done": 40, "total": 120}
        {"event": "end", "description": "Computing features", "done": 120, "total": 120,
         "ok": true}

    A ``progress`` event is sent when the count crosses another whole percent of
    ``total`` (so every item of a batch of 100 or fewer), at least every 0.1 s
    otherwise, on a change of description, and for the final count; never twice for
    the same state. ``total`` is ``null`` while unknown. Progress events are sent only
    inside the ``with`` block (counting done before it shows in ``start``), and the
    final count is always sent before ``end``; ``end`` has ``ok: false`` if the block
    raised. Events are sent even when the bar itself is disabled (``--quiet``), since
    they were asked for explicitly. If stderr can't be written (a reader went away),
    events stop but the count goes on; the command is unaffected. Writes block while
    stderr's reader isn't keeping up, as stdout's do.

    Example:
        >>> with ProgressBar(total=100, description="Processing files") as pb:
        ...     for file in files:
        ...         process(file)
        ...         pb.advance()
    """

    def __init__(
        self,
        total: int | None = None,
        description: str = "Processing",
        show_speed: bool = True,
        show_time: bool = True,
        transient: bool = False,
        disable: bool = False,
    ) -> None:
        """
        Initialize the progress bar.

        Args:
            total: Total number of items (None for indeterminate)
            description: Description text shown before the bar
            show_speed: Show processing speed
            show_time: Show elapsed and remaining time
            transient: Remove progress bar when complete
            disable: Disable progress bar entirely
        """
        self.total = total
        self.description = description
        self.show_speed = show_speed
        self.show_time = show_time
        self.transient = transient
        self.disable = disable
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._completed = 0
        self._jsonl = machine_progress()
        self._active = False  # events only between start and end
        self._stream_ok = True  # False once stderr can't be written
        self._last_event = 0.0
        self._last_sent: tuple | None = None
        self._last_percent = -1

    def _create_progress(self) -> Progress:
        """Create the Rich Progress instance with appropriate columns."""
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
        ]

        if self.show_time:
            columns.extend(
                [
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ]
            )

        return Progress(
            *columns,
            console=console,
            transient=self.transient,
            disable=self.disable,
        )

    def _emit(self, event: str, *, force: bool = False, **extra) -> None:
        """Write one progress event as a JSON line to stderr (machine mode only)."""
        if not self._stream_ok or (event == "progress" and not self._active):
            return
        state = (self._completed, self.total, self.description)
        now = time.monotonic()
        if event == "progress":
            if state == self._last_sent:
                return  # nothing new
            percent = int(100 * self._completed / self.total) if self.total else self._last_percent
            due = (
                force or percent != self._last_percent or now - self._last_event >= _JSONL_INTERVAL
            )
            if not due:
                return
            self._last_percent = percent
        record = {"event": event, "description": self.description, "done": self._completed}
        record["total"] = self.total
        record.update(extra)
        try:
            sys.stderr.write(json.dumps(record) + "\n")
            sys.stderr.flush()
        except (OSError, ValueError):  # a closed pipe or stream: stop, don't fail the run
            self._stream_ok = False  # keep counting; just send nothing more
            return
        if event != "start":  # the first update after start is never held back
            self._last_event = now
        self._last_sent = state

    def __enter__(self) -> "ProgressBar":
        """Enter the progress bar context."""
        if self._jsonl:
            self._emit("start")
            self._active = True
            return self
        self._progress = self._create_progress()
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description,
            total=self.total,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the progress bar context."""
        if self._jsonl:
            self._emit("progress", force=True)  # the final count, if it was held back
            self._active = False
            self._emit("end", ok=exc_type is None)
            return
        if self._progress:
            self._progress.stop()

    def advance(self, amount: int = 1) -> None:
        """Advance the progress bar."""
        if self._jsonl:
            self._completed += amount
            self._emit("progress", force=self.total is not None and self._completed >= self.total)
            return
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, amount)
            self._completed += amount

    def update(self, completed: int | None = None, description: str | None = None) -> None:
        """Update the progress bar state."""
        if self._jsonl:
            changed = description is not None and description != self.description
            if completed is not None:
                self._completed = completed
            if description is not None:
                self.description = description
            final = self.total is not None and self._completed >= self.total
            self._emit("progress", force=changed or final)
            return
        if self._progress and self._task_id is not None:
            kwargs = {}
            if completed is not None:
                kwargs["completed"] = completed
                self._completed = completed
            if description is not None:
                kwargs["description"] = description
            if kwargs:
                self._progress.update(self._task_id, **kwargs)

    def set_total(self, total: int) -> None:
        """Set the total count (useful when total is unknown at start)."""
        if self._jsonl:
            self.total = total
            return
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, total=total)
            self.total = total

    def on_progress(self, done: int, total: int) -> None:
        """An ``on_progress(done, total)`` callback for library functions.

        Sets the total on the first call, then moves the bar to ``done``.
        """
        if self.total is None:
            self.set_total(total)
        self.update(completed=done)

    @property
    def completed(self) -> int:
        """Get the number of completed items."""
        return self._completed


def track(
    iterable: Iterable[T],
    description: str = "Processing",
    total: int | None = None,
    transient: bool = False,
    disable: bool = False,
) -> Iterator[T]:
    """
    Wrap an iterable with a progress bar.

    This is a convenience function similar to tqdm.

    Args:
        iterable: Items to iterate over
        description: Description for the progress bar
        total: Total count (auto-detected for sequences)
        transient: Remove progress bar when complete
        disable: Disable progress bar

    Yields:
        Items from the iterable

    An item counts as done when the loop asks for the next one, or when the loop
    stops while holding it. In machine mode the ``end`` event's ``ok`` is true only if
    the loop ran to the end; stopping early (``break`` or an error) ends with
    ``ok: false``. A loop that is abandoned without being finished sends ``end`` only
    when the generator is closed or garbage-collected, so close it (or use
    ``contextlib.closing``) if you keep a reference to it.

    Example:
        >>> for file in track(files, description="Processing files"):
        ...     process(file)
    """
    # Try to get length if not provided
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            pass

    with ProgressBar(
        total=total,
        description=description,
        transient=transient,
        disable=disable,
    ) as progress:
        for item in iterable:
            try:
                yield item
            except GeneratorExit:
                progress.advance()  # the loop stopped holding this item
                raise
            progress.advance()


@contextmanager
def status(message: str) -> Iterator[None]:
    """
    Show a status spinner while performing an operation.

    Args:
        message: Status message to display

    Example:
        >>> with status("Loading model..."):
        ...     model = load_model()
    """
    with console.status(f"[bold blue]{message}"):
        yield


def echo(message: object = "", *, style: str | None = None) -> None:
    """Print a single line of plain text, never word-wrapped.

    Drop-in replacement for ``click.echo`` in command bodies. The message is
    rendered as literal text (no Rich markup interpretation, so values
    containing ``[...]`` are safe) and emitted on one line so long paths and
    piped output are never split. Pass ``style`` to colour the whole line.
    """
    console.print(str(message), style=style, soft_wrap=True, markup=False, highlight=False)


def print_success(message: str) -> None:
    """Print a success message with a green check."""
    console.print(f"[green]✓[/green] {escape(message)}", soft_wrap=True, highlight=False)


def print_error(message: str) -> None:
    """Print an error message with a bold-red ``Error:`` prefix (to stderr).

    Matches Click's own error formatting so every error the CLI emits — Click's
    built-in usage/abort errors, the central :class:`BioamlaError` handler, and
    these manual validation messages — reads with one consistent prefix.
    """
    err_console.print(
        f"[bold red]Error:[/bold red] {escape(message)}", soft_wrap=True, highlight=False
    )


def print_warning(message: str) -> None:
    """Print a warning message with a yellow bang (to stderr)."""
    err_console.print(f"[yellow]![/yellow] {escape(message)}", soft_wrap=True, highlight=False)


def print_info(message: str) -> None:
    """Print an info message with a blue marker."""
    console.print(f"[blue]ℹ[/blue] {escape(message)}", soft_wrap=True, highlight=False)


def print_header(text: str) -> None:
    """Print a bold section header."""
    console.print(
        f"[{HEADER_STYLE}]{escape(text)}[/{HEADER_STYLE}]", soft_wrap=True, highlight=False
    )


def print_path(label: str, path: object) -> None:
    """Print a ``label`` followed by a cyan, never-wrapped file path."""
    console.print(
        f"{escape(label)} [{PATH_STYLE}]{escape(str(path))}[/{PATH_STYLE}]",
        soft_wrap=True,
        highlight=False,
    )


def print_kv(label: str, value: object, *, value_style: str = VALUE_STYLE) -> None:
    """Print a ``label: value`` pair with the value emphasised."""
    console.print(
        f"{escape(label)}: [{value_style}]{escape(str(value))}[/{value_style}]",
        soft_wrap=True,
        highlight=False,
    )


def print_table(
    title: str,
    columns: list,
    rows: list,
    show_header: bool = True,
) -> None:
    """
    Print a formatted table.

    Args:
        title: Table title
        columns: List of column names
        rows: List of row data (each row is a list of values)
        show_header: Whether to show column headers
    """
    table = Table(title=title, show_header=show_header)

    for col in columns:
        table.add_column(col)

    for row in rows:
        table.add_row(*[str(v) for v in row])

    console.print(table)


def print_panel(
    content: str,
    title: str | None = None,
    style: str = "blue",
) -> None:
    """
    Print content in a panel/box.

    Args:
        content: Content to display
        title: Optional panel title
        style: Border style color
    """
    console.print(Panel(content, title=title, border_style=style))


def print_summary(
    title: str,
    stats: dict,
    style: str = "blue",
) -> None:
    """
    Print a summary panel with statistics.

    Args:
        title: Summary title
        stats: Dictionary of stat names to values
        style: Border style color
    """
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"[bold]{key}:[/bold] {value:.2f}")
        else:
            lines.append(f"[bold]{key}:[/bold] {value}")

    content = "\n".join(lines)
    print_panel(content, title=title, style=style)


def confirm(message: str, default: bool = False) -> bool:
    """
    Ask for user confirmation.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    response = console.input(f"{message} [{default_str}]: ").strip().lower()

    if not response:
        return default

    return response in ("y", "yes")


def is_terminal() -> bool:
    """Check if we're running in a terminal (TTY)."""
    return sys.stdout.isatty()


class BatchProcessor:
    """
    Helper class for batch processing with progress tracking.

    This class provides a convenient way to process multiple files
    with progress reporting and error handling.

    Example:
        >>> def process_file(filepath):
        ...     # Process the file
        ...     return result

        >>> processor = BatchProcessor(files, "Processing audio")
        >>> results = processor.run(process_file)
        >>> print(f"Processed: {processor.success_count}, Failed: {processor.error_count}")
    """

    def __init__(
        self,
        items: list,
        description: str = "Processing",
        verbose: bool = True,
        stop_on_error: bool = False,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            items: List of items to process
            description: Description for progress bar
            verbose: Show verbose output
            stop_on_error: Stop processing on first error
        """
        self.items = items
        self.description = description
        self.verbose = verbose
        self.stop_on_error = stop_on_error
        self.success_count = 0
        self.error_count = 0
        self.errors: list = []
        self.results: list = []

    def run(self, processor: Callable[[Any], Any]) -> list:
        """
        Run the batch processor.

        Args:
            processor: Function to call for each item

        Returns:
            List of results from successful processing
        """
        self.success_count = 0
        self.error_count = 0
        self.errors = []
        self.results = []

        with ProgressBar(
            total=len(self.items),
            description=self.description,
            disable=not self.verbose,
        ) as progress:
            for item in self.items:
                try:
                    result = processor(item)
                    self.results.append(result)
                    self.success_count += 1
                except Exception as e:
                    self.error_count += 1
                    self.errors.append((item, str(e)))
                    if self.verbose:
                        print_error(f"Failed to process {item}: {e}")
                    if self.stop_on_error:
                        break

                progress.advance()

        if self.verbose:
            self._print_summary()

        return self.results

    def _print_summary(self) -> None:
        """Print processing summary."""
        total = len(self.items)
        console.print()
        if self.error_count == 0:
            print_success(f"Processed {self.success_count}/{total} items successfully")
        else:
            print_warning(
                f"Processed {self.success_count}/{total} items, {self.error_count} failed"
            )
