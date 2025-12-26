"""
This module provides Rich-based progress bars and console output utilities
for batch operations in bioamla CLI.
"""

import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

from rich.console import Console
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

# Global console instance
console = Console()

# Type variable for generic iterables
T = TypeVar("T")


class ProgressBar:
    """
    Rich-based progress bar for batch operations.

    This class provides a convenient wrapper around Rich's Progress
    with sensible defaults for file processing operations.

    Example:
        >>> with ProgressBar(total=100, description="Processing files") as pb:
        ...     for file in files:
        ...         process(file)
        ...         pb.advance()
    """

    def __init__(
        self,
        total: Optional[int] = None,
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
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._completed = 0

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

    def __enter__(self) -> "ProgressBar":
        """Enter the progress bar context."""
        self._progress = self._create_progress()
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description,
            total=self.total,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the progress bar context."""
        if self._progress:
            self._progress.stop()

    def advance(self, amount: int = 1) -> None:
        """Advance the progress bar."""
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, amount)
            self._completed += amount

    def update(self, completed: Optional[int] = None, description: Optional[str] = None) -> None:
        """Update the progress bar state."""
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
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, total=total)
            self.total = total

    @property
    def completed(self) -> int:
        """Get the number of completed items."""
        return self._completed


def track(
    iterable: Iterable[T],
    description: str = "Processing",
    total: Optional[int] = None,
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
            yield item
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


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message in blue."""
    console.print(f"[blue]ℹ[/blue] {message}")


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
    title: Optional[str] = None,
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
                        print_error(f"Error processing {item}: {e}")
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
