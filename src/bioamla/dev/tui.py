"""Command Browser TUI for bioamla.

A terminal user interface that provides hierarchical navigation through
all bioamla CLI commands with interactive forms for configuring and
executing commands.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    Tree,
)
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from bioamla.dev.tui_commands import CommandInfo, GroupInfo, OptionInfo


# =============================================================================
# Custom Widgets
# =============================================================================


class CommandTree(Tree):
    """A tree widget showing the command hierarchy."""

    class CommandSelected(Message):
        """Message sent when a command is selected."""

        def __init__(self, command_info: CommandInfo) -> None:
            self.command_info = command_info
            super().__init__()

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__("bioamla", name=name, id=id, classes=classes)
        self.command_map: dict[str, CommandInfo] = {}

    def on_mount(self) -> None:
        """Build the command tree on mount."""
        from bioamla.dev.tui_commands import get_command_tree

        tree = get_command_tree()
        self._add_nodes(self.root, tree)
        self.root.expand()

    def _add_nodes(
        self,
        parent: TreeNode,
        items: dict[str, GroupInfo | CommandInfo],
    ) -> None:
        """Recursively add nodes to the tree."""
        from bioamla.dev.tui_commands import GroupInfo

        for name, item in items.items():
            if isinstance(item, GroupInfo):
                # Add group as expandable folder with folder icon
                node = parent.add(f"📂 {name}", expand=False)
                node.data = item
                self._add_nodes(node, item.children)
            else:
                # Add command as leaf with command icon
                node = parent.add(f"  {name}")
                node.data = item
                node.allow_expand = False
                self.command_map[item.path] = item

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection."""
        from bioamla.dev.tui_commands import CommandInfo

        if isinstance(event.node.data, CommandInfo):
            self.post_message(self.CommandSelected(event.node.data))


class FormField(Container):
    """A form field with label and input widget."""

    DEFAULT_CSS = """
    FormField {
        height: auto;
        margin-bottom: 1;
    }
    FormField .field-label {
        margin-bottom: 0;
    }
    FormField .field-help {
        color: $text-muted;
        margin-left: 2;
    }
    FormField Input {
        width: 100%;
    }
    FormField Select {
        width: 100%;
    }
    """

    def __init__(
        self,
        label: str,
        field_name: str,
        widget: Input | Select | Switch,
        help_text: str = "",
        required: bool = False,
    ) -> None:
        super().__init__()
        self.label_text = label
        self.field_name = field_name
        self.input_widget = widget
        self.help_text = help_text
        self.required = required

    def compose(self) -> ComposeResult:
        req_marker = "[red]*[/]" if self.required else ""
        yield Label(
            f"[bold]{self.label_text}[/]{req_marker}",
            classes="field-label",
        )
        yield self.input_widget
        if self.help_text:
            yield Label(f"[dim]{self.help_text}[/]", classes="field-help")

    def get_value(self) -> Any:
        """Get the current value of the field."""
        if isinstance(self.input_widget, Switch):
            return self.input_widget.value
        elif isinstance(self.input_widget, Select):
            val = self.input_widget.value
            return val if val != Select.BLANK else ""
        elif isinstance(self.input_widget, Input):
            return self.input_widget.value
        return None


class CommandForm(ScrollableContainer):
    """A form for configuring command options."""

    DEFAULT_CSS = """
    CommandForm {
        padding: 1 2;
    }
    CommandForm .form-title {
        text-style: bold;
        margin-bottom: 1;
    }
    CommandForm .form-description {
        color: $text-muted;
        margin-bottom: 2;
    }
    CommandForm .section-header {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
        color: $secondary;
    }
    CommandForm .button-row {
        margin-top: 2;
        height: auto;
    }
    CommandForm Button {
        margin-right: 2;
    }
    """

    current_command: reactive[CommandInfo | None] = reactive(None)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.argument_fields: dict[str, FormField] = {}
        self.option_fields: dict[str, FormField] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim]Select a command from the tree[/]",
            id="form-placeholder",
        )

    def watch_current_command(self, command: CommandInfo | None) -> None:
        """Update the form when command changes."""
        if command:
            self._build_form(command)

    def _build_form(self, command: CommandInfo) -> None:
        """Build the form for a command."""
        self.argument_fields.clear()
        self.option_fields.clear()

        # Clear existing content
        self.remove_children()

        # Title and description
        self.mount(Static(f"[bold blue]{command.path}[/]", classes="form-title"))
        if command.help:
            self.mount(Static(command.help, classes="form-description"))

        # Arguments section
        if command.arguments:
            self.mount(Static("Arguments", classes="section-header"))
            for arg in command.arguments:
                field = self._create_argument_field(arg)
                self.argument_fields[arg.name] = field
                self.mount(field)

        # Options section
        if command.options:
            self.mount(Static("Options", classes="section-header"))
            for opt in command.options:
                # Skip help option
                if opt.name == "help":
                    continue
                field = self._create_option_field(opt)
                self.option_fields[opt.name] = field
                self.mount(field)

        # Buttons
        button_row = Horizontal(classes="button-row")
        button_row.mount(
            Button("Run", id="run-btn", variant="primary"),
            Button("Copy Command", id="copy-btn", variant="default"),
        )
        self.mount(button_row)

    def _create_argument_field(self, arg) -> FormField:
        """Create a form field for an argument."""
        widget = Input(
            placeholder=arg.name,
            id=f"arg-{arg.name}",
        )

        return FormField(
            label=arg.name,
            field_name=arg.name,
            widget=widget,
            help_text=arg.help,
            required=arg.required,
        )

    def _create_option_field(self, opt: OptionInfo) -> FormField:
        """Create a form field for an option."""
        if opt.is_flag:
            widget = Switch(value=bool(opt.default), id=f"opt-{opt.name}")
        elif opt.choices:
            options = [(c, c) for c in opt.choices]
            # Add blank option at start
            options.insert(0, ("", Select.BLANK))
            widget = Select(
                options=options,
                value=opt.default if opt.default in opt.choices else Select.BLANK,
                id=f"opt-{opt.name}",
                allow_blank=True,
            )
        else:
            placeholder = f"{opt.type_name.lower()}"
            if opt.default is not None and opt.default != ():
                placeholder = f"{opt.default}"
            widget = Input(
                placeholder=placeholder,
                value=str(opt.default) if opt.default and opt.default != () else "",
                id=f"opt-{opt.name}",
            )

        # Build option label from param declarations
        opt_label = ", ".join(opt.param_decls)

        return FormField(
            label=opt_label,
            field_name=opt.name,
            widget=widget,
            help_text=opt.help,
            required=opt.required,
        )

    def get_argument_values(self) -> dict[str, str]:
        """Get current argument values from the form."""
        return {name: field.get_value() for name, field in self.argument_fields.items()}

    def get_option_values(self) -> dict[str, Any]:
        """Get current option values from the form."""
        return {name: field.get_value() for name, field in self.option_fields.items()}


class CommandOutput(Container):
    """Widget showing command preview and output."""

    DEFAULT_CSS = """
    CommandOutput {
        border-top: solid $primary;
    }
    CommandOutput .preview-section {
        height: auto;
        padding: 1;
        background: $surface;
    }
    CommandOutput .preview-label {
        text-style: bold;
        margin-bottom: 0;
    }
    CommandOutput .preview-command {
        color: $success;
    }
    CommandOutput RichLog {
        height: 1fr;
        border-top: solid $surface-lighten-1;
    }
    """

    command_preview: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Vertical(classes="preview-section"):
            yield Label("Command Preview:", classes="preview-label")
            yield Static("$ [dim]select a command[/]", id="preview-text")
        yield RichLog(id="output-log", highlight=True, markup=True)

    def watch_command_preview(self, preview: str) -> None:
        """Update preview display."""
        preview_widget = self.query_one("#preview-text", Static)
        if preview:
            preview_widget.update(f"$ [green]{preview}[/]")
        else:
            preview_widget.update("$ [dim]select a command[/]")

    def write_output(self, text: str, style: str = "") -> None:
        """Write text to the output log."""
        log = self.query_one("#output-log", RichLog)
        if style:
            log.write(f"[{style}]{text}[/]")
        else:
            log.write(text)

    def clear_output(self) -> None:
        """Clear the output log."""
        log = self.query_one("#output-log", RichLog)
        log.clear()


# =============================================================================
# Main Application
# =============================================================================


class CommandBrowser(App):
    """The main command browser application."""

    TITLE = "bioamla Command Browser"
    SUB_TITLE = "Navigate and execute CLI commands"

    CSS = """
    #top-panes {
        height: 1fr;
    }

    #tree-panel {
        width: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #form-panel {
        width: 2fr;
        border: solid $primary;
    }

    #output-panel {
        height: 12;
        border-top: solid $primary;
    }

    CommandTree {
        height: 100%;
    }

    CommandTree > .tree--cursor {
        background: $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "run_command", "Run"),
        Binding("c", "copy_command", "Copy"),
        Binding("/", "focus_search", "Search"),
        Binding("?", "show_help", "Help"),
        Binding("escape", "focus_tree", "Tree"),
    ]

    current_command: reactive[CommandInfo | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-panes"):
            yield CommandTree(id="tree-panel")
            yield CommandForm(id="form-panel")
        yield CommandOutput(id="output-panel")
        yield Footer()

    def on_command_tree_command_selected(
        self, event: CommandTree.CommandSelected
    ) -> None:
        """Handle command selection from tree."""
        self.current_command = event.command_info
        form = self.query_one(CommandForm)
        form.current_command = event.command_info
        self._update_preview()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update preview when form inputs change."""
        self._update_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update preview when select changes."""
        self._update_preview()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Update preview when switch changes."""
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the command preview."""
        if not self.current_command:
            return

        from bioamla.dev.tui_commands import build_command_string

        form = self.query_one(CommandForm)
        cmd_str = build_command_string(
            self.current_command,
            form.get_argument_values(),
            form.get_option_values(),
        )

        output = self.query_one(CommandOutput)
        output.command_preview = cmd_str

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run-btn":
            self.action_run_command()
        elif event.button.id == "copy-btn":
            self.action_copy_command()

    def action_run_command(self) -> None:
        """Run the current command."""
        if not self.current_command:
            return

        from bioamla.dev.tui_commands import build_command_string

        form = self.query_one(CommandForm)
        cmd_str = build_command_string(
            self.current_command,
            form.get_argument_values(),
            form.get_option_values(),
        )

        output = self.query_one(CommandOutput)
        output.clear_output()
        output.write_output(f"$ {cmd_str}\n", "bold green")

        self._execute_command(cmd_str)

    @work(thread=True)
    def _execute_command(self, cmd_str: str) -> None:
        """Execute a command in a background thread."""
        output = self.query_one(CommandOutput)

        try:
            # Run the command
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in process.stdout:
                self.call_from_thread(output.write_output, line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.call_from_thread(
                    output.write_output, "\n[Command completed successfully]", "green"
                )
            else:
                self.call_from_thread(
                    output.write_output,
                    f"\n[Command exited with code {process.returncode}]",
                    "red",
                )

        except Exception as e:
            self.call_from_thread(output.write_output, f"\nError: {e}", "red")

    def action_copy_command(self) -> None:
        """Copy the current command to clipboard."""
        if not self.current_command:
            return

        from bioamla.dev.tui_commands import build_command_string

        form = self.query_one(CommandForm)
        cmd_str = build_command_string(
            self.current_command,
            form.get_argument_values(),
            form.get_option_values(),
        )

        # Try to copy to clipboard
        try:
            import pyperclip

            pyperclip.copy(cmd_str)
            output = self.query_one(CommandOutput)
            output.write_output("[Copied to clipboard]", "green")
        except ImportError:
            output = self.query_one(CommandOutput)
            output.write_output(
                "[Install pyperclip for clipboard support]", "yellow"
            )

    def action_focus_tree(self) -> None:
        """Focus the command tree."""
        self.query_one(CommandTree).focus()

    def action_show_help(self) -> None:
        """Show help information."""
        output = self.query_one(CommandOutput)
        output.clear_output()
        output.write_output("[bold]bioamla Command Browser Help[/]\n")
        output.write_output("Navigation:")
        output.write_output("  Arrow keys - Navigate tree")
        output.write_output("  Enter      - Select command / Expand group")
        output.write_output("  Tab        - Move between fields")
        output.write_output("")
        output.write_output("Actions:")
        output.write_output("  r          - Run current command")
        output.write_output("  c          - Copy command to clipboard")
        output.write_output("  /          - Search commands")
        output.write_output("  ?          - Show this help")
        output.write_output("  q          - Quit")
        output.write_output("  Escape     - Focus tree")


def run_explorer(directory: str | None = None) -> None:
    """Launch the command browser TUI.

    Args:
        directory: Ignored (kept for backwards compatibility)
    """
    app = CommandBrowser()
    app.run()


# Keep the old function name for compatibility
run_dashboard = run_explorer


if __name__ == "__main__":
    run_explorer()
