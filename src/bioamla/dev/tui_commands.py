"""Command introspection module for the TUI command browser.

This module provides utilities to introspect Click commands and build
a tree structure that can be used by the TUI to display and configure
commands interactively.
"""

from dataclasses import dataclass, field
from typing import Any

import click


@dataclass
class OptionInfo:
    """Information about a command option."""

    name: str
    param_decls: list[str]  # e.g., ['-o', '--output']
    type_name: str  # e.g., 'STRING', 'INT', 'CHOICE'
    default: Any
    required: bool
    is_flag: bool
    help: str
    choices: list[str] | None = None
    multiple: bool = False
    nargs: int = 1


@dataclass
class ArgumentInfo:
    """Information about a command argument."""

    name: str
    type_name: str
    required: bool
    default: Any
    nargs: int
    help: str


@dataclass
class CommandInfo:
    """Information about a Click command."""

    name: str
    path: str  # Full command path, e.g., "audio convert"
    help: str
    arguments: list[ArgumentInfo] = field(default_factory=list)
    options: list[OptionInfo] = field(default_factory=list)


@dataclass
class GroupInfo:
    """Information about a Click command group."""

    name: str
    path: str
    help: str
    children: dict[str, "GroupInfo | CommandInfo"] = field(default_factory=dict)


def get_type_name(param_type: click.ParamType) -> str:
    """Get a string representation of a Click parameter type."""
    if isinstance(param_type, click.Choice):
        return "CHOICE"
    elif isinstance(param_type, click.IntRange):
        return "INT"
    elif isinstance(param_type, click.FloatRange):
        return "FLOAT"
    elif isinstance(param_type, click.Path):
        return "PATH"
    elif isinstance(param_type, click.File):
        return "FILE"
    elif isinstance(param_type, click.DateTime):
        return "DATETIME"
    elif isinstance(param_type, click.Tuple):
        return "TUPLE"

    type_name = param_type.name.upper() if hasattr(param_type, "name") else "STRING"
    return type_name


def extract_option_info(option: click.Option) -> OptionInfo:
    """Extract information from a Click Option."""
    choices = None
    if isinstance(option.type, click.Choice):
        choices = list(option.type.choices)

    return OptionInfo(
        name=option.name,
        param_decls=list(option.opts),
        type_name=get_type_name(option.type),
        default=option.default,
        required=option.required,
        is_flag=option.is_flag,
        help=option.help or "",
        choices=choices,
        multiple=option.multiple,
        nargs=option.nargs if option.nargs != 1 else 1,
    )


def extract_argument_info(argument: click.Argument) -> ArgumentInfo:
    """Extract information from a Click Argument."""
    # Arguments don't have help in Click, but we can use the name
    return ArgumentInfo(
        name=argument.name,
        type_name=get_type_name(argument.type),
        required=argument.required,
        default=argument.default,
        nargs=argument.nargs if argument.nargs != 1 else 1,
        help="",  # Arguments don't have help text in Click
    )


def extract_command_info(
    cmd: click.Command, name: str, path: str
) -> CommandInfo:
    """Extract all information from a Click command."""
    arguments = []
    options = []

    for param in cmd.params:
        if isinstance(param, click.Option):
            options.append(extract_option_info(param))
        elif isinstance(param, click.Argument):
            arguments.append(extract_argument_info(param))

    # Get the first line of help for display
    help_text = ""
    if cmd.help:
        help_text = cmd.help.split("\n")[0].strip()

    return CommandInfo(
        name=name,
        path=path,
        help=help_text,
        arguments=arguments,
        options=options,
    )


def build_command_tree(
    group: click.Group,
    parent_path: str = "bioamla",
    exclude: set[str] | None = None,
) -> dict[str, GroupInfo | CommandInfo]:
    """Recursively build a command tree from a Click group.

    Args:
        group: The Click group to introspect
        parent_path: The command path prefix
        exclude: Set of command names to exclude

    Returns:
        A dictionary mapping command names to their info objects
    """
    if exclude is None:
        exclude = {"explore"}  # Default: exclude explore command

    tree: dict[str, GroupInfo | CommandInfo] = {}

    for name in sorted(group.commands.keys()):
        if name in exclude:
            continue

        cmd = group.commands[name]
        cmd_path = f"{parent_path} {name}"

        if isinstance(cmd, click.Group):
            # Recursively process subgroups
            children = build_command_tree(cmd, cmd_path, exclude)
            help_text = ""
            if cmd.help:
                help_text = cmd.help.split("\n")[0].strip()

            tree[name] = GroupInfo(
                name=name,
                path=cmd_path,
                help=help_text,
                children=children,
            )
        else:
            # Extract command info
            tree[name] = extract_command_info(cmd, name, cmd_path)

    return tree


def get_command_tree() -> dict[str, GroupInfo | CommandInfo]:
    """Get the full command tree for the bioamla CLI.

    Returns:
        A dictionary representing the command hierarchy
    """
    # Import here to avoid circular imports
    from bioamla.cli import cli

    return build_command_tree(cli)


def flatten_commands(
    tree: dict[str, GroupInfo | CommandInfo],
    result: list[CommandInfo] | None = None,
) -> list[CommandInfo]:
    """Flatten the command tree into a list of all commands.

    Useful for searching across all commands.

    Args:
        tree: The command tree to flatten

    Returns:
        A list of all CommandInfo objects
    """
    if result is None:
        result = []

    for item in tree.values():
        if isinstance(item, CommandInfo):
            result.append(item)
        elif isinstance(item, GroupInfo):
            flatten_commands(item.children, result)

    return result


def search_commands(
    query: str,
    tree: dict[str, GroupInfo | CommandInfo] | None = None,
) -> list[CommandInfo]:
    """Search for commands matching a query string.

    Searches command names, paths, and help text.

    Args:
        query: The search query (case-insensitive)
        tree: Optional command tree (loads from CLI if not provided)

    Returns:
        List of matching CommandInfo objects
    """
    if tree is None:
        tree = get_command_tree()

    all_commands = flatten_commands(tree)
    query_lower = query.lower()

    matches = []
    for cmd in all_commands:
        if (
            query_lower in cmd.name.lower()
            or query_lower in cmd.path.lower()
            or query_lower in cmd.help.lower()
        ):
            matches.append(cmd)

    return matches


def build_command_string(
    command_info: CommandInfo,
    argument_values: dict[str, str],
    option_values: dict[str, Any],
) -> str:
    """Build a command string from command info and form values.

    Args:
        command_info: The command info object
        argument_values: Dictionary mapping argument names to values
        option_values: Dictionary mapping option names to values

    Returns:
        The full command string ready to execute
    """
    parts = [command_info.path]

    # Add arguments in order
    for arg in command_info.arguments:
        value = argument_values.get(arg.name, "")
        if value:
            # Quote if contains spaces
            if " " in str(value):
                parts.append(f'"{value}"')
            else:
                parts.append(str(value))

    # Add options
    for opt in command_info.options:
        value = option_values.get(opt.name)

        if opt.is_flag:
            if value:
                # Use the longest option declaration
                opt_str = max(opt.param_decls, key=len)
                parts.append(opt_str)
        elif value is not None and value != "" and value != opt.default:
            # Use the longest option declaration
            opt_str = max(opt.param_decls, key=len)
            if " " in str(value):
                parts.append(f'{opt_str} "{value}"')
            else:
                parts.append(f"{opt_str} {value}")

    return " ".join(parts)
