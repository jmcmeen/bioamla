"""
Unit tests for bioamla.tui_commands module.
"""

import click
import pytest

from bioamla.tui_commands import (
    ArgumentInfo,
    CommandInfo,
    GroupInfo,
    OptionInfo,
    build_command_string,
    build_command_tree,
    extract_argument_info,
    extract_option_info,
    flatten_commands,
    get_type_name,
    search_commands,
)


class TestGetTypeName:
    """Tests for get_type_name function."""

    def test_string_type(self):
        """Test TEXT type detection."""
        assert get_type_name(click.STRING) == "TEXT"

    def test_int_type(self):
        """Test INTEGER type detection."""
        assert get_type_name(click.INT) == "INTEGER"

    def test_float_type(self):
        """Test FLOAT type detection."""
        assert get_type_name(click.FLOAT) == "FLOAT"

    def test_bool_type(self):
        """Test BOOLEAN type detection."""
        assert get_type_name(click.BOOL) == "BOOLEAN"

    def test_choice_type(self):
        """Test CHOICE type detection."""
        choice_type = click.Choice(["a", "b", "c"])
        assert get_type_name(choice_type) == "CHOICE"

    def test_path_type(self):
        """Test PATH type detection."""
        path_type = click.Path()
        assert get_type_name(path_type) == "PATH"

    def test_int_range_type(self):
        """Test IntRange type detection."""
        int_range = click.IntRange(0, 100)
        assert get_type_name(int_range) == "INT"

    def test_float_range_type(self):
        """Test FloatRange type detection."""
        float_range = click.FloatRange(0.0, 1.0)
        assert get_type_name(float_range) == "FLOAT"


class TestExtractOptionInfo:
    """Tests for extract_option_info function."""

    def test_simple_string_option(self):
        """Test extracting info from a simple string option."""

        @click.command()
        @click.option("--name", "-n", help="The name")
        def cmd(name):
            pass

        option = cmd.params[0]
        info = extract_option_info(option)

        assert info.name == "name"
        assert "--name" in info.param_decls
        assert "-n" in info.param_decls
        assert info.type_name == "TEXT"
        assert info.help == "The name"
        assert info.is_flag is False
        assert info.required is False

    def test_flag_option(self):
        """Test extracting info from a flag option."""

        @click.command()
        @click.option("--verbose", is_flag=True, help="Enable verbose mode")
        def cmd(verbose):
            pass

        option = cmd.params[0]
        info = extract_option_info(option)

        assert info.name == "verbose"
        assert info.is_flag is True
        assert info.default is False

    def test_choice_option(self):
        """Test extracting info from a choice option."""

        @click.command()
        @click.option("--format", type=click.Choice(["json", "csv", "xml"]))
        def cmd(format):
            pass

        option = cmd.params[0]
        info = extract_option_info(option)

        assert info.type_name == "CHOICE"
        assert info.choices == ["json", "csv", "xml"]

    def test_required_option(self):
        """Test extracting info from a required option."""

        @click.command()
        @click.option("--input", required=True)
        def cmd(input):
            pass

        option = cmd.params[0]
        info = extract_option_info(option)

        assert info.required is True

    def test_option_with_default(self):
        """Test extracting info from an option with default."""

        @click.command()
        @click.option("--count", type=int, default=10)
        def cmd(count):
            pass

        option = cmd.params[0]
        info = extract_option_info(option)

        assert info.default == 10
        assert info.type_name == "INTEGER"


class TestExtractArgumentInfo:
    """Tests for extract_argument_info function."""

    def test_simple_argument(self):
        """Test extracting info from a simple argument."""

        @click.command()
        @click.argument("filename")
        def cmd(filename):
            pass

        argument = cmd.params[0]
        info = extract_argument_info(argument)

        assert info.name == "filename"
        assert info.type_name == "TEXT"
        assert info.required is True

    def test_optional_argument(self):
        """Test extracting info from an optional argument."""

        @click.command()
        @click.argument("output", required=False, default="output.txt")
        def cmd(output):
            pass

        argument = cmd.params[0]
        info = extract_argument_info(argument)

        assert info.required is False
        assert info.default == "output.txt"


class TestBuildCommandTree:
    """Tests for build_command_tree function."""

    def test_builds_tree_from_group(self):
        """Test building a command tree from a group."""

        @click.group()
        def cli():
            pass

        @cli.command()
        def hello():
            """Say hello."""
            pass

        @cli.command()
        def goodbye():
            """Say goodbye."""
            pass

        tree = build_command_tree(cli)

        assert "hello" in tree
        assert "goodbye" in tree
        assert isinstance(tree["hello"], CommandInfo)
        assert isinstance(tree["goodbye"], CommandInfo)
        assert tree["hello"].help == "Say hello."

    def test_builds_nested_groups(self):
        """Test building a tree with nested groups."""

        @click.group()
        def cli():
            pass

        @cli.group()
        def sub():
            """Sub group."""
            pass

        @sub.command()
        def nested():
            """Nested command."""
            pass

        tree = build_command_tree(cli)

        assert "sub" in tree
        assert isinstance(tree["sub"], GroupInfo)
        assert "nested" in tree["sub"].children
        assert tree["sub"].children["nested"].help == "Nested command."

    def test_excludes_specified_commands(self):
        """Test excluding specified commands."""

        @click.group()
        def cli():
            pass

        @cli.command()
        def keep():
            pass

        @cli.command()
        def exclude():
            pass

        tree = build_command_tree(cli, exclude={"exclude"})

        assert "keep" in tree
        assert "exclude" not in tree

    def test_extracts_command_options(self):
        """Test that command options are extracted."""

        @click.group()
        def cli():
            pass

        @cli.command()
        @click.option("--verbose", is_flag=True)
        @click.option("--output", "-o", required=True)
        def cmd(verbose, output):
            """Command with options."""
            pass

        tree = build_command_tree(cli)
        cmd_info = tree["cmd"]

        assert len(cmd_info.options) == 2
        option_names = {opt.name for opt in cmd_info.options}
        assert "verbose" in option_names
        assert "output" in option_names


class TestFlattenCommands:
    """Tests for flatten_commands function."""

    def test_flattens_simple_tree(self):
        """Test flattening a simple command tree."""
        tree = {
            "cmd1": CommandInfo(name="cmd1", path="cli cmd1", help="Command 1"),
            "cmd2": CommandInfo(name="cmd2", path="cli cmd2", help="Command 2"),
        }

        result = flatten_commands(tree)

        assert len(result) == 2
        assert all(isinstance(c, CommandInfo) for c in result)

    def test_flattens_nested_tree(self):
        """Test flattening a nested command tree."""
        tree = {
            "group": GroupInfo(
                name="group",
                path="cli group",
                help="A group",
                children={
                    "nested": CommandInfo(name="nested", path="cli group nested", help="Nested")
                },
            ),
            "top": CommandInfo(name="top", path="cli top", help="Top level"),
        }

        result = flatten_commands(tree)

        assert len(result) == 2
        paths = {c.path for c in result}
        assert "cli group nested" in paths
        assert "cli top" in paths


class TestSearchCommands:
    """Tests for search_commands function."""

    def test_searches_by_name(self):
        """Test searching commands by name."""
        tree = {
            "audio": CommandInfo(name="audio", path="cli audio", help="Audio processing"),
            "video": CommandInfo(name="video", path="cli video", help="Video processing"),
        }

        result = search_commands("audio", tree)

        assert len(result) == 1
        assert result[0].name == "audio"

    def test_searches_by_path(self):
        """Test searching commands by path."""
        tree = {
            "convert": CommandInfo(name="convert", path="cli audio convert", help="Convert files"),
            "analyze": CommandInfo(name="analyze", path="cli video analyze", help="Analyze"),
        }

        result = search_commands("audio", tree)

        assert len(result) == 1
        assert result[0].name == "convert"

    def test_searches_by_help(self):
        """Test searching commands by help text."""
        tree = {
            "cmd1": CommandInfo(name="cmd1", path="cli cmd1", help="Process audio files"),
            "cmd2": CommandInfo(name="cmd2", path="cli cmd2", help="Process video files"),
        }

        result = search_commands("audio", tree)

        assert len(result) == 1
        assert result[0].name == "cmd1"

    def test_case_insensitive_search(self):
        """Test that search is case insensitive."""
        tree = {
            "Audio": CommandInfo(name="Audio", path="cli Audio", help="Audio processing"),
        }

        result = search_commands("audio", tree)

        assert len(result) == 1


class TestBuildCommandString:
    """Tests for build_command_string function."""

    def test_builds_simple_command(self):
        """Test building a simple command string."""
        cmd = CommandInfo(name="hello", path="bioamla hello", help="Say hello")

        result = build_command_string(cmd, {}, {})

        assert result == "bioamla hello"

    def test_includes_arguments(self):
        """Test that arguments are included."""
        cmd = CommandInfo(
            name="convert",
            path="bioamla convert",
            help="Convert files",
            arguments=[
                ArgumentInfo(name="input", type_name="STRING", required=True, default=None, nargs=1, help=""),
                ArgumentInfo(name="output", type_name="STRING", required=True, default=None, nargs=1, help=""),
            ],
        )

        result = build_command_string(cmd, {"input": "in.wav", "output": "out.wav"}, {})

        assert "in.wav" in result
        assert "out.wav" in result

    def test_includes_flag_options(self):
        """Test that flag options are included when enabled."""
        cmd = CommandInfo(
            name="cmd",
            path="bioamla cmd",
            help="A command",
            options=[
                OptionInfo(
                    name="verbose",
                    param_decls=["--verbose", "-v"],
                    type_name="BOOL",
                    default=False,
                    required=False,
                    is_flag=True,
                    help="",
                ),
            ],
        )

        result = build_command_string(cmd, {}, {"verbose": True})

        assert "--verbose" in result

    def test_excludes_disabled_flags(self):
        """Test that disabled flags are not included."""
        cmd = CommandInfo(
            name="cmd",
            path="bioamla cmd",
            help="A command",
            options=[
                OptionInfo(
                    name="verbose",
                    param_decls=["--verbose"],
                    type_name="BOOL",
                    default=False,
                    required=False,
                    is_flag=True,
                    help="",
                ),
            ],
        )

        result = build_command_string(cmd, {}, {"verbose": False})

        assert "--verbose" not in result

    def test_includes_value_options(self):
        """Test that value options are included."""
        cmd = CommandInfo(
            name="cmd",
            path="bioamla cmd",
            help="A command",
            options=[
                OptionInfo(
                    name="output",
                    param_decls=["-o", "--output"],
                    type_name="STRING",
                    default=None,
                    required=False,
                    is_flag=False,
                    help="",
                ),
            ],
        )

        result = build_command_string(cmd, {}, {"output": "result.txt"})

        assert "--output result.txt" in result

    def test_quotes_values_with_spaces(self):
        """Test that values with spaces are quoted."""
        cmd = CommandInfo(
            name="cmd",
            path="bioamla cmd",
            help="A command",
            arguments=[
                ArgumentInfo(name="path", type_name="STRING", required=True, default=None, nargs=1, help=""),
            ],
        )

        result = build_command_string(cmd, {"path": "/my path/file.wav"}, {})

        assert '"/my path/file.wav"' in result

    def test_excludes_default_values(self):
        """Test that options with default values are not included."""
        cmd = CommandInfo(
            name="cmd",
            path="bioamla cmd",
            help="A command",
            options=[
                OptionInfo(
                    name="count",
                    param_decls=["--count"],
                    type_name="INT",
                    default=10,
                    required=False,
                    is_flag=False,
                    help="",
                ),
            ],
        )

        result = build_command_string(cmd, {}, {"count": 10})

        assert "--count" not in result


class TestCommandInfoDataclass:
    """Tests for CommandInfo dataclass."""

    def test_default_values(self):
        """Test default values for CommandInfo."""
        cmd = CommandInfo(name="test", path="cli test", help="Test command")

        assert cmd.arguments == []
        assert cmd.options == []


class TestGroupInfoDataclass:
    """Tests for GroupInfo dataclass."""

    def test_default_values(self):
        """Test default values for GroupInfo."""
        group = GroupInfo(name="test", path="cli test", help="Test group")

        assert group.children == {}


class TestOptionInfoDataclass:
    """Tests for OptionInfo dataclass."""

    def test_default_values(self):
        """Test default values for OptionInfo."""
        opt = OptionInfo(
            name="test",
            param_decls=["--test"],
            type_name="STRING",
            default=None,
            required=False,
            is_flag=False,
            help="Test option",
        )

        assert opt.choices is None
        assert opt.multiple is False
        assert opt.nargs == 1


class TestIntegrationWithRealCLI:
    """Integration tests using the actual bioamla CLI."""

    def test_get_command_tree_succeeds(self):
        """Test that we can get the command tree from the real CLI."""
        from bioamla.tui_commands import get_command_tree

        tree = get_command_tree()

        assert tree is not None
        assert len(tree) > 0

    def test_explore_command_excluded(self):
        """Test that explore command is excluded by default."""
        from bioamla.tui_commands import get_command_tree

        tree = get_command_tree()

        assert "explore" not in tree

    def test_tree_contains_expected_groups(self):
        """Test that expected command groups are present."""
        from bioamla.tui_commands import get_command_tree

        tree = get_command_tree()

        # Check for some expected groups
        expected_groups = ["audio", "models", "config", "detect", "indices"]
        for group in expected_groups:
            assert group in tree, f"Expected group '{group}' not found in tree"

    def test_commands_have_options(self):
        """Test that commands have their options extracted."""
        from bioamla.tui_commands import get_command_tree

        tree = get_command_tree()
        all_commands = flatten_commands(tree)

        # Find a command with options
        commands_with_options = [c for c in all_commands if len(c.options) > 0]

        assert len(commands_with_options) > 0, "No commands with options found"
