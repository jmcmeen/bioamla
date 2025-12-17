"""
Unit tests for the examples CLI command group.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from bioamla.cli import cli
from bioamla.examples import EXAMPLES, get_example_content, list_examples


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestExamplesListCommand:
    """Tests for bioamla examples list command."""

    def test_list_shows_all_examples(self, runner):
        """Test that list shows all available examples."""
        result = runner.invoke(cli, ["examples", "list"])

        assert result.exit_code == 0
        # Check that all example IDs are shown (or truncated with ...)
        # Long IDs may be truncated in table output
        for example_id in EXAMPLES.keys():
            # Check if full ID or truncated version (first 3 chars + ...) is present
            assert example_id in result.output or example_id[:3] in result.output

    def test_list_shows_titles(self, runner):
        """Test that list shows example titles."""
        result = runner.invoke(cli, ["examples", "list"])

        assert result.exit_code == 0
        # Check some known titles
        assert "Audio Preprocessing" in result.output
        assert "Starting a Project" in result.output

    def test_list_shows_descriptions(self, runner):
        """Test that list shows example descriptions."""
        result = runner.invoke(cli, ["examples", "list"])

        assert result.exit_code == 0
        # Check that descriptions are included
        for _, (_, _, desc) in EXAMPLES.items():
            # At least part of the description should be visible
            assert any(word in result.output for word in desc.split()[:3])


class TestExamplesShowCommand:
    """Tests for bioamla examples show command."""

    def test_show_displays_example_content(self, runner):
        """Test that show displays the content of an example."""
        result = runner.invoke(cli, ["examples", "show", "00"])

        assert result.exit_code == 0
        # Check for expected content from 00_starting_a_project.sh
        assert "#!/bin/bash" in result.output or "bioamla" in result.output

    def test_show_with_valid_id(self, runner):
        """Test show with various valid IDs."""
        for example_id in ["00", "01", "05", "10"]:
            result = runner.invoke(cli, ["examples", "show", example_id])
            assert result.exit_code == 0
            assert len(result.output) > 0

    def test_show_with_invalid_id(self, runner):
        """Test show with invalid example ID."""
        result = runner.invoke(cli, ["examples", "show", "99"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_show_requires_example_id(self, runner):
        """Test that show requires an example ID argument."""
        result = runner.invoke(cli, ["examples", "show"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "EXAMPLE_ID" in result.output


class TestExamplesCopyCommand:
    """Tests for bioamla examples copy command."""

    def test_copy_creates_file(self, runner, temp_dir):
        """Test that copy creates the example file in the target directory."""
        result = runner.invoke(cli, ["examples", "copy", "00", str(temp_dir)])

        assert result.exit_code == 0
        copied_file = temp_dir / "00_starting_a_project.sh"
        assert copied_file.exists()

    def test_copy_file_has_correct_content(self, runner, temp_dir):
        """Test that copied file has the correct content."""
        result = runner.invoke(cli, ["examples", "copy", "01", str(temp_dir)])

        assert result.exit_code == 0
        copied_file = temp_dir / "01_audio_preprocessing.sh"
        assert copied_file.exists()

        # Compare content
        expected_content = get_example_content("01")
        actual_content = copied_file.read_text()
        assert actual_content == expected_content

    def test_copy_refuses_overwrite_without_force(self, runner, temp_dir):
        """Test that copy refuses to overwrite existing file without --force."""
        # First copy
        runner.invoke(cli, ["examples", "copy", "00", str(temp_dir)])

        # Second copy without --force should fail
        result = runner.invoke(cli, ["examples", "copy", "00", str(temp_dir)])

        assert result.exit_code != 0
        assert "exists" in result.output.lower() or "force" in result.output.lower()

    def test_copy_with_force_overwrites(self, runner, temp_dir):
        """Test that copy with --force overwrites existing file."""
        # First copy
        runner.invoke(cli, ["examples", "copy", "00", str(temp_dir)])

        # Second copy with --force should succeed
        result = runner.invoke(cli, ["examples", "copy", "00", str(temp_dir), "--force"])

        assert result.exit_code == 0

    def test_copy_creates_output_directory(self, runner, temp_dir):
        """Test that copy creates output directory if it doesn't exist."""
        new_dir = temp_dir / "new_subdir"
        result = runner.invoke(cli, ["examples", "copy", "00", str(new_dir)])

        assert result.exit_code == 0
        assert new_dir.exists()
        assert (new_dir / "00_starting_a_project.sh").exists()

    def test_copy_with_invalid_id(self, runner, temp_dir):
        """Test copy with invalid example ID."""
        result = runner.invoke(cli, ["examples", "copy", "99", str(temp_dir)])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestExamplesCopyAllCommand:
    """Tests for bioamla examples copy-all command."""

    def test_copy_all_creates_all_files(self, runner, temp_dir):
        """Test that copy-all creates all example files."""
        result = runner.invoke(cli, ["examples", "copy-all", str(temp_dir)])

        assert result.exit_code == 0
        # Check that all example files were created
        for example_id, (filename, _, _) in EXAMPLES.items():
            copied_file = temp_dir / filename
            assert copied_file.exists(), f"Missing file: {filename}"

    def test_copy_all_shows_count(self, runner, temp_dir):
        """Test that copy-all shows the number of copied files."""
        result = runner.invoke(cli, ["examples", "copy-all", str(temp_dir)])

        assert result.exit_code == 0
        # Should mention the number of examples copied
        assert str(len(EXAMPLES)) in result.output or "all" in result.output.lower()

    def test_copy_all_skips_existing_without_force(self, runner, temp_dir):
        """Test that copy-all skips existing files without --force."""
        # First copy
        runner.invoke(cli, ["examples", "copy-all", str(temp_dir)])

        # Second copy without --force should skip existing files
        result = runner.invoke(cli, ["examples", "copy-all", str(temp_dir)])

        assert result.exit_code == 0
        # Should indicate files were skipped
        assert "skipped" in result.output.lower() or "exists" in result.output.lower()

    def test_copy_all_with_force_overwrites(self, runner, temp_dir):
        """Test that copy-all with --force overwrites existing files."""
        # First copy
        runner.invoke(cli, ["examples", "copy-all", str(temp_dir)])

        # Second copy with --force should succeed
        result = runner.invoke(cli, ["examples", "copy-all", str(temp_dir), "--force"])

        assert result.exit_code == 0

    def test_copy_all_creates_output_directory(self, runner, temp_dir):
        """Test that copy-all creates output directory if it doesn't exist."""
        new_dir = temp_dir / "examples_output"
        result = runner.invoke(cli, ["examples", "copy-all", str(new_dir)])

        assert result.exit_code == 0
        assert new_dir.exists()


class TestExamplesInfoCommand:
    """Tests for bioamla examples info command."""

    def test_info_shows_example_details(self, runner):
        """Test that info shows details about an example."""
        result = runner.invoke(cli, ["examples", "info", "01"])

        assert result.exit_code == 0
        # Should show title
        assert "Audio Preprocessing" in result.output
        # Should show filename
        assert "01_audio_preprocessing.sh" in result.output

    def test_info_with_various_ids(self, runner):
        """Test info with various valid IDs."""
        for example_id, (filename, title, _) in EXAMPLES.items():
            result = runner.invoke(cli, ["examples", "info", example_id])
            assert result.exit_code == 0
            assert title in result.output
            assert filename in result.output

    def test_info_with_invalid_id(self, runner):
        """Test info with invalid example ID."""
        result = runner.invoke(cli, ["examples", "info", "99"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_info_requires_example_id(self, runner):
        """Test that info requires an example ID argument."""
        result = runner.invoke(cli, ["examples", "info"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "EXAMPLE_ID" in result.output


class TestExamplesHelpCommand:
    """Tests for examples command help."""

    def test_examples_help(self, runner):
        """Test examples --help shows available subcommands."""
        result = runner.invoke(cli, ["examples", "--help"])

        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "copy" in result.output
        assert "copy-all" in result.output
        assert "info" in result.output

    def test_examples_list_help(self, runner):
        """Test examples list --help."""
        result = runner.invoke(cli, ["examples", "list", "--help"])

        assert result.exit_code == 0

    def test_examples_show_help(self, runner):
        """Test examples show --help shows arguments."""
        result = runner.invoke(cli, ["examples", "show", "--help"])

        assert result.exit_code == 0
        assert "EXAMPLE_ID" in result.output

    def test_examples_copy_help(self, runner):
        """Test examples copy --help shows all options."""
        result = runner.invoke(cli, ["examples", "copy", "--help"])

        assert result.exit_code == 0
        assert "EXAMPLE_ID" in result.output
        assert "OUTPUT_DIR" in result.output
        assert "--force" in result.output

    def test_examples_copy_all_help(self, runner):
        """Test examples copy-all --help shows all options."""
        result = runner.invoke(cli, ["examples", "copy-all", "--help"])

        assert result.exit_code == 0
        assert "OUTPUT_DIR" in result.output
        assert "--force" in result.output


class TestExamplesModule:
    """Tests for the examples module directly."""

    def test_list_examples_returns_all(self):
        """Test that list_examples returns all examples."""
        examples = list_examples()

        assert len(examples) == len(EXAMPLES)

    def test_list_examples_format(self):
        """Test that list_examples returns correct tuple format."""
        examples = list_examples()

        for example in examples:
            assert len(example) == 3  # (id, title, description)
            assert isinstance(example[0], str)
            assert isinstance(example[1], str)
            assert isinstance(example[2], str)

    def test_get_example_content_returns_string(self):
        """Test that get_example_content returns content as string."""
        content = get_example_content("00")

        assert isinstance(content, str)
        assert len(content) > 0
        assert "#!/bin/bash" in content

    def test_get_example_content_invalid_id(self):
        """Test that get_example_content raises for invalid ID."""
        with pytest.raises(ValueError):
            get_example_content("99")

    def test_examples_dict_has_required_fields(self):
        """Test that EXAMPLES dict has all required fields."""
        for example_id, data in EXAMPLES.items():
            assert len(data) == 3  # (filename, title, description)
            filename, title, description = data
            assert filename.endswith(".sh")
            assert example_id in filename
            assert len(title) > 0
            assert len(description) > 0
