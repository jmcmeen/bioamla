#!/bin/bash
# =============================================================================
# Run All Example Scripts
# =============================================================================
# PURPOSE: Execute all example scripts in numeric order for testing.
#
# USAGE: ./run_all.sh [--dry-run] [--stop-on-error]
#
# OPTIONS:
#   --dry-run        Print commands without executing
#   --stop-on-error  Stop execution if any script fails (default: continue)
#
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
DRY_RUN=false
STOP_ON_ERROR=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --stop-on-error)
            STOP_ON_ERROR=true
            ;;
    esac
done

echo "=============================================="
echo "  Running All Example Scripts"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Dry run: $DRY_RUN"
echo "Stop on error: $STOP_ON_ERROR"
echo ""

# Get all numbered scripts in order, excluding 99_run_all.sh
SCRIPTS=$(find "$SCRIPT_DIR" -maxdepth 1 -name "[0-9]*.sh" ! -name "99_run_all.sh" | sort -V)

TOTAL=0
PASSED=0
FAILED=0
FAILED_SCRIPTS=""

for script in $SCRIPTS; do
    TOTAL=$((TOTAL + 1))
    script_name=$(basename "$script")

    echo "----------------------------------------------"
    echo "[$TOTAL] Running: $script_name"
    echo "----------------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would execute: bash $script"
        PASSED=$((PASSED + 1))
    else
        if bash "$script"; then
            echo "  [PASSED] $script_name"
            PASSED=$((PASSED + 1))
        else
            echo "  [FAILED] $script_name"
            FAILED=$((FAILED + 1))
            FAILED_SCRIPTS="$FAILED_SCRIPTS\n  - $script_name"

            if [ "$STOP_ON_ERROR" = true ]; then
                echo ""
                echo "Stopping due to --stop-on-error flag"
                break
            fi
        fi
    fi
    echo ""
done

echo "=============================================="
echo "  Summary"
echo "=============================================="
echo "Total scripts: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo -e "Failed scripts:$FAILED_SCRIPTS"
    exit 1
fi

echo ""
echo "All scripts completed successfully!"
