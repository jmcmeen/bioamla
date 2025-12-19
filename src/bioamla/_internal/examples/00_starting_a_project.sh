#!/bin/bash
# =============================================================================
# Starting a Bioamla Project
# =============================================================================
# PURPOSE: Initialize a bioamla project and set up the directory structure
#          for an acoustic monitoring study. This is the foundation for all
#          other example workflows.
#
# FEATURES DEMONSTRATED:
#   - Project initialization with templates
#   - Configuration management
#   - Directory structure setup
#   - System information and device detection
#   - Command history and logging
#
# This script should be run first before exploring other examples.
# All subsequent examples assume you are working within a project directory.
# =============================================================================

set -e  # Exit on error

# Configuration - customize these for your study
PROJECT_NAME="frog_acoustic_study"


PROJECT_DIR="./${PROJECT_NAME}"

echo "=============================================="
echo "  Bioamla Project Setup"
echo "=============================================="
echo ""

# Step 1: Check bioamla installation and version
echo "Step 1: Checking bioamla installation..."
bioamla version

# Step 2: Check available compute devices (CPU/GPU)
echo ""
echo "Step 2: Detecting compute devices..."
bioamla devices

# Step 3: View default configuration
echo ""
echo "Step 3: Viewing default configuration..."
bioamla config path
bioamla config show

# Step 4: Initialize a new project
echo ""
echo "Step 4: Initializing project '${PROJECT_NAME}'..."
echo ""
echo "Available project templates:"
echo "  - default    : Standard project structure"
echo "  - minimal    : Bare minimum for quick tasks"
echo "  - research   : Full structure for research studies"
echo "  - production : Optimized for deployment pipelines"
echo ""

# Create project with research template (includes all directories)
bioamla project init "$PROJECT_DIR" \
    --name "$PROJECT_NAME" \
    --description "Acoustic monitoring study initialized with bioamla" \
    --template research \
    --force

# Step 5: Enter project directory and verify
echo ""
echo "Step 5: Entering project directory..."
cd "$PROJECT_DIR"

# Step 6: Check project status
echo ""
echo "Step 6: Checking project status..."
bioamla project status

# Step 7: View project configuration
echo ""
echo "Step 7: Viewing project configuration..."
bioamla project config show

# Step 8: Set up recommended directory structure
echo ""
echo "Step 8: Creating recommended directory structure..."
mkdir -p raw_recordings      # Original field recordings
mkdir -p processed           # Preprocessed audio files
mkdir -p results             # Analysis outputs
mkdir -p models              # Trained models
mkdir -p annotations         # Manual annotations
mkdir -p exports             # Exported data for external tools

echo "Created directories:"
ls -la

# Step 9: Initialize command logging
echo ""
echo "Step 9: Command history is automatically tracked..."
bioamla log stats

# Step 10: View available commands
echo ""
echo "Step 10: Available bioamla commands..."
echo ""
echo "Core commands:"
echo "  bioamla audio      - Audio file operations"
echo "  bioamla models     - ML model operations"
echo "  bioamla detect     - Acoustic detection algorithms"
echo "  bioamla indices    - Acoustic indices computation"
echo "  bioamla cluster    - Clustering and discovery"
echo "  bioamla learn      - Active learning workflows"
echo "  bioamla services   - External data services"
echo "  bioamla realtime   - Real-time processing"
echo ""
echo "Project commands:"
echo "  bioamla project    - Project management"
echo "  bioamla config     - Configuration management"
echo "  bioamla log        - Command history"
echo "  bioamla explore    - Dataset exploration"

echo ""
echo "=============================================="
echo "  Project Setup Complete!"
echo "=============================================="
echo ""
echo "Project location: $(pwd)"
echo ""
echo "Directory structure:"
echo "  ${PROJECT_NAME}/"
echo "  ├── .bioamla/           # Project marker and config"
echo "  ├── raw_recordings/     # Place your field recordings here"
echo "  ├── processed/          # Preprocessed audio output"
echo "  ├── results/            # Analysis results"
echo "  ├── models/             # Trained models"
echo "  ├── annotations/        # Manual annotations"
echo "  └── exports/            # Data exports"
echo ""
echo "Next steps:"
echo "  1. Copy your audio recordings to raw_recordings/"
echo "  2. Run 01_audio_preprocessing.sh to prepare files"
echo "  3. Explore other example workflows with: bioamla examples list"
echo ""
echo "Get help anytime:"
echo "  bioamla --help           # General help"
echo "  bioamla <command> --help # Command-specific help"
echo "  bioamla examples show 01 # View example scripts"

