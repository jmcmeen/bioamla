#!/bin/bash
# =============================================================================
# Bioamla System Dependencies Installer
# =============================================================================
# This script installs system-level dependencies required by bioamla.
# Run this BEFORE installing bioamla via pip.
#
# Usage:
#   ./scripts/install-deps.sh          # Install all dependencies
#   ./scripts/install-deps.sh --check  # Check what's installed
#
# Supported platforms:
#   - Ubuntu/Debian (apt)
#   - Fedora/RHEL/CentOS (dnf/yum)
#   - macOS (Homebrew)
#   - Arch Linux (pacman)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo "=============================================="
    echo " $1"
    echo "=============================================="
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "debian"
        elif command -v dnf &> /dev/null; then
            echo "fedora"
        elif command -v yum &> /dev/null; then
            echo "rhel"
        elif command -v pacman &> /dev/null; then
            echo "arch"
        else
            echo "unknown-linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Check if a command exists
check_command() {
    command -v "$1" &> /dev/null
}

# Check installed dependencies
check_dependencies() {
    print_header "Checking System Dependencies"

    local all_installed=true

    # FFmpeg
    if check_command ffmpeg; then
        print_status "FFmpeg: $(ffmpeg -version 2>&1 | head -n1)"
    else
        print_error "FFmpeg: Not installed"
        all_installed=false
    fi

    # libsndfile (check via pkg-config or library presence)
    if pkg-config --exists sndfile 2>/dev/null || \
       [ -f /usr/lib/libsndfile.so ] || \
       [ -f /usr/local/lib/libsndfile.dylib ] || \
       [ -f /opt/homebrew/lib/libsndfile.dylib ]; then
        print_status "libsndfile: Installed"
    else
        print_warning "libsndfile: May not be installed (soundfile pip package may still work)"
    fi

    # PortAudio
    if pkg-config --exists portaudio-2.0 2>/dev/null || \
       [ -f /usr/lib/libportaudio.so ] || \
       [ -f /usr/local/lib/libportaudio.dylib ] || \
       [ -f /opt/homebrew/lib/libportaudio.dylib ]; then
        print_status "PortAudio: Installed"
    else
        print_warning "PortAudio: Not installed (needed for real-time audio recording)"
        all_installed=false
    fi

    echo ""
    if $all_installed; then
        print_status "All required dependencies are installed!"
    else
        print_warning "Some dependencies are missing. Run this script without --check to install them."
    fi
}

# Install on Debian/Ubuntu
install_debian() {
    print_header "Installing on Debian/Ubuntu"

    print_status "Updating package lists..."
    sudo apt-get update

    print_status "Installing FFmpeg..."
    sudo apt-get install -y ffmpeg

    print_status "Installing libsndfile..."
    sudo apt-get install -y libsndfile1 libsndfile1-dev

    print_status "Installing PortAudio..."
    sudo apt-get install -y portaudio19-dev

    print_status "Installation complete!"
}

# Install on Fedora
install_fedora() {
    print_header "Installing on Fedora"

    print_status "Installing FFmpeg..."
    # FFmpeg may require RPM Fusion
    if ! sudo dnf list installed ffmpeg &> /dev/null; then
        print_warning "FFmpeg requires RPM Fusion repository. Attempting to enable..."
        sudo dnf install -y \
            https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
            https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm \
            2>/dev/null || true
    fi
    sudo dnf install -y ffmpeg ffmpeg-devel

    print_status "Installing libsndfile..."
    sudo dnf install -y libsndfile libsndfile-devel

    print_status "Installing PortAudio..."
    sudo dnf install -y portaudio portaudio-devel

    print_status "Installation complete!"
}

# Install on RHEL/CentOS
install_rhel() {
    print_header "Installing on RHEL/CentOS"

    print_status "Installing EPEL repository..."
    sudo yum install -y epel-release

    print_status "Installing FFmpeg..."
    # FFmpeg may require additional repos on RHEL
    sudo yum install -y ffmpeg ffmpeg-devel 2>/dev/null || \
        print_warning "FFmpeg not available in default repos. Install manually or enable RPM Fusion."

    print_status "Installing libsndfile..."
    sudo yum install -y libsndfile libsndfile-devel

    print_status "Installing PortAudio..."
    sudo yum install -y portaudio portaudio-devel

    print_status "Installation complete!"
}

# Install on Arch Linux
install_arch() {
    print_header "Installing on Arch Linux"

    print_status "Installing dependencies..."
    sudo pacman -S --noconfirm ffmpeg libsndfile portaudio

    print_status "Installation complete!"
}

# Install on macOS
install_macos() {
    print_header "Installing on macOS"

    # Check for Homebrew
    if ! check_command brew; then
        print_error "Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    print_status "Installing FFmpeg..."
    brew install ffmpeg

    print_status "Installing libsndfile..."
    brew install libsndfile

    print_status "Installing PortAudio..."
    brew install portaudio

    print_status "Installation complete!"
}

# Main
main() {
    echo "=============================================="
    echo " Bioamla System Dependencies Installer"
    echo "=============================================="

    # Check-only mode
    if [[ "$1" == "--check" ]]; then
        check_dependencies
        exit 0
    fi

    OS=$(detect_os)
    echo "Detected OS: $OS"

    case $OS in
        debian)
            install_debian
            ;;
        fedora)
            install_fedora
            ;;
        rhel)
            install_rhel
            ;;
        arch)
            install_arch
            ;;
        macos)
            install_macos
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            echo ""
            echo "Please install the following packages manually:"
            echo "  - FFmpeg (for audio format conversion)"
            echo "  - libsndfile (for audio file I/O)"
            echo "  - PortAudio (for real-time audio recording)"
            exit 1
            ;;
    esac

    echo ""
    print_header "Verification"
    check_dependencies

    echo ""
    print_status "You can now install bioamla:"
    echo "  pip install bioamla"
}

main "$@"