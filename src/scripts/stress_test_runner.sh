#!/bin/bash

# DICOM Stress Test Runner for Orthanc
# Easy-to-use wrapper for the Python stress test generator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/../python"
PYTHON_SCRIPT="$PYTHON_DIR/stress_test_manager.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}  DICOM Stress Test Runner${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo
}

print_usage() {
    echo "Usage: $0 [command] [options]"
    echo
    echo "Commands:"
    echo "  test-connection         Test connection to Orthanc server"
    echo "  quick-test              Run a quick test (5 series, 2 workers)"
    echo "  medium-test             Run a medium test (20 series, 4 workers)"
    echo "  heavy-test              Run a heavy test (50 series, 8 workers)"
    echo "  concurrent-test         Test high concurrency (100 series, 16 workers)"
    echo "  noise-test              Test with added noise (10 series with 0.1 noise)"
    echo "  custom                  Run with custom parameters"
    echo
    echo "Options for custom:"
    echo "  --template PATH         Path to template DICOM file/directory"
    echo "  --series N              Number of series to generate"
    echo "  --workers N             Number of concurrent workers"
    echo "  --noise LEVEL           Noise level (0-1)"
    echo "  --delay SECONDS         Delay between sends"
    echo "  --max-series N          Max series to keep on disk"
    echo "  --orthanc-url URL       Orthanc server URL"
    echo "  --username USER         Orthanc username"
    echo "  --password PASS         Orthanc password"
    echo
    echo "Examples:"
    echo "  $0 quick-test --template /path/to/dicoms"
    echo "  $0 custom --template /path/to/dicoms --series 30 --workers 6 --noise 0.05"
}

check_requirements() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python 3 is required${NC}"
        exit 1
    fi
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo -e "${RED}Error: Python script not found at $PYTHON_SCRIPT${NC}"
        exit 1
    fi
}

install_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install pydicom numpy requests
}

run_stress_test() {
    local preset="$1"
    shift
    
    # Default values
    local template=""
    local series=10
    local workers=4
    local noise=0
    local delay=0
    local max_series=50
    local orthanc_url="http://localhost:8042"
    local username="demo"
    local password="demo"
    
    # Parse preset configurations
    case "$preset" in
        "quick-test")
            series=5
            workers=2
            echo -e "${GREEN}Running quick stress test (5 series, 2 workers)${NC}"
            ;;
        "medium-test")
            series=20
            workers=4
            echo -e "${GREEN}Running medium stress test (20 series, 4 workers)${NC}"
            ;;
        "heavy-test")
            series=50
            workers=8
            echo -e "${GREEN}Running heavy stress test (50 series, 8 workers)${NC}"
            ;;
        "concurrent-test")
            series=100
            workers=16
            echo -e "${GREEN}Running high concurrency test (100 series, 16 workers)${NC}"
            ;;
        "noise-test")
            series=10
            workers=4
            noise=0.1
            echo -e "${GREEN}Running noise test (10 series with 0.1 noise level)${NC}"
            ;;
        "custom")
            echo -e "${GREEN}Running custom stress test${NC}"
            ;;
    esac
    
    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --template)
                template="$2"
                shift 2
                ;;
            --series)
                series="$2"
                shift 2
                ;;
            --workers)
                workers="$2"
                shift 2
                ;;
            --noise)
                noise="$2"
                shift 2
                ;;
            --delay)
                delay="$2"
                shift 2
                ;;
            --max-series)
                max_series="$2"
                shift 2
                ;;
            --orthanc-url)
                orthanc_url="$2"
                shift 2
                ;;
            --username)
                username="$2"
                shift 2
                ;;
            --password)
                password="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                exit 1
                ;;
        esac
    done
    
    # Check if template is provided
    if [ -z "$template" ]; then
        echo -e "${RED}Error: Template path is required${NC}"
        echo "Use --template /path/to/dicom/files"
        exit 1
    fi
    
    # Check if template exists
    if [ ! -e "$template" ]; then
        echo -e "${RED}Error: Template path '$template' does not exist${NC}"
        exit 1
    fi
    
    echo "Configuration:"
    echo "  Template: $template"
    echo "  Series: $series"
    echo "  Workers: $workers"
    echo "  Noise: $noise"
    echo "  Delay: ${delay}s"
    echo "  Max series on disk: $max_series"
    echo "  Orthanc URL: $orthanc_url"
    echo
    
    # Run the Python script
    python "$PYTHON_SCRIPT" "$template" \
        --num-series "$series" \
        --workers "$workers" \
        --noise "$noise" \
        --delay "$delay" \
        --max-series "$max_series" \
        --orthanc-url "$orthanc_url" \
        --username "$username" \
        --password "$password"
}

test_connection() {
    local orthanc_url="http://localhost:8042"
    local username="demo"
    local password="demo"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --orthanc-url)
                orthanc_url="$2"
                shift 2
                ;;
            --username)
                username="$2"
                shift 2
                ;;
            --password)
                password="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    echo -e "${GREEN}Testing connection to Orthanc...${NC}"
    python "$PYTHON_SCRIPT" dummy \
        --test-connection \
        --orthanc-url "$orthanc_url" \
        --username "$username" \
        --password "$password"
}

# Main script
print_header

# Check if no arguments provided
if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

# Check requirements
check_requirements

# Handle commands
case "$1" in
    "test-connection")
        shift
        test_connection "$@"
        ;;
    "quick-test"|"medium-test"|"heavy-test"|"concurrent-test"|"noise-test"|"custom")
        run_stress_test "$@"
        ;;
    "install-deps")
        install_deps
        ;;
    "--help"|"-h"|"help")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo
        print_usage
        exit 1
        ;;
esac