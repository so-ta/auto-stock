#!/bin/bash
# =============================================================================
# Multi-Asset Portfolio - Dedicated Server Setup Script
# =============================================================================
# Usage: ./scripts/setup_dedicated_server.sh [--with-gpu] [--with-s3] [--with-ray]
#
# This script sets up the environment for running on a dedicated server
# with full resource utilization.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Multi-Asset Portfolio - Dedicated Server Setup ===${NC}"
echo ""

# Parse arguments
WITH_GPU=false
WITH_S3=false
WITH_RAY=false

for arg in "$@"; do
    case $arg in
        --with-gpu)
            WITH_GPU=true
            shift
            ;;
        --with-s3)
            WITH_S3=true
            shift
            ;;
        --with-ray)
            WITH_RAY=true
            shift
            ;;
        *)
            ;;
    esac
done

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    echo -e "${RED}Error: Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python version: OK ($PYTHON_VERSION)${NC}"

# Install base package
echo ""
echo -e "${YELLOW}Installing base package with dev dependencies...${NC}"
pip install -e ".[dev]"

# Install optional dependencies
if $WITH_S3; then
    echo ""
    echo -e "${YELLOW}Installing S3 support...${NC}"
    pip install s3fs boto3
fi

if $WITH_GPU; then
    echo ""
    echo -e "${YELLOW}Installing GPU support (CuPy)...${NC}"
    # Detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
        if [[ "$CUDA_VERSION" == "12" ]]; then
            pip install cupy-cuda12x
        else
            pip install cupy-cuda11x
        fi
    else
        echo -e "${RED}Warning: NVIDIA driver not found. Skipping GPU setup.${NC}"
    fi
fi

if $WITH_RAY; then
    echo ""
    echo -e "${YELLOW}Installing Ray for distributed processing...${NC}"
    pip install ray
fi

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
python3 -c "
from src.config import print_resource_summary
print()
print_resource_summary()
"

# Check S3 configuration
if $WITH_S3; then
    echo ""
    echo -e "${YELLOW}Checking AWS credentials...${NC}"
    if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
        echo -e "${RED}Warning: AWS credentials not set in environment.${NC}"
        echo "Set these environment variables:"
        echo "  export AWS_ACCESS_KEY_ID=your_access_key"
        echo "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
    else
        echo -e "${GREEN}AWS credentials: OK${NC}"
    fi
fi

# Create local config if not exists
if [[ ! -f "config/local.yaml" ]]; then
    echo ""
    echo -e "${YELLOW}Creating config/local.yaml from template...${NC}"
    cp config/local.yaml.example config/local.yaml 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Review and edit config/local.yaml for your environment"
echo "2. Set AWS credentials if using S3:"
echo "   export AWS_ACCESS_KEY_ID=your_key"
echo "   export AWS_SECRET_ACCESS_KEY=your_secret"
echo "3. Run a test backtest:"
echo "   python -m src.main --backtest --start 2020-01-01 --end 2024-12-31"
echo ""
