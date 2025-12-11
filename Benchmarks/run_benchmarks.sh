#!/bin/bash
#
# cLean GPU Benchmark Runner
# Runs both cLean and CUDA reference benchmarks and generates comparison reports
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              cLean GPU Benchmark Suite                           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Compile CUDA reference benchmarks
echo -e "${YELLOW}[1/4] Compiling CUDA reference benchmarks...${NC}"
if command -v nvcc &> /dev/null; then
    cd "$SCRIPT_DIR/cuda"
    nvcc -O3 -o cuda_benchmarks cuda_reference.cu 2>/dev/null || {
        echo -e "${RED}  ✗ CUDA compilation failed${NC}"
        CUDA_AVAILABLE=false
    }
    if [ -f cuda_benchmarks ]; then
        echo -e "${GREEN}  ✓ CUDA benchmarks compiled${NC}"
        CUDA_AVAILABLE=true
    fi
    cd "$PROJECT_ROOT"
else
    echo -e "${YELLOW}  ⚠ nvcc not found, skipping CUDA reference benchmarks${NC}"
    CUDA_AVAILABLE=false
fi

# Step 2: Run CUDA benchmarks (if available)
if [ "$CUDA_AVAILABLE" = true ]; then
    echo -e "${YELLOW}[2/4] Running CUDA reference benchmarks...${NC}"
    "$SCRIPT_DIR/cuda/cuda_benchmarks" > "$RESULTS_DIR/cuda_results.json" 2>"$RESULTS_DIR/cuda_log.txt"
    echo -e "${GREEN}  ✓ CUDA benchmarks complete${NC}"
else
    echo -e "${YELLOW}[2/4] Skipping CUDA benchmarks (nvcc not available)${NC}"
    echo "[]" > "$RESULTS_DIR/cuda_results.json"
fi

# Step 3: Build cLean benchmark module
echo -e "${YELLOW}[3/4] Building cLean benchmarks...${NC}"
cd "$PROJECT_ROOT"
lake build Benchmarks 2>&1 | tee "$RESULTS_DIR/clean_build.log" || {
    echo -e "${RED}  ✗ cLean benchmark build failed${NC}"
    echo "See $RESULTS_DIR/clean_build.log for details"
    exit 1
}
echo -e "${GREEN}  ✓ cLean benchmarks built${NC}"

# Step 4: Run cLean benchmarks
echo -e "${YELLOW}[4/4] Running cLean benchmarks...${NC}"
lake exe benchmarks 2>&1 | tee "$RESULTS_DIR/clean_log.txt" || {
    echo -e "${RED}  ✗ cLean benchmarks failed${NC}"
    exit 1
}
echo -e "${GREEN}  ✓ cLean benchmarks complete${NC}"

# Step 5: Generate visualizations
echo -e "${YELLOW}[5/5] Generating visualizations...${NC}"
if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/visualize.py" 2>&1 | tee -a "$RESULTS_DIR/visualize_log.txt" || {
        echo -e "${YELLOW}  ⚠ Visualization generation had errors (see log)${NC}"
    }
    if [ -d "$RESULTS_DIR/plots" ] && [ "$(ls -A $RESULTS_DIR/plots 2>/dev/null)" ]; then
        echo -e "${GREEN}  ✓ Visualizations generated${NC}"
    else
        echo -e "${YELLOW}  ⚠ No plots generated${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ python3 not found, skipping visualization${NC}"
fi

# Generate summary
echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Benchmark Results:${NC}"
echo "  • JSON data:       $RESULTS_DIR/benchmark_data.json"
echo "  • CSV data:        $RESULTS_DIR/benchmark_data.csv"
echo "  • Markdown report: $RESULTS_DIR/benchmark_report.md"
echo "  • LaTeX table:     $RESULTS_DIR/benchmark_table.tex"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "  • CUDA results:    $RESULTS_DIR/cuda_results.json"
fi
if [ -d "$RESULTS_DIR/plots" ] && [ "$(ls -A $RESULTS_DIR/plots 2>/dev/null)" ]; then
    echo "  • Plots:           $RESULTS_DIR/plots/"
fi
echo ""
echo -e "${GREEN}To regenerate plots:${NC}"
echo "  python3 Benchmarks/visualize.py"
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
