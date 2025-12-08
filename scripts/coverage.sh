#!/bin/bash
set -e

# Multi-feature coverage script for candle-vllm
# This script runs coverage analysis for different feature combinations
# and generates comprehensive HTML reports.

echo "🔍 Starting candle-vllm coverage analysis..."

# Clean previous coverage
echo "🧹 Cleaning previous coverage reports..."
rm -rf coverage/

# Ensure llvm-tools and cargo-llvm-cov are installed
if ! command -v cargo-llvm-cov &> /dev/null; then
    echo "❌ cargo-llvm-cov not found. Installing..."
    cargo install cargo-llvm-cov
fi

# Check if llvm-tools is installed
if ! rustup component list | grep -q "llvm-tools.*installed"; then
    echo "📦 Installing llvm-tools-preview..."
    rustup component add llvm-tools-preview
fi

# Function to run coverage for a specific feature set
run_coverage() {
    local feature_name=$1
    local features=$2
    local output_dir="coverage/$feature_name"
    
    echo ""
    echo "📊 Running coverage for: $feature_name"
    echo "   Features: $features"
    
    if [ -z "$features" ]; then
        cargo llvm-cov --workspace --html --output-dir "$output_dir" 2>&1 | tail -5
    else
        cargo llvm-cov --workspace --features "$features" --html --output-dir "$output_dir" 2>&1 | tail -5
    fi
    
    echo "✅ $feature_name coverage complete"
}

# Run coverage for each feature combination
run_coverage "cpu" ""
run_coverage "metal" "metal"
run_coverage "cuda" "cuda"
run_coverage "cuda-nccl" "cuda,nccl"

# Run merged coverage with all features
echo ""
echo "📊 Running merged coverage (all features)..."
cargo llvm-cov --workspace --all-features --html --output-dir coverage/merged 2>&1 | tail -10

# Generate text summary
echo ""
echo "📊 Generating coverage summary..."
cargo llvm-cov --workspace --all-features --summary-only > coverage/summary.txt 2>&1

# Display summary
echo ""
echo "════════════════════════════════════════════════════════════"
echo "                    COVERAGE SUMMARY                         "
echo "════════════════════════════════════════════════════════════"
cat coverage/summary.txt
echo "════════════════════════════════════════════════════════════"
echo ""
echo "✅ Coverage analysis complete!"
echo ""
echo "📁 Reports generated:"
echo "   - CPU:        coverage/cpu/index.html"
echo "   - Metal:      coverage/metal/index.html"
echo "   - CUDA:       coverage/cuda/index.html"
echo "   - CUDA+NCCL:  coverage/cuda-nccl/index.html"
echo "   - Merged:     coverage/merged/index.html"
echo ""
echo "🎯 Open coverage/merged/index.html to view comprehensive report"
