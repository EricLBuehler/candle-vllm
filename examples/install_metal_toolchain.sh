#!/bin/bash
# install-metal-toolchain.sh

if xcodebuild -showComponent metalToolchain >/dev/null 2>&1; then
    echo "✅ Metal toolchain is installed"
else
    echo "❌ Metal toolchain is not installed"
    echo "⬇️ Downloading metal toolchain..."
    xcodebuild -downloadComponent metalToolchain -exportPath /tmp/metalToolchainDownload/
    echo "🧰 Installing metal toolchain..."
    xcodebuild -importComponent metalToolchain -importPath /tmp/metalToolchainDownload/*.exportedBundle
fi
