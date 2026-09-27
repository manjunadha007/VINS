#!/bin/bash
set -e

echo "==========================================="
echo "Installing Intel RealSense DKMS Kernel Patches"
echo "==========================================="

# Register Intel apt repository
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/apt-repo.gpg | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt-get update
sudo apt-get install -y librealsense2-dkms librealsense2-udev-rules

echo "✅ RealSense kernel patches installed!"
echo "Please UNPLUG and REPLUG your RealSense camera now."
