#!/bin/bash
# Isaac ROS VSLAM Setup Script — Run this manually in your terminal
# Takes ~10-15 minutes total (mostly Docker pull time)
set -e

echo "═══════════════════════════════════════════"
echo "STEP 1/4: Installing Docker CE"
echo "═══════════════════════════════════════════"
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
echo "✅ Docker installed"

echo "═══════════════════════════════════════════"
echo "STEP 2/4: Installing NVIDIA Container Toolkit"
echo "═══════════════════════════════════════════"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "✅ NVIDIA Container Toolkit installed"

echo "═══════════════════════════════════════════"
echo "STEP 3/4: Setting up Isaac ROS workspace"
echo "═══════════════════════════════════════════"
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src

# Clone Isaac ROS common tools
if [ ! -d "isaac_ros_common" ]; then
  git clone -b release-3.0 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
fi

# Clone Isaac ROS VSLAM
if [ ! -d "isaac_ros_visual_slam" ]; then
  git clone -b release-3.0 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
fi

echo "✅ Isaac ROS workspace ready at ~/workspaces/isaac_ros-dev"

echo "═══════════════════════════════════════════"
echo "STEP 4/4: DONE — Next steps:"
echo "═══════════════════════════════════════════"
echo ""
echo "⚠️  IMPORTANT: Log out and back in (or run: newgrp docker)"
echo "   so your user can run Docker without sudo."
echo ""
echo "Then run the Isaac ROS container:"
echo "  cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common"
echo "  ./scripts/run_dev.sh"
echo ""
echo "Inside the container, build VSLAM:"
echo "  cd /workspaces/isaac_ros-dev"
echo "  colcon build --symlink-install --packages-select isaac_ros_visual_slam"
echo ""
echo "Then launch with D455:"
echo "  ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py"
