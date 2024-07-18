#!/bin/bash
set -e # Exit script immediately on first error.

# Script Name: setup-client-vm-v2.sh
# Description: This script is used to setup the client VM for the IOT example in the FEDn framework.
# Author: Atefeh Aramian
# Date: 2024-07-02
# Usage: ./setup-client-vm-v2.sh

###### DOCKER ######

# Create docker cache directory
echo "[SCRIPTLOG] Creating docker cache directory"
mkdir ~/volume/docker


# Add docker config
echo "[SCRIPTLOG] Adding docker config"
sudo touch /etc/docker/daemon.json

# Add docker config data-root
echo "[SCRIPTLOG] Adding docker config data-root"
echo '{"data-root": "/home/ubuntu/volume/docker"}' | sudo tee /etc/docker/daemon.json

# Restart docker
echo "[SCRIPTLOG] Restarting docker"
sudo systemctl restart docker

####### PIP #######
#install pip
echo "[SCRIPTLOG] Installing pip"
sudo apt-get install python3-pip
#install venv
echo "[SCRIPTLOG] Installing venv"
sudo apt-get install python3-venv
# Install ipykernel
echo "[SCRIPTLOG] Installing ipykernel"
pip install ipykernel

# Create venv
echo "[SCRIPTLOG] Creating .IOTH venv"
python3 -m venv .IOTH

# Create cache folder for pip
echo "[SCRIPTLOG] Creating cache folder for pip"
mkdir ~/volume/.cache/
mkdir ~/volume/.cache/pip

# Append directory to venv activate file
echo "[SCRIPTLOG] Appending directory to venv activate file"
echo "export PIP_CACHE_DIR=/home/ubuntu/volume/.cache/pip" >> ~/volume/fedn/examples/IOT/.IOTH/bin/activate


# source venv
echo "[SCRIPTLOG] Sourcing venv"
source ~/volume/fedn/examples/IOT/.IOTH/bin/activate

# Install packages
echo "[SCRIPTLOG] Installing packages from requirements.txt"
pip install -r ~/volume/fedn/examples/IOT/requirements.txt


# Add your virtualenv as a kernel
echo "[SCRIPTLOG] Adding virtualenv as a kernel"
python -m ipykernel install --user --name=.IOTH --display-name "IOTH"


# Install fedn package
echo "[SCRIPTLOG] Installing fedn 0.9.0 package"
pip install fedn==0.9.0

# Create client.yaml file
echo "[SCRIPTLOG] Creating client.yaml file"
touch client.yaml

# Add client.yaml configs
echo "[SCRIPTLOG] Adding client.yaml configs"
echo "network_id: fedn-network" >> ~/volume/fedn/examples/IOT/client.yaml
echo "discover_host: 192.168.2.56" >> ~/volume/fedn/examples/IOT/client.yaml
echo "discover_port: 8092" >> ~/volume/fedn/examples/IOT/client.yaml


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait