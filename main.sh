#!/bin/bash

set -e
export PYTHONWARNINGS="ignore"

# -------------------- Expert Data Collection --------------------
# Running expert data collection pipeline to use it in Behavior Cloning
echo ""
echo "-------------------- [1/4] Expert Data Collection --------------------"
if [ ! -f "data/expert.npz" ]; then
    python -m core.env
else
    echo ">>> Expert data located at: 'data/expert.npz'"
    echo ""

fi

# -------------------- Behavior Cloning Training --------------------
# Initiating training for Behavior Cloning, using expert data
echo "-------------------- [2/4] Behavior Cloning Training --------------------"
python -m core.model.train

# -------------------- PPO Training --------------------
#  Executing localized training workflows for PPO, 
# loading Behavior Cloning weights for warm-start
echo ""
echo "-------------------- [3/4] PPO Training --------------------"
python -m src.train

# -------------------- Real-time Simulation --------------------
# Running final inference, utilizing the trained policy
echo ""
echo "-------------------- [4/4] Real-time Simulation --------------------"
python -m src.main