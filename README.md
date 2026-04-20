# Chameleon

End-to-end sim-to-real pipeline for robotic manipulation. Trained entirely in simulation, deploying on a physical robot arm.

## What it does

Trains the SO-100 arm to pick up a cube and place it into a cup. No human demonstrations required.

## Pipeline

1. **Reinforcement learning in simulation**: PPO policy trained with domain randomization
2. **Automated dataset collection**: 500 successful rollouts exported as a [LeRobot-compatible dataset](https://huggingface.co/datasets/andaloo23/so100_pick_place_pi05)
3. **Vision-language-action model fine-tuning**: PI0.5 fine-tuned on collected episodes to produce a real-robot-ready policy *(in progress)*
4. **Real robot deployment**: coming soon