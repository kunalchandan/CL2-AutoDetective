# CL2 AutoCausal
Plan:
AV Stack consists of the following stages:

1. Sensing
2. Perception
3. Planning 
4. Actuation

In simulations we have ground truth, however irl we do not. 

Are there some metrics we can use as a proxy to lay blame on the portion of the stack responsible?


## Setup

```
cd ~

git clone https://github.com/kunalchandan/CL2-AutoDetective.git autoDetective

cd autoDetective/

git clone https://github.com/BerkeleyLearnVerify/VerifAI.git

git clone https://github.com/carla-simulator/carla.git CarlaGit

git clone https://github.com/carla-simulator/ros-bridge.git

git clone https://github.com/ultralytics/yolov5.git

git clone https://github.com/BerkeleyLearnVerify/Scenic.git
```
Move the borrowed repositories out to let intellisense help me out.

```
# This might fail, just do this manually
mv VerifAI/ VerifAI-git
mv VerifAI-git/src/verifai/ ./
mv CarlaGit/PythonAPI/carla/ ./
cp -r carla/agents/ ./
cp -r Scenic/src/scenic/ ./
# Had to alter some imports for these
mv ros-bridge/carla_ad_agent/src/carla_ad_agent/ ./
```

Build environment.
```
poetry update

```
I hate poetry, it takes well over 15000 seconds ~= 4 HOURS to create, initialize and resolve dependancy issues for environment and it doesn't even work 😭😭😭😭
