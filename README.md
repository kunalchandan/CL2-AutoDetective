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

git clone https://github.com/carla-simulator/carla.git

git clone https://github.com/carla-simulator/ros-bridge.git

git clone https://github.com/ultralytics/yolov5.git

git clone https://github.com/BerkeleyLearnVerify/Scenic.git
```

Build environment.
See (ENVIRONMENT.md document on how to setup the environment)[./ENVIRONMENT.md]

Carla setup:

Downloading carla should take ~30 minutes.
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.14.tar.gz

tar -xvzf CARLA_0.9.14.tar.gz -C FullCarla
```

I'm putting this on the second disk because we don't have enough space.
The path is

```
/media/e5_5044/OSDisk/carla/
```
