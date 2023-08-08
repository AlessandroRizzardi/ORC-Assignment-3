# ORC-Assignment-3
Assignment 3 of the course Optimization-Based Robot Control

## Introduction

The goal of this assignment is to implement the Deep Q-Learning algorithm with a simple continuous time (state-space) 1-pendulumn environment, using discrete control input space.

## Setup the enviroment

Open the terminal and execute the following commands:

```
sudo apt install python3-numpy python3-scipy python3-matplotlib curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

sudo apt-get update
```

If you are using Ubuntu 20.04 run:

```
sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

If you are using Ubuntu 22.04 run:

```
sudo apt install robotpkg-py310-pinocchio robotpkg-py310-example-robot-data robotpkg-urdfdom robotpkg-py310-qt5-gepetto-viewer-corba robotpkg-py310-quadprog robotpkg-py310-tsid
```

Configure the environment variables by adding the following lines to your file ~/.bashrc

To add the following command you can use the nano editor or the gedit editor:
* nano ~/.bashrc
* gedit ~/.bashrc

```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_dqn>
```

where <folder_containing_dqn> is the folder containing the "dqn_pendulumn" folder, which in turns contains all the python code of this class.

## Folder organization
* DQNtemplate.py: template for implementing the Deep-Q-Network
* display.py: connect to geppetto-viewer or webbrowser
* pendulumn.py: Create a continuous state simulation environment for a N-pendulum
* dpendulumn.py: Describe continuous state pendulum environment with discrete control input (derived from pendulumn.py)
* dqn.py: create a deep-q-network learning agent using the tensorflow library
* main_dqn.py: main file for the dqn algorithm applied to a single pendulumn enviroment
