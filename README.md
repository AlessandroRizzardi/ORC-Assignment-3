# ORC-Assignment-3
Assignment 3 of the course Optimization-Based Robot Control

## Introduction

The goal of this assignment is to implement the Deep Q-Learning algorithm with a simple continuous time (state-space) 1-pendulumn environment, using discrete control input space. The environment used for implementing this final project made use of [_Keras_](https://keras.io/) which is a framework built on top of [_TensorFlow_](https://www.tensorflow.org/).

## Folder organization
* DQNtemplate.py: template for implementing the Deep-Q-Network
* display.py: connect to gepetto-viewer or webbrowser
* pendulum.py: Create a continuous state simulation environment for a N-pendulum
* dpendulum.py: Describe continuous state pendulum environment with discrete control input (derived from pendulumn.py)
* dqn.py: create a Deep-Q-Network learning agent using the tensorflow library
* main_dqn.py: main file for the dqn algorithm applied to a single pendulum environment
