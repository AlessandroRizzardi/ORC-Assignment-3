# ORC-Assignment-3
Assignment n°3 of the Course Optimization-Based Robot Control held by [_Andrea Del Prete_](https://andreadelprete.github.io/) during Academic Year 2022-2023 ([_Department of Industrial Engineering_](https://www.dii.unitn.it/) (DII)).

## Introduction

The goal of this assignment is to implement the Deep Q-Learning algorithm with a simple continuous time (state-space) 1-pendulumn environment, using discrete control input space. The environment used for implementing this final project made use of [_Keras_](https://keras.io/) which is a framework built on top of [_TensorFlow_](https://www.tensorflow.org/).

## Description of files and folders contained in the repository
1. auxiliary_func: File containing code for creating a Neural Network using the tensor flow library, plotting of trajectories and for implementing the DQN algorithm and for dealing with a Pendulum with more than 1 joint
2. display.py    : Connects to gepetto-viewer or webbrowser
3. pendulum.py   : Creates a continuous state simulation environment for a N-pendulum
4. dpendulum.py  : Describes a continuous state pendulum environment with discrete control input
5. dqn.py        : Creates a Deep-Q-Network learning agent
6. main_dqn.py   : main file for the dqn algorithm applied to the environment pendulum
7. saved_models  : folder containing the model's weights got after the training of the NN (to save best performance of the algorithm)
8. Papers        : folder containing the text, questions of the assignment and the scientific papers used for both the DQN algorithm implementation and for writing down the final report.
