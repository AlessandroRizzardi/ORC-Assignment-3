# ORC-Assignment-3
Assignment 3 of the Course Optimization-Based Robot Control held by [_Andrea Del Prete_](https://andreadelprete.github.io/) during Academic Year 2022-2023 ([_Department of Industrial Engineering_](https://www.dii.unitn.it/) (DII)).

## Introduction

The goal of this assignment is to implement the Deep Q-Learning algorithm with a simple continuous time (state-space) 1-pendulumn environment, using discrete control input space. The environment used for implementing this final project made use of [_Keras_](https://keras.io/) which is a framework built on top of [_TensorFlow_](https://www.tensorflow.org/).

## Description of files and folders contained in the repository
1.  [display.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/display.py)    : Connects to gepetto-viewer or webbrowser
2.  [pendulum.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/pendulum.py)   : Creates a continuous state simulation environment for a N-pendulum
3.  [dpendulum.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/dpendulum.py)  : Describes a continuous state pendulum environment with discrete control input
4.  [dqn.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/dqn.py)      : Creates a Deep-Q-Network learning agent
5.  [main_dqn.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/main_dqn.py): main file for the dqn algorithm applied to the environment pendulum
6.  [network_utils.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/network_utils.py)
7.  [plot_utils.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/plot_utils.py)
8.  [saved_models](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/saved_models)  : folder containing the model's weights got after the training of the NN (to save best performance of the algorithm)
9.  [Papers](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/Papers)       : folder containing the text, questions of the assignment and the scientific papers used for both the DQN algorithm implementation and for writing down the final report.
