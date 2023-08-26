# ORC-Assignment-3
Assignment 3 of the Course Optimization-Based Robot Control held by [_Andrea Del Prete_](https://andreadelprete.github.io/) during Academic Year 2022-2023 ([_Department of Industrial Engineering_](https://www.dii.unitn.it/) (DII)).

## Introduction

The goal of this assignment is to implement the Deep Q-Learning algorithm with a simple continuous time (state-space) 1-pendulumn environment, using discrete control input space. The environment used for implementing this final project made use of [_Keras_](https://keras.io/) which is a framework built on top of [_TensorFlow_](https://www.tensorflow.org/).

## Description of files and folders contained in the repository

### Folders
1.  [saved_models](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/saved_models)  : Folder containing the model's weights got after the training of the NN (to save best performance of the algorithm)
2.  [Papers](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/Papers)       : Folder containing the text, questions of the assignment and the scientific papers used for both the DQN algorithm implementation and for writing down the final report.
3. [Tests_plots](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/Tests%20plots)        : Folder containing the different tests/simulations both for the single and double pendulum in order to try to get the optimal hyper-parameters for stabilizing the pendulum around the top position. 
4. [Videos](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/Videos)      : Folder containing the videos of both Single and Double Pendulum resulting in the best performance we could have been achieved.
5. [Prova](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/Prova)        : Folder containing some plots (only for plotting purposes during perfomed tests )
6. [src](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src)            : Folder containing the source code developed during the development of the DQN algorithm

### Files (source code)
1.  [display.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/display.py)    : Connects to gepetto-viewer or webbrowser
2.  [pendulum.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/pendulum.py)   : Creates a continuous state simulation environment for a N-pendulum
3.  [dpendulum.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/dpendulum.py)  : Describes a continuous state pendulum environment with discrete control input
4.  [dqn.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/dqn.py)      : Creates a Deep-Q-Network learning agent
5.  [main_dqn.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/main_dqn.py): main file for the dqn algorithm applied to the environment pendulum
6.  [network_utils.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/network_utils.py): Contains functions for creating the NN
7.  [plot_utils.py](https://github.com/AlessandroRizzardi/ORC-Assignment-3/tree/main/src/plot_utils.py): Contains functions for plotting, mainly V table, policy table and the trajectories of state and control


