import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
from numpy.random import randint, uniform
np_config.enable_numpy_behavior()

def get_critic(nx, nu): 
    ''' Create the neural network to represent the Q function '''
    inputs = keras.layers.Input(shape=(nx+nu,))
    state_out1 = keras.layers.Dense(16, activation="relu")(inputs)     # hidden layer 1
    state_out2 = keras.layers.Dense(32, activation="relu")(state_out1) # hidden layer 2
    state_out3 = keras.layers.Dense(64, activation="relu")(state_out2) # hidden layer 3
    state_out4 = keras.layers.Dense(64, activation="relu")(state_out3) # hidden layer 4
    outputs = keras.layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs) # creates the NN

    return model


def action_selection(exploration_prob, env, x, model, eps_greedy=True): # action_selection for epsilon-greedy policy
    # with probability exploration_prob we take a random control input
    if(uniform() < exploration_prob and eps_greedy):
        u  = randint(0, env.nu)
    else: # otherwise take a greedy control
        xu = np.append(x*np.ones(env.nu),[np.arange(env.nu)],axis=0)
        u  = np.argmin(model(xu.T))
    return u

def dyn_forNbigger_thanOne(env, u):
    # observe cost and next state
    if(env.nbJoint == 2):
        x_next, cost = env.step([u, env.c2du(0.0)])
    else: x_next, cost = env.step([u])
    
    return x_next, cost

def update(xu_batch, cost_batch, xu_next_batch, Q, Q_target, DISCOUNT, optimizer):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = Q_target(xu_next_batch, training=True)   
        # Compute 1-step targets for the critic loss
        y = cost_batch + DISCOUNT*target_values                       
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables)) 


def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out