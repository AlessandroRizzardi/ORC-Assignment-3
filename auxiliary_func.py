import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
import time

'''*** AUXILIARY FUNCTION (1) *** '''
def dyn_forNbigger_thanOne(env, u):
    # observe cost and next state
    if(env.nbJoint == 2):
        x_next, cost = env.step([u, env.c2du(0.0)])
    else: x_next, cost = env.step([u])
    
    return x_next, cost
'''*** END AUXILIARY FUNCTION (1)*** '''

'''*** AUXILIARY FUNCTION (2) *** '''
def action_selection(exploration_prob, env, x, model, eps_greedy=True): # action_selection for epsilon-greedy policy
    # with probability exploration_prob we take a random control input
    if(np.random.uniform() < exploration_prob and eps_greedy):
        u  = np.random.randint(0, env.nu)
    else: # otherwise take a greedy control
        xu = np.append(x*np.ones(env.nu),[np.arange(env.nu)],axis=0)
        u  = np.argmin(model(xu.T))
    return u
''' *** END AUXILIARY FUNCTION (2) *** '''

'''*** AUXILIARY FUNCTION (3)***'''
def update(xu_batch, cost_batch, xu_next_batch, Q, Q_target,DISCOUNT,optimizer):
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
'''*** END AUXILIARY FUNCTION (3)***'''

'''*** AUXILIARY FUNCTION (4)***'''
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
'''*** END AUXILIARY FUNCTION (4)***'''
    
'''*** AUXILIARY FUNCTION (5)***'''
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()
'''*** END AUXILIARY FUNCTION (5)***'''

'''*** AUXILIARY FUNCTION (6)***'''
def compute_V_pi_from_Q(env, model, plot_discretization=30):
    ''' Compute Value table and greedy policy pi from Q table. '''
    
    vMax = env.vMax
    dq   = 2*np.pi/plot_discretization
    dv   = 2*vMax/plot_discretization
    nx   = env.state_size()
    
    V  = np.zeros((plot_discretization+1, plot_discretization+1))
    pi = np.zeros((plot_discretization+1, plot_discretization+1))
    x  = np.zeros((nx, plot_discretization+1))

    x[0,:] = np.arange(-np.pi, np.pi + dq, dq)
    x[1,:] = np.arange(-vMax, vMax + dv, dv)
                       
    for q in range(plot_discretization+1):
        for v in range(plot_discretization+1):
            xu      = np.reshape([x[0,q] * np.ones(env.nu), x[1,v] * np.ones(env.nu), np.arange(env.nu)], (nx + 1, env.nu))
            V[q,v]  = np.min(model(xu.T))
            pi[q,v] = env.d2cu(np.argmin(model(xu.T)))
            
    return x, V, pi
'''*** END AUXILIARY FUNCTION (6)***'''

'''*** AUXILIARY FUNCTION (7)***'''
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
'''*** END AUXILIARY FUNCTION (7)***'''

'''*** AUXILIARY FUNCTION (8)***'''
def render_greedy_policy(env, model, target_model, gamma, x0=None, maxIter=90):
    ''' ***** render_greedy_policy *****
    Simulates the system using the policy I just computed:
    render_greedy_policy initializes the pendulum with a random initial state and then
    simulates it with the policy we computed
    '''
    x0 = env.reset(x0)
    x = x0
    costToGo = 0.0
    gamma_to_the_i = 1
    env.render()
    time.sleep(1) 

    # storing the histories over time of x, u and the cost in 3 lists
    hist_x   = np.zeros([maxIter, env.state_size()])
    hist_u   = []
    hist_cost = []

    for i in range(maxIter):
        '''# selecting policy using ϵ-greedy strategy (ϵ-greedy policy)' or random innput'''     
        u = action_selection(0, env, x, model, target_model)
        
        x, cost = dyn_forNbigger_thanOne(env, u)

        costToGo += gamma_to_the_i * cost
        gamma_to_the_i *= gamma

        hist_x[i,:]  = np.concatenate(x,axis=1).T
        hist_u.append(env.d2cu(u))
        hist_cost.append(cost)

        env.render() 

    print("Real cost-to-go of state x0,", x0, "=", costToGo)   

    return hist_x, hist_u, hist_cost

'''*** END AUXILIARY FUNCTION (8)***'''