'''
Implementation of the Deep Q Learning algorithm for a single/double pendulum 
'''
import tensorflow as tf
import numpy as np 
import random
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
from DQNtemplate import update
np_config.enable_numpy_behavior()
 
'''*** AUXILIARY FUNCTION (1) *** '''
def dyn_forNbigger_thanOne(env, u):
    # observe cost and next state
    if(env.nbJoint == 2):
        x_next, cost = env.step([u, env.c2du(0.0)])
    else: x_next, cost = env.step([u])
    
    return x_next, cost
'''*** END AUXILIARY FUNCTION (1)*** '''

'''*** AUXILIARY FUNCTION (2) *** '''
def action_selection(exploration_prob, env, x, model, target_model, eps_greedy=True): # action_selection for epsilon-greedy policy
    # with probability exploration_prob we take a random control input
    if(uniform() < exploration_prob and eps_greedy):
        u  = np.random.randint(0, env.nu)
    else: # otherwise take a greedy control
        x  = np.array([x]).T
        xu = np.reshape([np.append([x] * np.ones(env.nu), [np.arange(env.nu)])], (env.state_size() + 1, env.nu))
        if(not eps_greedy):
            u  = np.argmin(target_model(xu.T))
        else: 
            u  = np.argmin(model(xu.T))
    return u
''' *** END AUXILIARY FUNCTION (2) *** '''

def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx, nu): 
    ''' Create the neural network to represent the Q function '''
    inputs = keras.layers.Input(shape=(nx+nu,1))
    state_out1 = layers.Dense(16, activation="relu")(inputs)     # hidden layer 1
    state_out2 = layers.Dense(32, activation="relu")(state_out1) # hidden layer 2
    state_out3 = layers.Dense(64, activation="relu")(state_out2) # hidden layer 3
    state_out4 = layers.Dense(64, activation="relu")(state_out3) # hidden layer 4
    outputs = keras.layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs) # creates the NN

    return model

def dqn(env, gamma, nEpisodes, maxEpisodeLength, exploration_prob, \
        '''add min_buffer???''', '''replay_step???''', model, target_model, \
        minibatch_size, min_exploration_prob, exploration_decreasing_decay, \
        network_update_step, nprint=1000, plot = False, critic_optimizer):
    
    ''' 
        DQN learning algorithm input parameters description:
        env:                          environment
        model:                        Q network
        target_model:                 Q target
        gamma:                        discount factor
        nEpisodes:                    number of episodes to be used for evaluation
        maxEpisodeLength:             maximum length of an episode
        exploration_prob:             initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob:         lower bound of exploration probability
        compute_V_pi_from_Q:          function to compute V and pi from Q
        plot:                         if True plot the V table every nprint iterations
        nprint:                       print some info every nprint iterations
        network_update_step:          to complete !!!
    '''
    
    hist_x           = np.zeros([maxEpisodeLength, env.state_size()]) # list to keep track of state x
    h_ctg, hist_u = np.zeros(maxEpisodeLength)                     # lists to keep track of cost and control input u (for plotting)
    replay_buffer    = []                                             # # used for storing transitions and replaying experiences \ 
                                                                      # collected from interactions of the model with the environment

    # for every episode
    for k in range(nEpisodes):
        
        env.reset() # reset the state to rnd value
        
        J = 0 # initialize the cost-to-go at the beginning of each episode
        gamma_to_the_i = 1
        
        start = time.time()  # time of each episode
        
        #  START EPISODE
        step_count = 0
        for i in range(maxEpisodeLength):
            
            x = env.x # env state
            
            # selecting action
            u = action_selection(exploration_prob, env, x, model, target_model)
            
            #computing a step of the system dynamics
            x_next, cost = dyn_forNbigger_thanOne(env, u)
                
            #saving a transition into the replay_buffer  list *** Records an experience ***
            transition = (x, u, cost, x_next) # Experience
            replay_buffer.append(transition)

            # we update weight only if we have enough transitions into the buffer 
            if(len(replay_buffer) >= min_buffer): 
                '''*** Batch Sample step ***'''
                # Sampling a random set of transitions of a given dimension 'minibatch_size' of experience 
                # from replay_buffer for training the NN
                minibatch = random.sample(replay_buffer, minibatch_size) # nel caso mettiamo random.choices(...)
                x_batch, u_batch, cost_batch, x_next_batch = list(zip(*minibatch))  # loro hanno messo list(zip(*mini_batch))
                
                x_batch       = np.concatenate([x_batch], axis=1).T
                u_batch       = np.array(u_batch)
                xu_batch      = np.reshape(np.append(x_batch, u_batch), (env.state_size() + 1, minibatch_size))
                u_next_batch  = []
                
                # we now select the next action
                for j in range(minibatch_size):
                    u_next = action_selection(exploration_prob, env, x_next_batch[j], model, target_model, False)
                    u_next_batch.append(u_next)
                    
                # merge state and action of next step
                x_next_batch  = np.concatenate([x_next_batch], axis = 1).T
                xu_next_batch = np.reshape(np.append(x_next_batch, u_next_batch), (env.state_size() + 1, minibatch_size))
                cost_batch    = np.array(cost_batch)
                cost_batch    = np.reshape(cost_batch, minibatch_size)
                '''*** END Batch Sample step ***'''
                
                # CONVERSION From numpy array type to tensorflow tensor type
                xu_batch_tensor      = np2tf(xu_batch)
                xu_next_batch_tensor = np2tf(xu_next_batch)
                cost_batch_tensor    = np2tf(cost_batch)
                
                ''' ***** Update step (optimizer with SGD) ***** '''
                # update the weights of Q network using the provided batch of data
                update(xu_batch_tensor, cost_batch_tensor, xu_next_batch_tensor, \
                       model, target_model)
                
                # we update periodically the target model (Q_target) weights every 
                # or with a period of 'network_update_step' steps
                if(step_count % network_update_step == 0): 
                    target_model.set_weights(model.get_weights()) '''Update the current Q_target with the weight of Q'''
                ''' ***** END Update step (optimizer with SGD) ***** '''
            
            step_count += 1
            
            #keep track of the cost to go
            J += gamma_to_the_i * cost
            gamma_to_the_i *= gamma

        # END EPISODE 
        h_ctg.append(J)
            
        # update the exploration probability with an exponential decay: 
        # eps = exp(-decay*episode)
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay * k))
        elapsed_time = round((time.time() - start), 3)

        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(k % nprint == 0):
            print("Deep Q learning - Episode %d duration %.1f [s], Eps = %.1f, J = %.1f " % (k, elapsed_time, round(100*exploration_prob, 3), J) )
            if(plot):
                x, V, pi = compute_V_pi_from_Q(env, model, 20)
                env.plot_V_table(V, x[0], x[1])
                env.plot_policy(pi, x[0], x[1])
                plt.show()
                if(k == nprint):
                    hist_x, hist_u, h_ctg = render_greedy_policy(env, model, target_model, 0, None, maxEpisodeLength)
                    time_vec = np.linspace(0.0, maxEpisodeLength * env.pendulum.DT, maxEpisodeLength)
                    # plot_traj(time_vec, hist_x, hist_u, h_ctg, env) DA METTERE A POSTO !!!
                    plt.show()
                
    return h_ctg

