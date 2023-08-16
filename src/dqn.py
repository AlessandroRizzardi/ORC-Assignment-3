'''
Implementation of the Deep Q Learning algorithm for a single/double pendulum 
'''
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from network_utils import *
from plot_utils import *
from collections import deque
np_config.enable_numpy_behavior()
      
def dqn(env, gamma, nEpisodes, maxEpisodeLength, \
        exploration_prob, model, target_model, min_buffer,\
        minibatch_size, optimizer, network_update_step, min_exploration_prob,\
        exploration_decreasing_decay, replay_step, capacity_buffer, nprint = 1000):
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
        network_update_step:          number of steps used to update the model's weights
        '''
        
    replay_buffer = deque(maxlen=capacity_buffer)                         # used for storing transitions and replaying experiences \ 
                                                              # collected from interactions of the model with the environment

    hist_cost = []                                          # list to keep track of cost
    step_count = 0

    # for every episode
    for k in range(nEpisodes):
        
        env.reset() # reset the state to rnd value
        
        J = 0 # initialized the cost-to-go at the beginning of each episode
        gamma_i = 1
        
        start = time.time()  # time of each episode
        
        #  START EPISODE
        for i in range(maxEpisodeLength):
            
            x = env.x # env state
            
            # selecting action
            u = action_selection(exploration_prob, env, x, model)
            
            #computing a step of the system dynamics
            x_next, cost = dyn_forNbigger_thanOne(env, u)
            
            #saving a transition into the replay_buffer  list *** Records an experience ***
            transition = (x, u, cost, x_next) # Experience (S, A, R, S', A')
            replay_buffer.append(transition)

            #### MODEL UPDATE
            # we update weight only if we have enough transitions into the buffer 
            if(len(replay_buffer) >= min_buffer and step_count % replay_step == 0): # check performance & with (replay_step % 4 == 0) 
                
                '''*** Batch Sample step ***'''
                # Sampling a random set of transitions of a given dimension 'minibatch_size' of experience 
                # from replay_buffer for training the NN
                minibatch = random.choices(replay_buffer, k = minibatch_size) 
                x_batch, u_batch, cost_batch, x_next_batch = list(zip(*minibatch))  
                u_next_batch  = np.zeros(minibatch_size) # np.empty

                x_batch  = np.concatenate([x_batch],axis=1).T
                u_batch  = np.asarray(u_batch)
                xu_batch = np.reshape(np.append(x_batch, u_batch),(env.state_size() + 1, minibatch_size))

                # we now select the next action
                for j in range(minibatch_size):
                    u_next_batch[j] = action_selection(exploration_prob, env, x_next_batch[j], target_model, eps_greedy=False)
                    
                # merge state and action of next step
                x_next_batch  = np.concatenate([x_next_batch], axis = 1).T
                xu_next_batch = np.reshape(np.append(x_next_batch, u_next_batch),(env.state_size() + 1, minibatch_size))

                cost_batch    = np.asarray(cost_batch)
                cost_batch    = np.reshape(cost_batch, (minibatch_size))
                '''*** END Batch Sample step ***'''
                
                # CONVERSION From numpy array type to tensorflow tensor type
                xu_batch_tensor      = np2tf(xu_batch)
                xu_next_batch_tensor = np2tf(xu_next_batch)
                cost_batch_tensor    = np2tf(cost_batch)
                
                ''' ***** Update step (optimizer with SGD) ***** '''
                # update the weights of Q network using the provided batch of data
                update(xu_batch_tensor, cost_batch_tensor, xu_next_batch_tensor, model, target_model, gamma, optimizer)
                
                # we update periodically the target model (Q_target) weights every 
                # or with a period of 'network_update_step' steps
                if(step_count % network_update_step == 0): 
                    target_model.set_weights(model.get_weights())  # Update the current Q_target with the weight of Q
                ''' ***** END Update step (optimizer with SGD) ***** '''

                        
            #keep track of the cost to go
            J += gamma_i * cost # cost-to-go
            gamma_i *= gamma

            step_count += 1
        # END EPISODE 
        
        hist_cost.append(J)
            
        # update the exploration probability with an exponential decay: 
        # eps = exp(-decay*episode)
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay * k))
        elapsed_time     = round((time.time() - start), 3)  


        #use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(k % nprint == 0):
            # printing the training each nprint episodes
            print("Deep Q learning - Episode %d duration %.1f [s], Eps = %.1f, J = %.1f " % (k, elapsed_time, round(100*exploration_prob, 3), J) )  
            
    return hist_cost


