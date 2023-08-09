'''
Implementation of the Deep Q Learning algorithm for a single/double pendulum 
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np 
import random
import time
from DQNtemplate import update
import matplotlib.pyplot as plt
from auxiliary_func import *
np_config.enable_numpy_behavior()
      
def dqn(env, gamma, nEpisodes, maxEpisodeLength, \
        exploration_prob, model , target_model,min_buffer,\
        minibatch_size,optimizer, network_update_step,min_exploration_prob,\
        exploration_decreasing_decay, PLOT=False, nprint = 1000):
    
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
        
    h_cost = []                                         # lists to keep track of cost and control input u (for plotting)
    hist_x           = np.zeros([maxEpisodeLength, env.state_size()]) # list to keep track of state x
    hist_u = np.zeros(maxEpisodeLength)
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
            u = action_selection(exploration_prob, env, x, model)
            
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
                minibatch = random.choices(replay_buffer, k = minibatch_size) 
                x_batch, u_batch, cost_batch, x_next_batch = list(zip(*minibatch))  
                
                x_batch       = np.concatenate([x_batch], axis=1).T
                u_batch       = np.asarray(u_batch)
                xu_batch      = np.reshape(np.append(x_batch, u_batch), (env.state_size() + 1, minibatch_size))
                u_next_batch  = []
                
                # we now select the next action
                for j in range(minibatch_size):
                    u_next = action_selection(exploration_prob, env, x_next_batch[j], target_model, False)
                    u_next_batch.append(u_next)
                    
                # merge state and action of next step
                x_next_batch  = np.concatenate([x_next_batch], axis = 1).T
                xu_next_batch = np.reshape(np.append(x_next_batch, u_next_batch), (env.state_size() + 1, minibatch_size))
                cost_batch    = np.asarray(cost_batch)
                cost_batch    = np.reshape(cost_batch, (1,minibatch_size))
                '''*** END Batch Sample step ***'''
                
                # CONVERSION From numpy array type to tensorflow tensor type
                xu_batch_tensor      = np2tf(xu_batch)
                xu_next_batch_tensor = np2tf(xu_next_batch)
                cost_batch_tensor    = np2tf(cost_batch)
                
                ''' ***** Update step (optimizer with SGD) ***** '''
                # update the weights of Q network using the provided batch of data
                update(xu_batch_tensor, cost_batch_tensor, xu_next_batch_tensor, model, target_model, gamma ,optimizer)
                
                # we update periodically the target model (Q_target) weights every 
                # or with a period of 'network_update_step' steps
                if(step_count % network_update_step == 0): 
                    target_model.set_weights(model.get_weights())  # Update the current Q_target with the weight of Q
                ''' ***** END Update step (optimizer with SGD) ***** '''
            
            step_count += 1
            
            #keep track of the cost to go
            J += gamma_to_the_i * cost #cost-to-go
            gamma_to_the_i *= gamma

        # END EPISODE 
        h_cost.append(J)
            
        # update the exploration probability with an exponential decay: 
        # eps = exp(-decay*episode)
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay * k))
        elapsed_time = round((time.time() - start), 3)

        if not k % nprint:
            print('Episode #%d done with cost %d and %.1f exploration prob' % (k, J, 100*exploration_prob))            


        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
#        if(k % nprint == 0):
#            print("Deep Q learning - Episode %d duration %.1f [s], Eps = %.1f, J = %.1f " % (k, elapsed_time, round(100*exploration_prob, 3), J) )
#            if(PLOT):
#                x, V, pi = compute_V_pi_from_Q(env, model, 20)
#                env.plot_V_table(V, x[0], x[1])
#                env.plot_policy(pi, x[0], x[1])
#                plt.show()
#                if(k == nprint):
#                    hist_x, hist_u, h_ctg = render_greedy_policy(env, model, target_model, 0, None, maxEpisodeLength)
#                    time_vec = np.linspace(0.0, maxEpisodeLength * env.pendulum.DT, maxEpisodeLength)
#                    # plot_traj(time_vec, hist_x, hist_u, h_ctg, env) DA METTERE A POSTO !!!
#                    plt.show()
                
    return h_cost

