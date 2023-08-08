'''
Implementation of the Deep Q Learning algorithm for a single/double pendulum 
'''

import tensorflow as tf
import numpy as np 
import random
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 

def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx, nu): # as the repo into dqn_agent.py
    ''' Create the neural network to represent the Q function '''
    inputs = keras.layers.Input(shape=(nx+nu,1))
    state_out1 = layers.Dense(16, activation="relu")(inputs)     # hidden layer 1
    state_out2 = layers.Dense(32, activation="relu")(state_out1) # hidden layer 2
    state_out3 = layers.Dense(64, activation="relu")(state_out2) # hidden layer 3
    state_out4 = layers.Dense(64, activation="relu")(state_out3) # hidden layer 4
    outputs = keras.layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs) # creates the NN

    return model


def dqn(env, gamma, nEpisodes, maxEpisodeLength, exploration_prob, replay_step, model, target_model, minibatch_size,
         min_exploration_prob, exploration_decreasing_decay, network_update_step, nprint=1000, optimizer):
    
    ''' 
        DQN learning algorithm:
        buffer: replay buffer # used for storing and replaying experiences collected from interactions of the agent with the environment
        env: environment 
        gamma: discount factor
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    h_ctg = []  #list containing cost-to-go history
    replay_buffer = []  #buffer in which we save each transition
    step_count = 0

    # for every episode
    for k in range(nEpisodes):
        
        env.reset() # reset the state to rnd value
        
        J = 0 #initialize the cost-to-go at the beginning of each episode
        gamma_to_the_i = 1
        
        start = time.time()  # time of each episode #aggiunta dalla repo
        
        #  START EPISODE
        for i in range(maxEpisodeLength):
            
            x = env.x
            
            # selecting policy using ϵ-greedy strategy (ϵ-greedy policy)       
            if np.random.uniform() < exploration_prob:
                u = np.random.randint(0, env.nu) # with probability exploration_prob take a random control input
                
            '''
            else: #otherwise we take a greedy control so we take the action minimizing the Q function
                x = np.reshape(x,(env.state_size(),1))
                xu = np.append(x*np.ones(env.nu), [np.arange(env.nu)], axis=0)
                u = np.argmin(model(xu.T)) # Greedy action
                #u = np.argmax(target_model(xu) + np.random.randn(1,NU)/episode) # Greedy action with noise
            '''
            
            #computing a step of the system's dynamics (check for lines 105-108)
            #observe cost and next state (step = calculate dynamics)
            if (env.nbJoint == 2):
                x_next, cost = env.step([u,env.c2du(0.0)])
            else: x_next, cost = env.step([u])
                
            #saving a transition into the replay buffer *** Records an experience ***
            transition = (x, u, cost, x_next) #experience
            replay_buffer.append(transition)

            # we update weight only if we have enough transitions  into the buffer and every 'replay_step' number of step
            if len(replay_buffer) >= MIN_BUFFER '''(era minibatch_size)''' ): #c'era anche nel NOSTRO (and i % replay_step == 0):
                
                '''*** batch sample ***'''
                # Sampling a random set of transitions of a given dimension 'minibatch_size' of experience from replay_bufferfor training the NN
                # sample_batch method (replay_buffer.py)
                minibatch = random.sample(replay_buffer, minibatch_size) # nel caso mettiamo .choices(argomenti vanno bene)
                x_batch, u_batch, cost_batch, x_next_batch = list(zip(*minibatch))  # loro hanno messo list(zip(*mini_batch))
                
                *** forse da aggiungere ***
                x_batch = np.concatenate([x_batch], axis=1).T
                u_batch       = np.asarray(u_batch)
                *** forse da aggiungere ***
                
                # the neural network needs an input of dimension (number of state + number of action), so we merge action and state
                #x_batch = np.reshape(x_batch,(env.state_size(),minibatch_size))
                xu_batch = np.reshape(np.append(x_batch, u_batch), (env.state_size() + 1, minibatch_size))

                u_next_batch = [] # control get_action in the repo dqn_agent.py
                # we now select the next action
                for j in range(minibatch_size):
                    x = np.array([x_next_batch[j]]).T
                    xu = np.reshape([np.append([x_next_batch[j]]*np.ones(env.nu), [np.arange(env.nu)])], (env.state_size()+1, env.nu))
                    u_next = np.argmin(target_model(xu.T))
                    u_next_batch.append(u_next)

                # merge state and action of next step
                x_next_batch = np.concatenate([x_next_batch], axis = 1).T
                                
                #x_next_batch = np.reshape(x_next_batch,(env.state_size()+1,minibatch_size))
                xu_next_batch = np.reshape(np.append(x_next_batch, u_next_batch), (env.state_size() + 1, minibatch_size))
                cost_batch = np.array(cost_batch)
                cost_batch = np.reshape(cost_batch, minibatch_size)

                # we need to change variable type from numpy array type to tensorflow tensor type
                xu_batch_tensor      = np2tf(xu_batch)
                xu_next_batch_tensor = np2tf(xu_next_batch)
                cost_batch_tensor    = np2tf(cost_batch)
                
                # update (optimizer with SGD)
                with tf.GradientTape() as tape:     #ok come la repo
                    # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
                    # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
                    # Tensors can be manually watched by invoking the watch method on this context manager.
                    target_values = target_model(xu_next_batch_tensor, training=True) 

                    # Compute 1-step targets for the critic loss
                    y = cost_batch_tensor + gamma * target_values

                    # Compute batch of Values associated to the sampled batch of states
                    Q_value = model(xu_batch_tensor, training=True)         

                    # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
                    Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  

                # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
                Q_grad = tape.gradient(Q_loss, model.trainable_variables)        
                # Update the critic backpropagating the gradients
                optimizer.apply_gradients(zip(Q_grad, model.trainable_variables))  

                # we update periodically the target model weights every or with a period of 'network_update_step' steps
                if step_count % network_update_step == 0: 
                    target_model.set_weights(model.get_weights()) '''Update the current Q_target with the weight of Q'''
            
            step_count += 1
            
            #keep track of the cost to go
            J += gamma_to_the_i * cost
            gamma_to_the_i *= gamma

        # END EPISODE 
        h_ctg.append(J)
            
        # update the exploration probability with an exponential decay: 
        # eps = exp(-decay*episode)
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay*k))
        elapsed_time = round((time.time() - start),3)

        print("Deep Q learning - Episode %d duration %.1f [s], Eps = %.1f, J = %.1f " % (k, elapsed_time, round(100*exploration_prob, 3), J) )

        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(k % nprint == 0 and  k >= nprint):
            hist_X, hist_U, hist_cst = render_greedy_policy(env, model, 0, None, maxEpisodeLength)
            if(plot):
                time_vec = np.linspace(0.0, maxEpisodeLength * env.pendulum.DT, maxEpisodeLength)
                plot_traj(time_vec, hist_X, hist_U, hist_cost_sim, env)
                plt.show()
        if(k % nprint == 0):
            print('Episode #%d done with cost %d and %.1f exploration prob' % (k, J, 100*exploration_prob))            

    return h_ctg

