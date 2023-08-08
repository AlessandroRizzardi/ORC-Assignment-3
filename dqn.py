'''
Implementation of the Deep Q Learning algorithm for a single/double pendulum 
'''

import tensorflow as tf
import numpy as np 
import random
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 

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
    state_out1 = keras.layers.Dense(16, activation="relu")(inputs) 
    state_out2 = keras.layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = keras.layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = keras.layers.Dense(64, activation="relu")(state_out3)
    outputs = keras.layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs)

    return model


def dqn(env, gamma, nEpisodes, maxEpisodLength, exploration_prob,replay_step, model, target_model, minibatch_size,
         min_exploration_prob, exploration_decrasing_decay, network_update_step, nprint,optimizer):

    h_ctg = []  #list containing cost-to-go history
    replay_buffer = []  #buffer in which we save each transition
    step_count = 0

    for episode in range(1,nEpisodes):
        x = env.reset()
        cost_to_go = 0.0
        gamma_i = 1

        #  START EPISODE
        for step in range(maxEpisodLength):
            
            x = env.x

            # selectin policy using ϵ-greedy startegy        
            if np.random.uniform(0,1) < exploration_prob:
                u = np.random.randint(env.nu)
            else:
                x = np.reshape(x,(env.state_size(),1))
                xu = np.append(x*np.ones(env.nu), [np.arange(env.nu)], axis=0)
                u = np.argmax(model(xu.T)) # Greedy action
                #u = np.argmax(target_model(xu) + np.random.randn(1,NU)/episode) # Greedy action with noise

            #computing a step of the system's dynamics
            x_next, cost = env.step([u])

            #saving a trsition into the replay buffer
            transition = (x, u, cost, x_next)
            replay_buffer.append(transition)

            # we update weight only if we have enough transition into the buffer and every 'repaly_step' number of step
            if len(replay_buffer) >= minibatch_size and step % replay_step == 0:

                #sampling a random set of transition of a given dimension 'minibatch_size'
                minibatch = random.sample(replay_buffer, minibatch_size)
                x_batch, u_batch, cost_batch, x_next_batch = zip(*minibatch)

                # the neural network need an input of dimension (number of state + number of action), so we merge action and state
                x_batch = np.reshape(x_batch,(env.state_size(),minibatch_size))
                xu_batch = np.append(x_batch, [u_batch],axis=0)

                # altra opzione: x_batch  = np.concatenate([x_bacth],axis=1).T
                
                u_next_batch = []
                # we now select the next action
                for i in range(minibatch_size):
                    x = np.reshape(x_next_batch[i],(env.state_size(),1))
                    xu = np.append(x*np.ones(env.nu), [np.arange(env.nu)], axis=0)
                    u_next = np.argmax(target_model(xu.T))
                    u_next_batch.append(u_next)

                # merge state and action of next step
                x_next_batch = np.reshape(x_next_batch,(env.state_size(),minibatch_size))
                xu_next_batch = np.append(x_next_batch, [u_next_batch],axis=0)

                cost_batch = np.array(cost_batch)
                cost_batch = np.reshape(cost_batch,(1,minibatch_size))

                # we need to change variable typr from numpy array type to tensorflow tensor type
                xu_batch_tensor = np2tf(xu_batch)
                xu_next_batch_tensor = np2tf(xu_next_batch)
                cost_batch_tensor = np2tf(cost_batch)

                with tf.GradientTape() as tape:     
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


                # we update the target model weights every 'network_update_step' steps
                if step_count % network_update_step == 0:
                    target_model.set_weights(model.get_weights())
            
            step_count += 1

            #print("Print di check sul costo:",round(cost,3))
            cost_to_go += gamma_i*cost
            gamma_i *= gamma
            #print("Check",step)

        # END EPISODE 
        
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decrasing_decay*episode))
        
        h_ctg.append(cost_to_go)

        if not episode%nprint:
            print('Episode #%d done with cost %d and %.1f exploration prob' % (episode, cost_to_go, 100*exploration_prob))            

    return h_ctg

