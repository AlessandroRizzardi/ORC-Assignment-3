from dpendulum import DPendulum
from dqn import dqn
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt 
import numpy as np
import time
from auxiliary_func import get_critic, trajectories, render_greedy_policy
np_config.enable_numpy_behavior()

if __name__ == '__main__':

    
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES               = 30                   # Number of training episodes
    NPRINT                  = 1                    # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 100                  # Max episode length
    DISCOUNT                = 0.99                 # Discount factor 
    PLOT                    = True                 # Plot stuff if True
    PLOT_TRAJ               = False                # Plot trajectories of state x and control input u together with the history of the cost
    BATCH_SIZE              = 32                   # size of the batch for replay buffer
    MIN_BUFFER              = 100                  # lower bound as start for sampling from buffer
    
    NX                      = 2         # number of states
    NU                      = 1         # number of control inputs
    
    # REPLAY_STEP           = 4         # to keep ???
    NETWORK_UPDATE_STEP     = 100       # how many steps taken for updating w
    QVALUE_LEARNING_RATE    = 1e-3      # alpha coefficient of Q learning algorithm
    exploration_prob                = 1     # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.05  # exploration decay for exponential decreasing
    min_exploration_prob            = 0.001 # minimum of exploration probability
    
    ### --- Pendulum Environment
    nbJoint = 1                # joints number
    nx = 41                     # number of discretization steops for the state
    nu = 41                    # number of discretization steps for the torque u
    
    # ----- FLAG to TRAIN/LOAD
    TRAINING                        = True # False = Load Model
    env = DPendulum(nbJoint, nu)

    # Creation of the Deep Q-Network models (create critic and target NNs)
    model = get_critic(NX, NU)                                         # Q network
    target_model = get_critic(NX, NU)                                  # Target network
    target_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE) # optimizer specifying the learning rates
    model.summary()
    
    if(TRAINING == True):
        print("\n\n\n###############################################")
        print("*** DEEP Q LEARNING ***")
        print("###############################################\n\n")
              
        h_ctg = dqn(env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH,\
                    exploration_prob, model, target_model, MIN_BUFFER,\
                    BATCH_SIZE, optimizer, NETWORK_UPDATE_STEP, min_exploration_prob,\
                    exploration_decreasing_decay, PLOT, NPRINT)
        plt.show()
   
     # save model and weights
        print("\nTraining finished")
        print("\nSave NN weights to file (in HDF5)")
        if (nbJoint == 1):
            model.save('saved_models/Model1')
            model.save_weights('saved_models/weight1.h5')
        else:    
            model.save('saved_models/Model2')
            model.save_weights('saved_models/weight2.h5')

        #plot cost
        plt.figure()
        plt.plot(np.cumsum(h_ctg) / range(1, NEPISODES + 1))
        plt.title ("Average cost-to-go")

    if(TRAINING == False): #load model
        print("\n\n\n###############################################")
        print("*** SAVING WEIGHTS FOR DEEP Q LEARNING ***")
        print("###############################################\n\n")
              
        print("Load NN weights from file\n")
        if (nbJoint == 1):
            model = tf.keras.models.load_model('saved_models/Model1')
        else:
            model = tf.keras.models.load_model('saved_models/Model2')
        assert(model)
          
    hist_x, hist_u, hist_cost = render_greedy_policy(env, model, DISCOUNT, None, MAX_EPISODE_LENGTH)
    
    if(PLOT_TRAJ):
        time_vec = np.linspace(0.0, MAX_EPISODE_LENGTH * env.pendulum.DT, MAX_EPISODE_LENGTH)
        trajectories(time_vec, hist_x, hist_u, hist_cost, env)

    plt.show()
        
    #for i in range(20):
    #    render_greedy_policy(env,model,DISCOUNT)





    
            

    
