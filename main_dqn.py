from dpendulum import DPendulum
from dqn import dqn
from network_utils import *
from plot_utils import *
import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

if __name__ == '__main__':
    
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES               = 3000          # Number of training episodes
    NPRINT                  = 5               # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 200            # Max episode length
    DISCOUNT                = 0.99            # Discount factor 
    MINIBATCH_SIZE          = 64             # size of the batch for replay buffer
    MIN_BUFFER              = 300            # lower bound as starting point for sampling from buffer
    REPLAY_STEP             = 4                 
    NETWORK_UPDATE_STEP     = 800             # how many steps taken for updating w
    QVALUE_LEARNING_RATE    = 1e-3            # alpha coefficient of Q learning algorithm
    CAPACITY_BUFFER         = 20000
    exploration_prob                = 1       # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.001    # exploration decay for exponential decreasing
    min_exploration_prob            = 0.01   # minimum of exploration probability
    
    PLOT_TRAJ               = False            # Plot trajectories of state x and control input u together with the history of the cost
    ### --- Pendulum Environment
    nbJoint                   = 2             # joints number
    nu                        = 31           # number of discretization steps for the torque u
    state_discretization_plot = 41            # number of discretization steops for the state x
    
    NX                        = 2*nbJoint             # number of states
    NU                        = 1             # number of control inputs
    # ----- FLAG to TRAIN/LOAD
    TRAINING                        = False # False = Load Model

    #----- MODEL NAME FOLDER FOR SAVING PLOTS
    # for just testing the code for trainig put "Prova" as modelname, while to load a model insert one of the name in folder "saved models"
    model_name = "Model_double10"   # Model_#  Model_double_# 

    # creation of pendulum environment
    env = DPendulum(nbJoint, nu, vMax=4,uMax=10)

    # Creation of the Deep Q-Network models (create critic and target NNs)
    model = get_critic(NX, NU)                                         # Q network
    target_model = get_critic(NX, NU)                                  # Target network
    target_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)         # optimizer specifying the learning rates
    model.summary()
    
    if(TRAINING == True):
        print("\n\n\n###############################################")
        print("*** DEEP Q LEARNING ***")
        print("###############################################\n\n")

        start = time.time() #start training time

        # Deep-Q_Network algorithm  
        h_ctg = dqn(env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH,\
                    exploration_prob, model, target_model, MIN_BUFFER,\
                    MINIBATCH_SIZE, optimizer, NETWORK_UPDATE_STEP, min_exploration_prob,\
                    exploration_decreasing_decay, REPLAY_STEP, CAPACITY_BUFFER, NPRINT)
        plt.show()
        
        end = time.time() #end training time
        
        Time = round((end-start)/60,3)
        print("Training time:", Time ," min")


        # save model and weights
        print("\nTraining finished")
        print("\nSave NN weights to file (in HDF5)")
        
        model.save('saved_models/'+model_name)
        model.save_weights('saved_models/'+model_name+'_weights.h5')
        

        #plot cost
        plt.figure()
        plt.plot(np.cumsum(h_ctg) / range(1, NEPISODES + 1))
        plt.plot(range(1, NEPISODES + 1), 0 * np.ones(NEPISODES), "m--", alpha=0.8, linewidth=2)
        plt.xlabel('Episodes')
        plt.title ("Average cost-to-go")
        plt.savefig(model_name+"/HCostToGo.png")     
        plt.show()

        
    else: #load model
        print("\n\n\n###############################################")
        print("*** LOADING WEIGHTS FOR DEEP Q LEARNING ***")
        print("###############################################\n\n")
              
        print("Loading NN weights from file...\n")
        
        model = tf.keras.models.load_model('saved_models/'+ model_name)
        
          
    if(nbJoint==1):
        x,V,pi = compute_V_pi_from_Q(env, model, state_discretization_plot)
        env.plot_V_table(V,x[0],x[1],"Prova")
        env.plot_policy(pi,x[0],x[1],"Prova")
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) # used to describe how far i am from the optimal policy and optimal value function using Deep Q learning

    hist_x, hist_u, hist_cost = render_greedy_policy(env, model, DISCOUNT, None, MAX_EPISODE_LENGTH)
    
    if(PLOT_TRAJ):
        time = np.linspace(0.0, MAX_EPISODE_LENGTH * env.pendulum.DT, MAX_EPISODE_LENGTH)
        trajectories(time, hist_x, hist_u, hist_cost, env,"Prova")

    plt.show()

    for i in range(10):
        render_greedy_policy(env, model, DISCOUNT, None, 200)
        





    
            

    
