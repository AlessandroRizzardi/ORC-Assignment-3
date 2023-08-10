import numpy as np
import time
from dpendulum import DPendulum
from dqn import dqn
import matplotlib.pyplot as plt 
import tensorflow as tf
from auxiliary_func import *



def trajectories(time_vec, X_sim, U_sim, Cost_sim, env):
    figure()
    plt.plot(time_vec, U_sim[:], "b")
    if env.uMax:
        plt.plot(time_vec, env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
        plt.plot(time_vec, -env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[Nm]')
    plt.title ("Torque input")

    plt.figure()
    plt.plot(time_vec, Cost_sim[:], "b")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Cost')
    plt.title ("Cost")

    plt.figure()
    plt.plot(time_vec, X_sim[:,0],'b')
    if env.njoint == 2:
        plt.plot(time_vec, X_sim[:,1],'r')
        plt.legend(["1st joint position","2nd joint position"],loc='upper right')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad]')
    plt.title ("Joint position")
    
    plt.figure()
    if env.njoint == 1:
        plt.plot(time_vec, X_sim[:,1],'b')
    else:
        plt.plot(time_vec, X_sim[:,2],'b')
        plt.plot(time_vec, X_sim[:,3],'r')
        plt.legend(["1st joint velocity","2nd joint velocity"],loc='upper right')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad/s]')
    plt.title ("Joint velocity")


if __name__ == '__main__':

    
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES               = 50                # Number of training episodes
    NPRINT                  = 1                    # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 100                  # Max episode length
    DISCOUNT                = 0.99                 # Discount factor 
    PLOT                    = False                # Plot stuff if True
    BATCH_SIZE              = 32                   # size of the batch for replay buffer
    MIN_BUFFER              = 100                  # lower bound as start for sampling from buffer
    
    NX                      = 2         # number of states
    NU                      = 1         # number of control inputs
    
    # REPLAY_STEP           = 4       # TO KEEP ???
    NETWORK_UPDATE_STEP     = 100      # how many steps taken for updating w
    QVALUE_LEARNING_RATE    = 1e-3      # alpha coefficient of Q learning algorithm
    exploration_prob                = 1     # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.05  # exploration decay for exponential decreasing
    min_exploration_prob            = 0.001 # minimum of exploration probability
    
    ### --- Pendulum Environment
    nbJoint = 1                # joints number
    nx = 41                     # number of discretization steops for the state
    nu = 21                    # number of discretization steps for the torque u
    
    # ----- FLAG to TRAIN/LOAD
    FLAG                         = True # False = Load Model
    env = DPendulum(nbJoint, nu)

    # Creation of the Deep Q-Network models (create critic and target NNs)
    model = get_critic(NX,NU)                                         # Q network
    target_model = get_critic(NX,NU)                                  # Target network
    target_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE) # optimizer specifying the learning rates
    #model.summary()
    
    if(FLAG == True):
        print("\n\n\n###############################################")
        print("*** DEEP Q LEARNING ***")
        print("###############################################\n\n")

        start = time.time()  
        h_ctg = dqn(env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH,\
                    exploration_prob, model, target_model, MIN_BUFFER,\
                    BATCH_SIZE, optimizer,NETWORK_UPDATE_STEP, min_exploration_prob ,\
                    exploration_decreasing_decay , PLOT, NPRINT )
        end = time.time()
        Time = round((end-start)/60,3)
        print("Training time:",Time)
        #plt.show()
        
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
        plt.plot( np.cumsum(h_ctg)/range(1, NEPISODES+1) )
        plt.title ("Average cost-to-go")

    if(FLAG == False): #load model
        if (nbJoint == 1):
            model = tf.keras.models.load_model('saved_model/Model1')
        else:
            model = tf.keras.models.load_model('saved_model/Model2')
        assert(model)
          
    if(nbJoint == 1):
        x, V, pi = compute_V_pi_from_Q(env, model, 20)
        env.plot_V_table(V, x[0], x[1])
        env.plot_policy(pi, x[0], x[1])
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V))

    hist_x, hist_u, hist_cost = render_greedy_policy(env, model, target_model, DISCOUNT, None, MAX_EPISODE_LENGTH)
    
            
    '''
    *** METTI A POSTO! ***    
    if PLOT_TRAJ:
        time_vec = np.linspace(0.0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
        trajectories(time_vec, X_sim, U_sim, Cost_sim, env)

    plt.figure.max_open_warning = 50
    plt.show()
    '''    
        
    # time = np.linspace(0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)

    figure,axes = plt.subplots(4,1)

    ax = axes[0]
    ax.plot(time, hist_cost)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cost")
    ax.set_title('Cost history')

    ax = axes[1]
    ax.plot(time, hist_u)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [N*m]")
    ax.set_title('Torque history') 

    ax = axes[2]
    ax.plot(time, hist_x[:,0])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint position [rad]")
    ax.set_title('Joint position history') 

    ax = axes[3]
    ax.plot(time, hist_x[:,1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint angular velocity [rad/s]")
    ax.set_title('Angular velocity history') 

    figure.suptitle("Episode data")
    plt.show()


    #for i in range(20):
    #    render_greedy_policy(env,model,DISCOUNT)





    
            

    
