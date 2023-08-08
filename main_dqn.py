import numpy as np
import time
from dpendulum import DPendulum
from dqn import dqn,get_critic
import matplotlib.pyplot as plt 
import tensorflow as tf


def compute_V_pi_from_Q(env, model, plot_discretization=30):
    ''' Compute Value table and greedy policy pi from Q table. '''
    
    vMax = env.vMax
    dq   = 2*np.pi/plot_discretization
    dv   = 2*vMax/plot_discretization
    nx = env.state_size()
    
    V  = np.zeros((plot_discretization+1, plot_discretization+1))
    pi = np.zeros((plot_discretization+1, plot_discretization+1))
    x  = np.zeros((nx, plot_discretization+1))

    x[0,:] = np.arange(-np.pi, np.pi + dq, dq)
    x[1,:] = np.arange(-vMax, vMax + dv, dv)
                       
    for q in range(plot_discretization+1):
        for v in range(plot_discretization+1):
            xu = np.reshape([x[0,q] * np.ones(model.nu), x[1,v] * np.ones(model.nu), np.arange(model.nu)], (nx + 1, model.nu))
            #xu = np.append([x[0,q]*np.ones(env.nu)], [x[1,v]*np.ones(env.nu), np.arange(env.nu)], axis=0)
            V[q,v] = np.min(model(xu.T))
            pi[q,v] = env.d2cu(np.argmin(model(xu.T)))
            
    return x, V, pi


def render_greedy_policy(env, model, gamma, x0=None, maxIter=20):
    x0 = env.reset(x0)
    x = x0
    costToGo = 0.0
    gamma_to_the_i = 1
    time.sleep(1)

    # storing the history over time of x, u and the cost in 3 lists
    hist_x = np.zeros([maxIter, env.state_size()]) # modified from ale work
    hist_u = []
    hist_cst = []

    for i in range(maxIter):
        # selecting policy using ϵ-greedy strategy (ϵ-greedy policy)       
        if(np.random.uniform(0, 1) < exploration_prob):
            u = np.random.randint(0, env.nu) # with probability exploration_prob take a random control input
        
        if(env.nbJoint == 2): x, cost = env.step([u,env.c2du(0.0)])
        else:                 x, cost = env.step([u])
        
        '''
        x = np.reshape(x,(env.state_size(),1))
        xu = np.append(x*np.ones(env.nu), [np.arange(env.nu)], axis=0)
        u = np.argmax(model(xu.T)) # Greedy action
        '''

        costToGo += gamma_to_the_i * cost
        gamma_to_the_i *= gamma

        hist_x[i,:]  = np.concatenate(np.array([x]).T)
        hist_u.append(env.d2cu(u))
        hist_cst.append(cost)

        env.render()

    print("Real cost to go of state,", x0, ":", costToGo)   

    return hist_x, hist_u, hist_cst


if __name__ == '__main__':

    
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES               = 20        # Number of training episodes
    NPRINT                  = 1         # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 100       # Max episode length
    # LEARNING_RATE           = 0.8       # alpha coefficient of Q learning algorithm
    DISCOUNT                = 0.99      # Discount factor 
    PLOT                    = False     # Plot stuff if True
    BATCH_SIZE              = 32        # size of the batch for replay buffer
    NX                      = 2         # number of states
    NU                      = 1         # number of control inputs
    REPLAY_STEP             = 4
    NETWORK_UPDATE_STEP     = 100       # c_step of repo
    QVALUE_LEARNING_RATE    = 1e-3
    exploration_prob                = 1     # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.05  # exploration decay for exponential decreasing
    min_exploration_prob            = 0.001 # minimum of exploration probability
    
    ### --- Pendulum Environment
    nbJoint = 1
    nx = 2
    nu = 15 #number of discretization steps for the torque u
    
    # ----- FLAG to TRAIN/LOAD
    FLAG                         = True # False = Load Model
    env = DPendulum(nbJoint, nu)

    # Creation of the Deep Q-Network models
    model = get_critic(NX,NU)
    model_target = get_critic(NX,NU)
    model_target.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

    print("\n\n\n###############################################")
    print("DEEP Q LEARNING")
    print("###############################################\n\n")


    if FLAG == True:

        h_ctg = dqn(replay_buffer, model, env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH, MIN_BUFFER, C_STEP, EXPLORATION_PROB, EXPLORATION_DECREASING_DECAY, MIN_EXPLORATION_PROB, plot_traj, PLOT, NPRINT)
        plt.show()

        # save model and weights
        print("\nTraining finished")
        print("\nSave NN weights to file (in HDF5)")
        if (nbJoint == 1):
            model.Q.save('saved_model/my_model')
            model.Q.save_weights('saved_model/weight1.h5')
        else:    
            model.Q.save('saved_model/my_model2')
            model.Q.save_weights('saved_model/weight2.h5')

        #plot cost
        plt.figure()
        plt.plot( np.cumsum(h_ctg)/range(1,NEPISODES+1) )
        plt.title ("Average cost-to-go")

    if FLAG == False: #load model
        if (nbJoint == 1):
            model.Q = tf.keras.models.load_model('saved_model/my_model')
        else:
            model.Q = tf.keras.models.load_model('saved_model/my_model2')
        assert(model.Q)
        
        
    '''    
    X_sim, U_sim, Cost_sim = render_greedy_policy(env, agent, EXPLORATION_PROB, None, MAX_EPISODE_LENGTH)

    if PLOT_TRAJ:
        time_vec = np.linspace(0.0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
        plot_traj(time_vec, X_sim, U_sim, Cost_sim, env)

    plt.figure.max_open_warning = 50
    plt.show()
    '''    
          
          
    start = time.time()
    h_ctg = dqn(env,DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH, exploration_prob, REPLAY_STEP, model, model_target, BATCH_SIZE, \
                min_exploration_prob, exploration_decreasing_decay, NETWORK_UPDATE_STEP, NPRINT,optimizer)    #parametri della funzione da inserire
    end = time.time()
    Time = round((end-start)/60,3)
    print("Training time:", Time)

    if(nbJoint == 1):
        x, V, pi = compute_V_pi_from_Q(env, model, 20)
        env.plot_V_table(V, x[0], x[1])
        env.plot_policy(pi, x[0], x[1])
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V))

    plt.figure()
    plt.plot(np.cumsum(h_ctg)/range(1,NEPISODES+1) )
    plt.show()

    hist_x, hist_u, hist_cst = render_greedy_policy(env, model, DISCOUNT, maxIter=MAX_EPISODE_LENGTH)

    time = np.linspace(0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
    figure,axes = plt.subplots(4,1)

    ax = axes[0]
    ax.plot(time, hist_cst)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cost")
    ax.set_title('Cost history')

    ax = axes[1]
    ax.plot(time, hist_u)
    ax.set_xlabel("Time")
    ax.set_ylabel("Torque")
    ax.set_title('Torque history') 

    ax = axes[2]
    ax.plot(time, hist_x[:,0])
    ax.set_xlabel("Time")
    ax.set_ylabel("Joint position")
    ax.set_title('Joint position history') 

    ax = axes[3]
    ax.plot(time, hist_x[:,1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Joint angular velocity")
    ax.set_title('Angular velocity history') 

    figure.suptitle("Episode data")
    plt.show()


    #for i in range(20):
    #    render_greedy_policy(env,model,DISCOUNT)





    
            

    
