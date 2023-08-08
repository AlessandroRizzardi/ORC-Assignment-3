import numpy as np
import time
from dpendulum import DPendulum
from dqn import dqn,get_critic
import matplotlib.pyplot as plt 
import tensorflow as tf


def compute_V_pi_from_Q(env,model,plot_discretization):

    dq = 2*np.pi/plot_discretization
    dv = 2*env.vMax/plot_discretization

    V = np.zeros((plot_discretization, plot_discretization))
    pi = np.zeros((plot_discretization, plot_discretization))

    x = np.zeros((env.state_size(), plot_discretization))
    x[0,:] = np.arange(-np.pi, np.pi, dq)
    x[1,:] = np.arange(-env.vMax, env.vMax, dv)
                       
    for q in range(plot_discretization):
        for v in range(plot_discretization):

            xu = np.append([x[0,q]*np.ones(env.nu)], [x[1,v]*np.ones(env.nu), np.arange(env.nu)], axis=0)

            V[q,v] = np.max(model(xu.T))
            pi[q,v] = env.d2cu(np.argmax(model(xu.T)))
    
    return x,V,pi


def render_greedy_policy(env,model, gamma, x0=None, maxiter=20):
    x0 = env.reset(x0)
    x = x0
    costToGo = 0.0
    gamma_i = 1
    env.render()
    time.sleep(1)

    history_x = []
    history_u = []
    history_cost = []

    for i in range(maxiter):
        x = np.reshape(x,(env.state_size(),1))
        xu = np.append(x*np.ones(env.nu), [np.arange(env.nu)], axis=0)
        u = np.argmax(model(xu.T)) # Greedy action
        

        x,cost = env.step([u])
        costToGo += gamma_i*cost
        gamma_i *= gamma


        history_x.append(x)
        history_u.append(env.d2cu(u))
        history_cost.append(cost)

        env.render()

    print("Real cost to go of state,", x0, ":", costToGo)   

    history_x = np.reshape(history_x,(maxiter,env.state_size()))

    return history_x, history_u, history_cost


if __name__ == '__main__':

    
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES               = 20        # Number of training episodes
    NPRINT                  = 1           # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 100          # Max episode length
    LEARNING_RATE           = 0.8           # alpha coefficient of Q learning algorithm
    DISCOUNT                = 0.99           # Discount factor 
    PLOT                    = False         # Plot stuff if True
    BATCH_SIZE              = 32            # size of the butch for replay buffer 
    NX                      = 2            
    NU                      = 1
    REPLAY_STEP             = 4
    NETWORK_UPDATE_STEP     = 100
    QVALUE_LEARNING_RATE    = 1e-3
    exploration_prob                = 1     # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.05 # exploration decay for exponential decreasing
    min_exploration_prob            = 0.001 # minimum of exploration proba
    
    ### --- Environment
    nu = 20 #number of discretization steps for the torque u
    env = DPendulum(1,nu)

    # Creation of the Deep Q-Network models
    model = get_critic(NX,NU)
    model_target = get_critic(NX,NU)
    model_target.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

    print("\n\n\n###############################################")
    print("DEEP Q LEARNING")
    print("###############################################\n\n")

    start = time.time()
    h_ctg = dqn(env,DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH, exploration_prob, REPLAY_STEP, model, model_target, BATCH_SIZE, \
                min_exploration_prob, exploration_decreasing_decay, NETWORK_UPDATE_STEP, NPRINT,optimizer)    #parametri della funzione da inserire
    end = time.time()
    Time = round((end-start)/60,3)
    print("Training time:",Time)
    
    #print("QUA 1")
    x,V,pi = compute_V_pi_from_Q(env, model, 20)

    env.plot_V_table(V,x[0],x[1])
    env.plot_policy(pi,x[0],x[1])
    print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V))

    plt.figure()
    plt.plot(np.cumsum(h_ctg)/range(1,NEPISODES) )
    plt.show()

    history_x, history_u, history_cost = render_greedy_policy(env,model,DISCOUNT,maxiter=MAX_EPISODE_LENGTH)

    time = np.linspace(0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
    figure,axes = plt.subplots(4,1)

    ax = axes[0]
    ax.plot(time,history_cost)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cost")
    ax.set_title('Cost history')

    ax = axes[1]
    ax.plot(time,history_u)
    ax.set_xlabel("Time")
    ax.set_ylabel("Torque")
    ax.set_title('Torque history') 

    ax = axes[2]
    ax.plot(time,history_x[:,0])
    ax.set_xlabel("Time")
    ax.set_ylabel("Joint position")
    ax.set_title('Joint position history') 

    ax = axes[3]
    ax.plot(time,history_x[:,1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Joint angular velocity")
    ax.set_title('Angular velocity history') 

    figure.suptitle("Episode data")
    plt.show()


    #for i in range(20):
    #    render_greedy_policy(env,model,DISCOUNT)





    
            

    
