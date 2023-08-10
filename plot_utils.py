import numpy as np
import time
from network_utils import action_selection,dyn_forNbigger_thanOne
import matplotlib.pyplot as plt

def compute_V_pi_from_Q(env, model, plot_discretization=30):
    ''' Compute Value table and greedy policy pi from Q table. '''
    
    vMax = env.vMax
    dq   = 2*np.pi/plot_discretization
    dv   = 2*vMax/plot_discretization
    nx   = env.state_size()
    
    V  = np.zeros((plot_discretization+1, plot_discretization+1))
    pi = np.zeros((plot_discretization+1, plot_discretization+1))
    x  = np.zeros((nx, plot_discretization+1))

    x[0,:] = np.arange(-np.pi, np.pi + dq, dq)
    x[1,:] = np.arange(-vMax, vMax + dv, dv)
                       
    for q in range(plot_discretization+1):
        for v in range(plot_discretization+1):

            xu      = np.reshape([x[0,q] * np.ones(env.nu), x[1,v] * np.ones(env.nu), np.arange(env.nu)], (nx + 1, env.nu))

            V[q,v]  = np.min(model(xu.T))
            pi[q,v] = env.d2cu(np.argmin(model(xu.T)))
            
    return x, V, pi

def render_greedy_policy(env, model, gamma, x0=None, maxIter=90):
    ''' ***** render_greedy_policy *****
    Simulates the system using the policy I just computed:
    render_greedy_policy initializes the pendulum with a random initial state and then
    simulates it with the policy we computed
    We roll-out from random state using greedy policy  '''
    x0 = env.reset(x0)
    x = x0
    costToGo = 0.0
    gamma_to_the_i = 1
    env.render()
    time.sleep(1) 

    # storing the histories over time of x, u and the cost in 3 lists
    hist_x   = np.zeros([maxIter, env.state_size()])
    hist_u   = []
    hist_cost = []

    for i in range(maxIter):
        '''# selecting policy using ϵ-greedy strategy (ϵ-greedy policy)' or random innput'''     
        u = action_selection(0, env, x, model)
        
        x, cost = dyn_forNbigger_thanOne(env, u)

        costToGo += gamma_to_the_i * cost
        gamma_to_the_i *= gamma

        hist_x[i,:]  = np.concatenate(x,axis=1).T
        hist_u.append(env.d2cu(u))
        hist_cost.append(cost)

        env.render() 

    print("Real cost-to-go of state x0,", x0, "=", costToGo)   

    return hist_x, hist_u, hist_cost

def trajectories(time_vec, hist_x, hist_u, hist_cost, env):
    figure, axes = plt.subplots(6,1)

    ax = axes[0]
    ax.plot(time_vec, hist_cost, "o--", linewidth = 2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cost")
    ax.set_title('Cost history')

    ax = axes[1]
    ax.plot(time_vec, hist_u, "c--", linewidth = 2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title('Torque input history') 
    
    conta = 0
    if(env.uMax):
        ax = axes[2]
        ax.plot(time_vec, env.uMax*np.ones(len(time)), "m..", alpha=0.8, linewidth=1.5)
        ax.plot(time_vec, -env.uMax*np.ones(len(time)), "m..", alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque [Nm]")
        ax.set_title('Torque input saturation limits') 
        conta += 1

    if(conta == 1): ax = axes[3]
    else: ax = axes[2]
    
    conta = 0
    ax.plot(time_vec, hist_x[:,0], "r--", linewidth = 2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position of joint #1 [rad]")
    ax.set_title('Joint #1 position history') 
    if(env.nbJoint == 2):
        ax = axes[4]
        ax.plot(time_vec, hist_x[:,1], "o--", linewidth = 2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position of joint #2 [rad]")
        ax.set_title('Joint #2 position history') 
        conta += 1
    
    if(conta == 1): ax = axes[5]
    else: ax = axes[4]
    
    conta = 0
    
    if(env.nbJoint == 1):
        ax.plot(time_vec, hist_x[:,1],'b--')
    else:
        ax.plot(time_vec, hist_x[:,2],'y--')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[rad/s]')
        ax.set_title("Joint #1 angular velocity")
        ax = axes[6]
        ax.plot(time_vec, hist_x[:,3],'r--')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[rad/s]')
        ax.set_title("Joint #2 angular velocity")
        
    figure.suptitle("Episode data")
    plt.show()


