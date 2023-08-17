import numpy as np
import time
from network_utils import *
import matplotlib.pyplot as plt

def compute_V_pi_from_Q(env, model, plot_discretization=30):
    ''' Compute Value table and greedy policy pi from Q table. '''
    
    vMax = env.vMax
    dq   = 2*np.pi/plot_discretization
    dv   = 2*vMax/plot_discretization
    
    V  = np.zeros((plot_discretization+1, plot_discretization+1))
    pi = np.zeros((plot_discretization+1, plot_discretization+1))
    x  = np.zeros((env.state_size(), plot_discretization+1)) # np.empty

    x[0,:] = np.arange(-np.pi, np.pi + dq, dq)
    x[1,:] = np.arange(-vMax, vMax + dv, dv)
                       
    for q in range(plot_discretization+1):
        for v in range(plot_discretization+1):

            xu      = np.reshape([x[0,q] * np.ones(env.nu), x[1,v] * np.ones(env.nu), np.arange(env.nu)], (env.state_size() + 1, env.nu))
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
    hist_x    = np.zeros([maxIter, env.state_size()])
    hist_u    = []
    hist_cost = []

    for i in range(maxIter):
        '''# selecting policy using ϵ-greedy strategy (ϵ-greedy policy)' or random innput'''     
        u = action_selection(0, env, x, model)
        
        x, cost = dyn_forNbigger_thanOne(env, u)
        
        costToGo += gamma_to_the_i * cost
        gamma_to_the_i *= gamma

        env.render() 

        hist_x[i,:]  = np.concatenate(np.array([x]).T)
        hist_u.append(env.d2cu(u))
        hist_cost.append(cost)

    print("Real cost-to-go of state x0", x0, "=", costToGo)   
    return hist_x, hist_u, hist_cost

def trajectories(time, hist_x, hist_u, hist_cost, env,model_name):
    
    plt.figure()
    plt.plot(time, hist_cost, "-", linewidth = 2)
    plt.xlabel("Time [s]")
    plt.ylabel("Cost")
    plt.title("Cost")

    plt.figure()
    plt.plot(time, hist_u, "m")
    plt.plot(time, env.uMax * np.ones(len(time)), "b--", alpha=0.8, linewidth=2)
    plt.plot(time, -env.uMax * np.ones(len(time)), "b--", alpha=0.8, linewidth=2)
    plt.ylim([-env.uMax -1, env.uMax +1])
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title("Torque input")   
    plt.savefig(model_name+"/Torque.png")         # Model_#  Model_double_#   
        
    plt.figure()
    plt.plot(time, hist_x[:,0], "c")
    if(env.nbJoint == 2):
        plt.plot(time, hist_x[:,1], "k")
        plt.legend(["joint 1 position","joint 2 position"], loc = "best")
    plt.plot(time, 0 * np.ones(len(time)), "m--", alpha=0.8, linewidth=2)
    plt.ylim([-np.pi - 1,np.pi + 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("Joints positions")
    plt.savefig(model_name+"/Position.png")         # Model_#  Model_double_# 

    plt.figure()
    if(env.nbJoint==1):
        plt.plot(time, hist_x[:,1], "g")
    else:
        plt.plot(time, hist_x[:,2], "b")
        plt.plot(time, hist_x[:,3], "r")
        plt.legend(["joint 1 ang. velocity","joint 2 ang. velocity"], loc = "best")
    plt.plot(time, env.vMax * np.ones(len(time)), "b--", alpha=0.8, linewidth=2)
    plt.plot(time, -env.vMax * np.ones(len(time)), "b--", alpha=0.8, linewidth=2)
    plt.plot(time, 0 * np.ones(len(time)), "m--", alpha=0.8, linewidth=2)
    plt.ylim([-env.vMax-1, env.vMax+1])
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Joints velocities")
    plt.savefig(model_name+"/Velocity.png")         # Model_#  Model_double_# 

   


