from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
    

class DPendulum:
    ''' Discrete Pendulum environment (continuous state pendulum environment with discrete control input).
        Torque is discretized with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, nbJoint = 1, nu=11, vMax=5, uMax=5, dt=0.1, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(nbJoint, vMax, uMax, noise_stddev)
        self.pendulum.DT  = dt      # discrete-time step
        self.pendulum.NDT = ndt     # integration method step
        self.vMax = vMax            # Max velocity (v in [-vmax,vmax])
        self.nu = nu                # Number (must be odd) of discretization steps for joint torque 
        self.uMax = uMax            # Max torque (u in [-umax,umax])
        self.dt = dt                # time step
        self.DU = 2*uMax/(nu-1)     # discretization resolution for joint torque
        self.nbJoint = nbJoint      # number of robot joints
    
    def state_size(self):
        return self.pendulum.nx
    def control_size(self):
        return self.pendulum.nu

    # Continuous to discrete    
    def c2du(self, u): #joint torques saturation
        u = np.clip(u,-self.uMax,self.uMax)
        return int(np.floor((u+self.uMax)/self.DU))
    
    # Discrete to continuous
    def d2cu(self, iu): #joint torques saturation
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def reset(self,x=None): # continuos time reset
        self.x = self.pendulum.reset(x)
        return self.x

    def step(self,iu): # simulates the system forward of 1 time step and computes the cost
        self.x, cost = self.dynamics(self.x,iu)
        return self.x, cost

    def render(self): 
        self.pendulum.render()
        time.sleep(self.pendulum.DT)

    def dynamics(self, x, iu): 
        u   = self.d2cu(iu)
        self.x, cost = self.pendulum.step(u)
        return self.x, cost
    
    def plot_V_table(self, V, q, dq,model_name): 
        ''' Plot the given Value table V '''
        plt.figure() 
        plt.pcolormesh(q, dq, V, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title("V table")
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.savefig(model_name+"/Vtable.png")   
        plt.show()
        
    def plot_policy(self, policy, q, dq,model_name): 
        ''' Plot the given policy table pi '''
        plt.figure()
        plt.pcolormesh(q, dq, policy, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title("Policy table")
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.savefig(model_name+"/Policy.png")    
        plt.show() 