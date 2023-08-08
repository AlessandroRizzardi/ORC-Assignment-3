from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
    

class DPendulum:
    ''' Discrete Pendulum environment. Joint angle, velocity and torque are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, nbJoint = 1, nu=11, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(nbJoint,noise_stddev)
        self.pendulum.DT  = dt      # discrete-time step
        self.pendulum.NDT = ndt     # integration method step
        self.vMax = vMax    # Max velocity (v in [-vmax,vmax])
        self.nu = nu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.dt = dt        # time step
        self.DU = 2*uMax/nu # discretization resolution for joint torque
        self.nbJoint = nbJoint

    
    def state_size(self):
        return self.pendulum.nx

    # Continuous to discrete    
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))
    

    # Discrete to continuous
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def reset(self,x=None):
        self.x = self.pendulum.reset(x)
        return self.x

    def step(self,iu):
        self.x, cost = self.dynamics(self.x,iu)
        return self.x, cost
 
    def render(self):
        self.pendulum.render()
        time.sleep(self.pendulum.DT)

    def dynamics(self,x,iu):
        u   = self.d2cu(iu)
        #self.xc,cost= self.pendulum.dynamics(x,u)
        self.xc,cost= self.pendulum.step(u)
        return self.xc,cost
    
    def plot_V_table(self, V, q , dq):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt

        plt.pcolormesh(q, dq, V, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('V table')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_policy(self, pi, q, dq):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt
  
        plt.pcolormesh(q, dq, pi, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title('Policy')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_Q_table(self, Q):
        ''' Plot the given Q table '''
        import matplotlib.pyplot as plt
        X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
        plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('Q table')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.show()
        