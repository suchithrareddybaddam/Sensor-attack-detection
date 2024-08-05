from numpy.random import randn
import math

class RadarSim(object):
    """ Simulates the radar signal returns from an object flying 
    at a constant altityude and velocity in 1D. 
    """
    
    def __init__(self, dt, pos, vel, alt):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt
        
    def get_range(self):
        """ Returns slant range to the object. Call once for each
        new measurement at dt time from last call.
        """
        a = 0.02540858775238177
        b = 0.38865052218213947
        # add some process noise to the system
        self.vel = self.vel  + .1*a
        self.alt = self.alt + .1*b
        self.pos = self.pos + self.vel*self.dt
        # add measurement noise
        err = self.pos * 0.05*-0.13397292877918187
        slant_dist = math.sqrt(self.pos**2 + self.alt**2)
        
        return slant_dist + err
