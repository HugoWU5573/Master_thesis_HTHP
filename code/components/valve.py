import CoolProp
import numpy as np
import matplotlib.pyplot as plt

class Valve:
    def __init__(self, a_0, a_1) :
        self.a_0 = a_0
        self.a_1 = a_1
    
    def mass_flow_rate(self, state_in, p_out, z) :
        heos = state_in.heos
        if state_in.Q > 0 and state_in.Q < 1:
            heos.update(CoolProp.QT_INPUTS, state_in.Q, state_in.T)
            rho_in = heos.rhomass()
        else:
            heos.update(CoolProp.PT_INPUTS, state_in.p, state_in.T)
            rho_in = heos.rhomass()
            
        return (self.a_0 + self.a_1 * z) * np.sqrt(2 * rho_in * (state_in.p - p_out))