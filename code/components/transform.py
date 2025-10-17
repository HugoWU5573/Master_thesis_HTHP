from components.state import State
import numpy as np
import CoolProp

class Transform:
    def __init__(self, type,label_in, label_out, component) :
        self.type = type
        self.label_in = label_in
        self.label_out = label_out
        self.component = component

    def get_points_between(self, state_in, state_out, n_points=100):
        """
        Generate intermediate points between two thermodynamic states for plotting purposes.
        
        Parameters:
        state_in (State): The initial thermodynamic state.
        state_out (State): The final thermodynamic state.
        n_points (int): Number of intermediate points to generate.
        
        Returns:
        list: A list of State objects representing the intermediate points.
        """
        if self.type == 'None' : 
            raise ValueError("Transform type must be defined to generate intermediate points.")
        
        T = np.zeros(n_points)
        s = np.zeros(n_points)

        heos = state_in.heos  # Assuming both states use the same fluid
        
        if self.type == 'adex' : 
            p_max = state_in.p
            p_min = state_out.p
            p = np.linspace(p_max, p_min, n_points)
            h = state_in.h

            for i, p_val in enumerate(p):
                heos.update(CoolProp.HmassP_INPUTS, h, p_val)
                T[i] = heos.T()
                s[i] = heos.smass()

        elif self.type == 'hex' : 
            p = state_in.p
            s_max = state_in.s
            s_min = state_out.s
            s = np.linspace(s_max, s_min, n_points)
            for i, s_val in enumerate(s):
                heos.update(CoolProp.PSmass_INPUTS, p, s_val)
                T[i] = heos.T()
        
        elif self.type == 'comp' :
            T, s = self.component.get_points_between(state_in, state_out, n_points)
        return T, s