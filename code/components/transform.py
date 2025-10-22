from components.state import State
import numpy as np
import CoolProp

class Transform:
    def __init__(self, type,label_in, label_out, component, label_in_secondary=None, label_out_secondary=None) :
        self.type = type
        self.label_in = label_in
        self.label_out = label_out
        self.component = component
        self.label_in_secondary = label_in_secondary
        self.label_out_secondary = label_out_secondary

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
            p_crit = heos.p_critical()
            h = state_in.h

            for i, p_val in enumerate(p):
                if np.isclose(p_val/1e5, p_crit/1e5, atol=1e-1):
                    # Skip the critical pressure to avoid problem with heos.update
                    T[i] = T[i-1]  # Assign previous temperature to avoid discontinuity
                    s[i] = s[i-1]
                else :
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

        elif self.type == 'isobaric_mixing' :
            p = state_in.p
            s_max = state_in.s
            s_min = state_out.s
            s = np.linspace(s_max, s_min, n_points)
            for i, s_val in enumerate(s):
                heos.update(CoolProp.PSmass_INPUTS, p, s_val)
                T[i] = heos.T()

        return T, s
    
    def energy_analysis(self, state_in, state_out, args) : 
        """
        Placeholder for energy analysis of the transform.
        """
        if self.type == 'None' :
            raise ValueError("Transform type must be defined to perform energy analysis.")
        
        if self.type == 'adex' :
            return None 
        
        elif self.type == 'comp' :
            P_el = args['P_el']
            mdot_wf = args['mdot_wf']
            return self.component.energy_analysis(P_el, state_in, state_out, mdot_wf)
        
        elif self.type == 'hex' :
            mdot_wf = args['mdot_wf']
            mdot_secondary = args['mdot_secondary']
            state_in_secondary = args['state_in_secondary']
            state_out_secondary = args['state_out_secondary']
            return self.component.energy_analysis(state_in, state_out, mdot_wf, mdot_secondary, state_in_secondary, state_out_secondary)
    
    def exergy_analysis(self, T0, P0, state_in, state_out, args) : 
        """
        Placeholder for exergy analysis of the transform.
        """
        if self.type == 'None' :
            raise ValueError("Transform type must be defined to perform exergy analysis.")
        
        if self.type == 'adex' :
            mdot_wf = args['mdot_wf']
            exergy_losses = (state_in.exergy(T0, P0) - state_out.exergy(T0, P0)) * mdot_wf
            dict_exergy = {'P_{irr}' : exergy_losses}
            return dict_exergy

        elif self.type == 'comp' :
            P_el = args['P_el']
            mdot_wf = args['mdot_wf']
            return self.component.exergy_analysis(T0, P0, P_el, state_in, state_out, mdot_wf)
        
        elif self.type == 'hex' :
            mdot_wf = args['mdot_wf']
            mdot_secondary = args['mdot_secondary']
            state_in_secondary = args['state_in_secondary']
            state_out_secondary = args['state_out_secondary']
            return self.component.exergy_analysis(T0, P0, state_in, state_out, mdot_wf, mdot_secondary, state_in_secondary, state_out_secondary)
