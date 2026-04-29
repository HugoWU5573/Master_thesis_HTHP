from components.state import State
from components.HEX import HEX_Operational
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
        p = np.zeros(n_points)
        h = np.zeros(n_points)

        heos = state_in.heos  # Assuming both states use the same fluid
        
        if self.type == 'adex' : 
            p_max = state_in.p
            p_min = state_out.p
            p = np.linspace(p_max, p_min, n_points)
            p_crit = heos.p_critical()
            h = state_in.h * np.ones(n_points)

            for i, p_val in enumerate(p):
                if np.isclose(p_val/1e5, p_crit/1e5, atol=1e-1):
                    # Skip the critical pressure to avoid problem with heos.update
                    T[i] = T[i-1]  # Assign previous temperature to avoid discontinuity
                    s[i] = s[i-1]
                else :
                    heos.update(CoolProp.HmassP_INPUTS, h[0], p_val)
                    T[i] = heos.T()
                    s[i] = heos.smass()

        elif self.type == 'hex' :
            if isinstance(self.component, HEX_Operational) :
                T_in, s_in, p_in, h_in = self.component.get_points_between()
                if len(T_in) <= n_points:
                    T[:len(T_in)] = T_in
                    T = T[:len(T_in)]
                    s[:len(s_in)] = s_in
                    s = s[:len(s_in)]
                    p[:len(p_in)] = p_in
                    p = p[:len(p_in)]
                    h[:len(h_in)] = h_in
                    h = h[:len(h_in)]
                else :
                    raise ValueError("Number of points from the component is greater than n_points. Consider increasing n_points or reducing the number of points returned by the component.")
                
            else :
                p = state_in.p * np.ones(n_points)
                s_max = state_in.s
                s_min = state_out.s
                s = np.linspace(s_max, s_min, n_points)
                for i, s_val in enumerate(s):
                    heos.update(CoolProp.PSmass_INPUTS, p[i], s_val)
                    T[i] = heos.T()
                    h[i] = heos.hmass()    
        
        elif self.type == 'comp' :
            T, s, p, h = self.component.get_points_between(state_in, state_out, n_points)
            for i in range(n_points):
                if np.isnan(T[i]) or np.isnan(s[i]):
                    T[i] = T[i-1]
                    s[i] = s[i-1]

        elif self.type == 'isobaric_mixing' :
            p = state_in.p * np.ones(n_points)
            s_max = state_in.s
            s_min = state_out.s
            s = np.linspace(s_max, s_min, n_points)
            for i, s_val in enumerate(s):
                heos.update(CoolProp.PSmass_INPUTS, p[i], s_val)
                T[i] = heos.T()
                h[i] = heos.hmass()

        return T, s, p, h

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
        
        if self.type == 'isobaric_mixing' :
            mdot_wf = args['mdot_wf']
            exergy_in = mdot_wf[0] * state_in[0].exergy(T0, P0) + mdot_wf[1] * state_in[1].exergy(T0, P0)
            exergy_out = abs(mdot_wf[0] + mdot_wf[1]) * state_out.exergy(T0, P0)
            exergy_losses = exergy_in - exergy_out
            dict_exergy = {'P_{irr}' : exergy_losses}
            return dict_exergy
