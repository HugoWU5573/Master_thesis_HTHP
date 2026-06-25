import CoolProp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class Valve:
    """
    Operationnal model of a valve
    """
    def __init__(self, type) :

        if type == 'ETS6-25' :
            self.coeffs = np.array([-3.71259328e-10, 6.85466880e-08, -3.66184812e-07])
        elif type == 'ETS6-32' :
            self.coeffs = np.array([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
        else : 
            raise ValueError("Unsupported valve type. Please choose 'ETS6-25' or 'ETS6-32'.")
    
    def Solve(self, state_in, mdot, z) :
        """
        Resolve the outlet pressure and enthalpy of the valve given the inlet state, mass flow rate, and valve opening position.

        Parameters:
            - state_in: The inlet state of the fluid (object with properties p, T, h, Q, and heos).
            - mdot: The mass flow rate through the valve.
            - z: The valve opening position (0 to 100).
        
        Returns:
            - p_out: The outlet pressure after the valve.
            - h_out: The outlet enthalpy after the valve (same as inlet enthalpy).
        """
        heos = state_in.heos
        if state_in.Q > 0 and state_in.Q < 1:
            heos.update(CoolProp.QT_INPUTS, state_in.Q, state_in.T)
            rho_in = heos.rhomass()
        else:
            heos.update(CoolProp.PT_INPUTS, state_in.p, state_in.T)
            rho_in = heos.rhomass()
        
        def iteration_function(sqrt_dp):
            mass_flow_rate = np.polyval(self.coeffs, z) * np.sqrt(2 * rho_in) * sqrt_dp
            return mass_flow_rate - mdot

        sqrt_dp = fsolve(iteration_function, state_in.p)[0]
        if sqrt_dp < 0:
            raise ValueError("Calculated sqrt_dp is negative, check the input parameters and coefficients.")
        p_out = state_in.p - sqrt_dp**2
        if p_out < 0:
            raise ValueError("Calculated outlet pressure is negative, check the input parameters and coefficients.")
        
        #print(p_out, state_in.h)
        return p_out, state_in.h
    
class Valve_other :
    """
    Operationnal model of a valve but with a switch bewteen inputs and outputs.
    
    """
    def __init__(self, type) :
        if type == 'ETS6-25' :
            self.coeffs = np.array([-3.71259328e-10, 6.85466880e-08, -3.66184812e-07])
        elif type == 'ETS6-32' :
            self.coeffs = np.array([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
        else : 
            raise ValueError("Unsupported valve type. Please choose 'ETS6-25' or 'ETS6-32'.")

    def Solve(self, state_in, p_out, mdot) :
        """
        Resolve the outlet pressure and enthalpy of the valve given the inlet state, desired outlet pressure, and mass flow rate.

        Parameters:
            - state_in: The inlet state of the fluid (object with properties p, T, h, Q, and heos).
            - p_out: The desired outlet pressure after the valve.
            - mdot: The mass flow rate through the valve.
        
        Returns:
            - h_out: The outlet enthalpy after the valve (same as inlet enthalpy).
            - z: The outlet pressure after the valve.
            
        """
        heos = state_in.heos
        heos.update(CoolProp.HmassP_INPUTS, state_in.h, state_in.p)
        
        def iteration_function(z):
            mass_flow_rate = np.polyval(self.coeffs, z) * np.sqrt(2 * heos.rhomass()) * np.sqrt(state_in.p - p_out)
            return mass_flow_rate - mdot
        
        z = fsolve(iteration_function, 50)[0]
        return state_in.h, z
    
    def get_points_between(self, state_in, state_out, n_points=100):
        heos = state_in.heos
        p_min = state_in.p
        p_max = state_out.p
        p = np.linspace(p_min, p_max, n_points)
        T = np.zeros(n_points)
        s = np.zeros(n_points)
        h = np.zeros(n_points)
        for i, p_val in enumerate(p):
            try : 
                heos.update(CoolProp.HmassP_INPUTS, state_in.h, p_val)
                T[i] = heos.T()
                s[i] = heos.smass()
                h[i] = state_in.h
                
            except ValueError as e:
                T[i] = float('nan')
                s[i] = float('nan')
                h[i] = float('nan')
        return T, s, p, h
    
    def energy_analysis(self, P_el, state_in, state_out, mdot_wf) : 
        

        return {}
    
    def exergy_analysis(self, T0, P0, P_el, state_in, state_out, mdot_wf) :


        return {}

