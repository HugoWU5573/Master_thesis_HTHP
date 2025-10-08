from CoolProp.CoolProp import PropsSI
import numpy as np

class state:
    def __init__(self, T=None, p=None, Q=None, h=None, s=None, fluid = 'R290'):
        """
        Initialize the state of a fluid using any two independent properties.
        Parameters:
        T (float): Temperature in Kelvin.
        p (float): Pressure in Pascal.
        Q (float): Quality (0 to 1 for saturated mixtures).
        h (float): Enthalpy in J/kg.
        s (float): Entropy in J/kg-K.
        fluid (str): The working fluid (default is 'R290').
        
        Valid combinations of inputs under the saturation curve:
        - T and p 
        - T and s
        - p and h
        - p and s
        - h and s

        Valid combinations of inputs outside the saturation curve:
        - T and Q
        - p and Q
        - p and h
        - p and s
        - T and s
        - h and s

        Raises:
        ValueError: If the provided combination of properties is invalid or insufficient.

        """
        self.e = None
        if Q is not None : 
            if Q == -1 : 
                raise ValueError("Point not under the saturation curve")
            self.Q = Q
            if T is not None :
                self.T = T
                self.p = PropsSI('P', 'T', T, 'Q', Q, fluid)
                self.h = PropsSI('H', 'T', T, 'Q', Q, fluid)
                self.s = PropsSI('S', 'T', T, 'Q', Q, fluid)

            elif p is not None :
                self.p = p
                self.T = PropsSI('T', 'P', p, 'Q', Q, fluid)
                self.h = PropsSI('H', 'P', p, 'Q', Q, fluid)
                self.s = PropsSI('S', 'P', p, 'Q', Q, fluid)

            elif h is not None :
                raise ValueError("Cannot define state with h and Q only")
            
            elif s is not None :
                raise ValueError("Cannot define state with s and Q only")
        
        else : 
            if T is not None and p is not None :
                self.T = T
                self.p = p
                self.h = PropsSI('H', 'T', T, 'P', p, fluid)
                self.s = PropsSI('S', 'T', T, 'P', p, fluid)
                self.Q = PropsSI('Q', 'T', T, 'P', p, fluid)

            elif T is not None and h is not None :
                raise ValueError("Cannot define state with T and h only")

            elif T is not None and s is not None :
                self.T = T
                self.s = s
                self.p = PropsSI('P', 'T', T, 'S', s, fluid)
                self.h = PropsSI('H', 'T', T, 'S', s, fluid)
                self.Q = PropsSI('Q', 'T', T, 'S', s, fluid)

            elif p is not None and h is not None :
                self.p = p
                self.h = h
                self.T = PropsSI('T', 'P', p, 'H', h, fluid)
                self.s = PropsSI('S', 'P', p, 'H', h, fluid)
                self.Q = PropsSI('Q', 'P', p, 'H', h, fluid)

            elif p is not None and s is not None :
                self.p = p
                self.s = s
                self.T = PropsSI('T', 'P', p, 'S', s, fluid)
                self.h = PropsSI('H', 'P', p, 'S', s, fluid)
                self.Q = PropsSI('Q', 'P', p, 'S', s, fluid)

            elif h is not None and s is not None :
                self.h = h
                self.s = s
                self.T = PropsSI('T', 'H', h, 'S', s, fluid)
                self.p = PropsSI('P', 'H', h, 'S', s, fluid)
                self.Q = PropsSI('Q', 'H', h, 'S', s, fluid)
    
    def _exergy(self, T0):
        """
        Calculate the specific exergy of the fluid state.
        Parameters:
        T0 (float): Reference temperature in Kelvin.
        Returns:
        float: Specific exergy in J/kg.
        """
        if self.h is not None and self.s is not None:
            self.exergy = self.h - T0 * self.s
        else:
            raise ValueError("Enthalpy (h) and Entropy (s) must be defined to calculate exergy.")
        return self.exergy
    
'''
    
state_in = state(p=4e5, Q = 0.5, fluid='R290') #inside point of curve
state_out = state(p=4e5, s = 2500, fluid='R290') #outside point of curve
state_out_2 = state(p=4e5, s = 500, fluid='R290') #outside point of curve

T_in, p_in, Q_in, h_in, s_in = state_in.T, state_in.p, state_in.Q, state_in.h, state_in.s
T_out, p_out, Q_out, h_out, s_out = state_out.T, state_out.p, state_out.Q, state_out.h, state_out.s
T_out_2, p_out_2, Q_out_2, h_out_2, s_out_2 = state_out_2.T, state_out_2.p, state_out_2.Q, state_out_2.h, state_out_2.s

# Test cases for point below saturation curve

state_test = state(T=T_in, Q = Q_in, fluid='R290')
state_test_2 = state(p=p_in, Q = Q_in, fluid='R290')
#state_test_3 = state(h=h_in, Q = Q_in, fluid='R290')
#state_test_4 = state(s=s_in, Q = Q_in, fluid='R290')

state_test_5 = state(T=T_in, p = p_in, fluid='R290')
#state_test_6 = state(T=T_in, h = h_in, fluid='R290')
state_test_7 = state(T=T_in, s = s_in, fluid='R290')
state_test_8 = state(p=p_in, h = h_in, fluid='R290')
state_test_9 = state(p=p_in, s = s_in, fluid='R290')
state_test_10 = state(h=h_in, s = s_in, fluid='R290')

# Test cases for point beside saturation curve
state_test = state(T=T_out, s = s_out, fluid='R290')
state_test_2 = state(p=p_out, s = s_out, fluid='R290')
state_test_3 = state(h=h_out, s = s_out, fluid='R290')
state_test_4 = state(T=T_out, p = p_out, fluid='R290')
#state_test_5 = state(T=T_out, h = h_out, fluid='R290')
state_test_6 = state(p=p_out, h = h_out, fluid='R290')
state_test_7 = state(T=T_out_2, s = s_out_2, fluid='R290')
state_test_8 = state(p=p_out_2, s = s_out_2, fluid='R290')
state_test_9 = state(h=h_out_2, s = s_out_2, fluid='R290')
state_test_10 = state(T=T_out_2, p = p_out_2, fluid='R290')
#state_test_11 = state(T=T_out_2, h = h_out_2, fluid='R290')
state_test_12 = state(p=p_out_2, h = h_out_2, fluid='R290')

import matplotlib.pyplot as plt

fluid = 'R290'
T_min = PropsSI('TMIN', fluid)
T_max = PropsSI('TCRIT', fluid) - 1  # Avoid critical point

T_range = np.linspace(T_min, T_max, 300)
s_liq = [PropsSI('S', 'T', T, 'Q', 0, fluid) for T in T_range]
s_vap = [PropsSI('S', 'T', T, 'Q', 1, fluid) for T in T_range]

plt.figure(figsize=(8,6))
plt.plot(s_liq, T_range, label='Saturated Liquid')
plt.plot(s_vap, T_range, label='Saturated Vapor')
plt.scatter([state_in.s], [state_in.T], color='red', label='State In', zorder=5)
plt.annotate('In', (state_in.s, state_in.T), textcoords="offset points", xytext=(10,10), ha='center', color='red')
plt.scatter([state_out.s], [state_out.T], color='blue', label='State Out', zorder=5)
plt.annotate('Out', (state_out.s, state_out.T), textcoords="offset points", xytext=(10,10), ha='center', color='blue')
plt.scatter([state_out_2.s], [state_out_2.T], color='green', label='State Out 2', zorder=5)
plt.annotate('Out 2', (state_out_2.s, state_out_2.T), textcoords="offset points", xytext=(10,10), ha='center', color='green')
plt.xlabel('Entropy [J/kg-K]')
plt.ylabel('Temperature [K]')
plt.title('T-s Diagram of Propane (R290)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

