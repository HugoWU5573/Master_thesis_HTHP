from CoolProp.CoolProp import PropsSI

class Compressor():

    def __init__(self, BVR, eta_v, eta_is_max, eta_elme = 0.98):
        self.BVR = BVR
        self.eta_v = eta_v
        self.eta_is_max = eta_is_max
        self.eta_elme = eta_elme

    def pout(self, P_el, mdot_wf, state_in, fluid):
        """
        Calculate the outlet pressure of the compressor based on the inlet state, electrical power input, and mass flow rate with respect 
        to the 3 parameters model.
        
        Parameters:
        state_in (state): The inlet state of the fluid.
        P_el (float): The electrical power input to the compressor in Watts.
        mdot_wf (float): The mass flow rate of the working fluid in kg/s.
        fluid (str): The working fluid.
        
        Returns:
        float: The outlet pressure in Pa.
        """
        if state_in.p is None:
            raise ValueError("Inlet pressure (p) must be defined to calculate outlet pressure.")
        
        # Isentropic compression 
        v_su = PropsSI('V', 'T', state_in.T, 'P', state_in.p, fluid)             # Specific volume at inlet [m^3/kg]
        v_ad = v_su * self.BVR                                                   # Specific volume at outlet assuming isentropic compression [m^3/kg]
        h_ad = PropsSI('H', 'P', state_in.p * self.BVR, 'S', state_in.s, fluid)  # Enthalpy at outlet assuming isentropic compression [J/kg]
        p_ad = PropsSI('P', 'V', v_ad, 'H', h_ad, fluid)                         # Outlet pressure assuming isentropic compression [Pa]
        w_su_ad = h_ad - state_in.h                                              # Specific work for isentropic compression [J/kg]
        
        # Isochoric compression with electrical power input
        w_tot = P_el * self.eta_elme / (mdot_wf * self.eta_v)                    # Total specific work based on electrical power input [J/kg]
        w_ad_ex = w_tot * self.eta_is_max - w_su_ad                              # Specific work for isentropic compression based on efficiency [J/kg]
        p_ex = w_ad_ex / v_ad + p_ad                                             # Outlet pressure based on efficiency [Pa]
        T_ex = PropsSI('T', 'P', p_ex, 'H', h_ad + w_ad_ex, fluid)               # Outlet temperature based on efficiency [K]

        return p_ex, T_ex
    

    def modelCompressor2(self, P_el, p_ex, state_in, fluid):
        """
        2-parameter model for compressor outlet state calculation.
        
        Parameters:
        state_in (state): The inlet state of the fluid.
        P_el (float): The electrical power input to the compressor in Watts.
        mdot_wf (float): The mass flow rate of the working fluid in kg/s.
        fluid (str): The working fluid.
        
        Returns:
        state: The outlet state of the fluid after compression.
        """
        if state_in.p is None  or state_in.s is None:
            raise ValueError("Inlet pressure (p), enthalpy (h), and entropy (s) must be defined to calculate outlet state.")
        
        # Isentropic compression
        v_su = 1/PropsSI('D', 'T', state_in.T, 'P', state_in.p, fluid)             # Specific volume at inlet [m^3/kg]
        v_ad = v_su * self.BVR                                                   # Specific volume at outlet assuming isentropic compression [m^3/kg]
        h_ad = PropsSI('H', 'D', 1/v_ad, 'S', state_in.s, fluid)                   # Enthalpy at outlet assuming isentropic compression [J/kg]
        p_ad = PropsSI('P', 'D', 1/v_ad, 'H', h_ad, fluid)                         # Outlet pressure assuming isentropic compression [Pa]
        w_su_ad = h_ad - state_in.h                                              # Specific work for isentropic compression [J/kg]

        # Isochoric compression with electrical power input
        w_ad_ex = v_ad * (p_ex - p_ad)                                           # Specific work for isentropic compression based on efficiency [J/kg]
        T_ex = PropsSI('T', 'P', p_ex, 'H', h_ad + w_ad_ex, fluid)               # Outlet temperature based on efficiency [K]
        w_tot = (w_ad_ex + w_su_ad) / self.eta_is_max                            # Total specific work based on efficiency [J/kg]
        mdot_wf = P_el * self.eta_elme / (w_tot * self.eta_v)                    # Mass flow rate based on electrical power input [kg/s]

        return mdot_wf, T_ex 





