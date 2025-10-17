import CoolProp
from CoolProp.CoolProp import PropsSI
from components.transform import Transform
from components.state import State
import numpy as np

class Compressor_3():

    def __init__(self, BVR, eta_v, eta_is_max, eta_elme = 0.98):
        self.BVR = BVR
        self.eta_v = eta_v
        self.eta_is_max = eta_is_max
        self.eta_elme = eta_elme

    
    def Solve(self, P_el, p_ex, state_in, fluid):
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
        v_ad = v_su * self.BVR                                                     # Specific volume at outlet assuming isentropic compression [m^3/kg]
        h_ad = PropsSI('H', 'D', 1/v_ad, 'S', state_in.s, fluid)                   # Enthalpy at outlet assuming isentropic compression [J/kg]
        p_ad = PropsSI('P', 'D', 1/v_ad, 'H', h_ad, fluid)                         # Outlet pressure assuming isentropic compression [Pa]
        w_su_ad = h_ad - state_in.h                                                # Specific work for isentropic compression [J/kg]

        # Isochoric compression with electrical power input
        w_ad_ex = v_ad * (p_ex - p_ad)                                             # Specific work for isentropic compression based on efficiency [J/kg]
        T_ex = PropsSI('T', 'P', p_ex, 'H', h_ad + w_ad_ex, fluid)                 # Outlet temperature based on efficiency [K]
        w_tot = (w_ad_ex + w_su_ad) / self.eta_is_max                              # Total specific work based on efficiency [J/kg]
        mdot_wf = P_el * self.eta_elme / (w_tot * self.eta_v)                      # Mass flow rate based on electrical power input [kg/s]

        return mdot_wf, T_ex 
    

class Compressor_2_param():
    def __init__(self, cycle, eta_v, eta_is_max, fluid = 'R290', eta_elme = 0.98):
        self.cycle = cycle
        self.eta_v = eta_v
        self.eta_is_max = eta_is_max
        self.eta_elme = eta_elme
        self.fluid = fluid

    def Solve(self, P_el, p_ex, state_in):
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
        
        self.state_in = state_in

        # Isentropic compression
        heos = state_in.heos
        heos.update(CoolProp.PSmass_INPUTS, p_ex, state_in.s)
        h_ex_s = heos.hmass()
        h_ex = (h_ex_s - state_in.h)/self.eta_is_max + state_in.h
        heos.update(CoolProp.HmassP_INPUTS, h_ex, p_ex)
        T_ex = heos.T()

        w_tot = h_ex - state_in.h
        mdot_wf = P_el * self.eta_elme / (w_tot * self.eta_v)


        #self.cycle.transforms.append(Transform(label_in='Compressor Inlet', label_out='Compressor Outlet', type='comp'))                                               

        return mdot_wf, T_ex
    
    def get_points_between(self, state_in, state_out, n_points=100):
        heos = state_in.heos
        p_min = state_in.p
        p_max = state_out.p
        p = np.linspace(p_min, p_max, n_points)
        T = np.zeros(n_points)
        s = np.zeros(n_points)
        for i, p_val in enumerate(p):
            T[i] = self.Solve(P_el=0, p_ex=p_val, state_in=state_in)[1]
            heos.update(CoolProp.PT_INPUTS, p_val, T[i])
            s[i] = heos.smass()
        return T, s
        


if __name__ == "__main__" :
    # Theoretical graph

    import matplotlib.pyplot as plt
    import numpy as np
    from state import State

    T = 300
    p = PropsSI('P', 'T', T, 'Q', 1, 'R290')
    state_in = State(T=T+3, p=p, fluid='R290')
    compressor = Compressor_2_param(eta_v=0.95, eta_is_max=0.65, eta_elme=0.98)
    mdot_wf, T_ex = compressor.Solve(P_el=5e3, p_ex=2*p, state_in=state_in)
    print(f'Mass flow rate: {mdot_wf} kg/s')
    print(f'Outlet temperature: {T_ex} K')
    print(PropsSI('Q', 'T', T_ex, 'P', 2*p, 'R290'))



    '''

    p_in = np.linspace(5e5, 40e5, 100)
    T_in = PropsSI('T', 'P', p_in, 'Q', 1, 'R290') + 3

    pi = [1.5, 2, 2.2477]
    p_out = np.zeros((len(pi), len(p_in)))

    P_comp = 5e3 
    w_comp = np.zeros((len(pi), len(p_in)))

    for k in range(len(p_in)) : 
        for j in range(len(pi)) :
            ratio = pi[j]
            p_out[j, k] = p_in[k] * ratio
            state_in = State(T=T_in[k], p=p_in[k], fluid='R290')
            compressor = Compressor(BVR=1/2.1260, eta_v=0.95, eta_is_max=0.65, eta_elme=0.98)
            mdot_wf= compressor.modelCompressor2(P_el=P_comp, p_ex=p_out[j, k], state_in=state_in, fluid='R290')[0]
            w_comp[j, k] = P_comp / mdot_wf / 1000 # in kJ/kg

    plt.figure(figsize=(8,6))
    for j in range(len(pi)) :
        plt.plot(p_in/1e5, w_comp[j, :], label=f'Pressure Ratio: {pi[j]}')
    plt.xlabel('Inlet Pressure [bar]')
    plt.legend(frameon=False)
    #plt.grid()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Compressor Work [kJ/kg]', loc='left')

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    plt.show()


    # Impact of BVR

    BVR = 1/np.linspace(1, 10, 10)
    w_comp_BVR = np.zeros((len(pi), len(BVR)))
    p_in = 15e5 
    T_in = PropsSI('T', 'P', p_in, 'Q', 1, 'R290') + 3
    pi = [1.5, 2, 2.2477]

    #p_out = 3 * p_in

    for i in range(len(pi)) :
        p_out = p_in * pi[i]
        for j in range(len(BVR)) :
            state_in = State(T=T_in, p=p_in, fluid='R290')
            compressor = Compressor(BVR=BVR[j], eta_v=0.95, eta_is_max=0.65, eta_elme=0.98)
            mdot_wf= compressor.modelCompressor2(P_el=P_comp, p_ex=p_out, state_in=state_in, fluid='R290')[0]
            w_comp_BVR[i][j] = P_comp / mdot_wf / 1000 # in kJ/kg

    plt.figure(figsize=(8,6))
    for i in range(len(pi)):
        plt.plot(BVR, w_comp_BVR[i], marker='o', label=f'Pressure Ratio: {pi[i]}')
    plt.xlabel('BVR')
    plt.legend(frameon=False)
    #plt.grid()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Compressor Work [kJ/kg]', loc='left')

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    plt.show()


    # Impact of eta_is_max

    eta_is_max = np.linspace(0.5, 0.8, 4)
    w_comp_eta = np.zeros(len(eta_is_max))
    p_in = 15e5
    T_in = PropsSI('T', 'P', p_in, 'Q', 1, 'R290') + 3
    p_out = p_in * 2.2477
    for j in range(len(eta_is_max)) :
        state_in = State(T=T_in, p=p_in, fluid='R290')
        compressor = Compressor(BVR=1/2.1260, eta_v=0.95, eta_is_max=eta_is_max[j], eta_elme=0.98)
        mdot_wf= compressor.modelCompressor2(P_el=P_comp, p_ex=p_out, state_in=state_in, fluid='R290')[0]
        w_comp_eta[j] = P_comp / mdot_wf / 1000 # in kJ/kg
    plt.figure(figsize=(8,6))
    plt.plot(eta_is_max, w_comp_eta, marker='o', color='black')
    plt.xlabel('Maximum Isentropic Efficiency')
    #plt.grid()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Compressor Work [kJ/kg]', loc='left')

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    plt.show()


    # Impact of eta_v
    eta_v = np.linspace(0.7, 1, 4)
    w_comp_eta_v = np.zeros(len(eta_v))
    p_in = 15e5
    T_in = PropsSI('T', 'P', p_in, 'Q', 1, 'R290') + 3
    p_out = p_in * 2.2477
    for j in range(len(eta_v)) :
        state_in = State(T=T_in, p=p_in, fluid='R290')
        compressor = Compressor(BVR=1/2.1260, eta_v=eta_v[j], eta_is_max=0.65, eta_elme=0.98)
        mdot_wf= compressor.modelCompressor2(P_el=P_comp, p_ex=p_out, state_in=state_in, fluid='R290')[0]
        w_comp_eta_v[j] = P_comp / mdot_wf / 1000 # in kJ/kg
    plt.figure(figsize=(8,6))
    plt.plot(eta_v, w_comp_eta_v, marker='o', color = 'black')
    plt.xlabel('Volumetric Efficiency')
    #plt.grid()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Compressor Work [kJ/kg]', loc='left')
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    plt.show()
    '''