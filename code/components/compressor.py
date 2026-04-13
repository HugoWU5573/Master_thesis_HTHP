import CoolProp
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import CoolProp
from scipy.optimize import fsolve, minimize, brentq

class Compressor_LP():
    def __init__(self) :
        # Geometric parameters
        self.BVR = 3.273206992615867
        self.bore = 41e-3
        self.stroke = 39.3e-3
        self.n_cylinders = 4 
        self.V_s = self.bore**2 * np.pi/4 * self.stroke * self.n_cylinders

        # Coefficients for efficiency models
        self.coeffs_eta_elme = np.array([[-0.07816065,  0.81897356, -2.53186216,  3.38938997],  #25 Hz
                                            [-0.06039962,  0.61249275, -1.83029312,  2.65642627]]) #50 Hz
        self.coeffs_eta_v = np.array([[-0.0675744911611597, 0.96117376],                        #25 Hz
                                        [-0.0675744911611597, 1.00639232]])                        #50 Hz                    
        self.coeffs_eta_is_max = np.array([0.7041695778652344,                                  #25 Hz
                                           0.6623896446847006])                                 #50 Hz
    def mass_flow(self, v1, v2, N) :
        ratio = v1 / v2
        grad = (self.coeffs_eta_v[1] - self.coeffs_eta_v[0]) / (50 - 25)
        coeffs = self.coeffs_eta_v[0] + grad * (N - 25)
        eta_v = np.polyval(coeffs, ratio)
        mdot = eta_v * self.V_s * N/2 / v1
        return mdot
    
    def eta_elme(self, v1, v2, N) :
        ratio = v1 / v2
        grad = (self.coeffs_eta_elme[1] - self.coeffs_eta_elme[0]) / (50 - 25)
        coeffs = self.coeffs_eta_elme[0] + grad * (N - 25)
        eta_elme = np.polyval(coeffs, ratio)
        return eta_elme
        
    def eta_is_max(self, N) :
        grad = (self.coeffs_eta_is_max[1] - self.coeffs_eta_is_max[0]) / (50 - 25)
        eta_is_max = self.coeffs_eta_is_max[0] + grad * (N - 25)
        return eta_is_max
    
    def Solve(self, state_in, p_2, N) :
        '''
        if state_in.Q != -1: 
            raise ValueError("Inlet state must be a saturated state (Q = -1) for this compressor model.")
        '''
        
        # Isentropic compression
        heos = state_in.heos
        heos.update(CoolProp.HmassP_INPUTS, state_in.h, state_in.p)
        v_1 = 1/heos.rhomass()
        v_ad = v_1 / self.BVR
        heos.update(CoolProp.DmassSmass_INPUTS, 1/v_ad, state_in.s)
        h_ad = heos.hmass()
        p_ad = heos.p()
        w_su_ad = h_ad - state_in.h

        #Isochoric compression 
        w_ad_ex = v_ad * (p_2 - p_ad)
        eta_is_max = self.eta_is_max(N)
        w_tot = (w_ad_ex + w_su_ad) / eta_is_max

        # Discharge state
        heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_tot, p_2)
        T_2 = heos.T()
        v_2 = 1/heos.rhomass()

        mdot = self.mass_flow(v_1, v_2, N)
        eta_elme = self.eta_elme(v_1, v_2, N)
        P_el = mdot * w_tot / eta_elme

        return T_2, P_el, mdot 


    def get_points_between(self, state_in, state_out, n_points=100):
        heos = state_in.heos
        # Compute the isentropic compression 
        p_2 = state_out.p
        heos.update(CoolProp.PSmass_INPUTS, p_2, state_in.s)
        h_2_is = heos.hmass()
        eta_is_mean = (h_2_is - state_in.h) / (state_out.h - state_in.h)

        p_min = state_in.p
        p_max = state_out.p
        p = np.linspace(p_min, p_max, n_points)
        T = np.zeros(n_points)
        s = np.zeros(n_points)
        h = np.zeros(n_points)
        for i, p_val in enumerate(p):
            try : 
                heos.update(CoolProp.PSmass_INPUTS, p_val, state_in.s)
                h_is = heos.hmass()
                h_val = state_in.h + (h_is - state_in.h) / eta_is_mean
                heos.update(CoolProp.HmassP_INPUTS, h_val, p_val)
                s[i] = heos.smass()
                T[i] = heos.T()
                h[i] = h_val
            except ValueError as e:
                T[i] = float('nan')
                s[i] = float('nan')
                h[i] = float('nan')
        return T, s, p, h
    


class Compressor_HP():
    def __init__(self) :
        # Geometric parameters
        self.BVR = 2.559206490261362
        self.bore = 41e-3
        self.stroke = 27.3e-3
        self.n_cylinders = 4 
        self.V_s = self.bore**2 * np.pi/4 * self.stroke * self.n_cylinders

        # Coefficients for efficiency models
        self.coeffs_eta_elme = np.array([-0.00899985,  0.0856807,  -0.24639968,  1.20386369])   #50 Hz
        self.coeffs_eta_v = np.array([-0.08511643151843135, 1.01003541])                        #50 Hz
        self.coeffs_eta_is_max = 0.6823063620613579                                             #50 Hz
        
    def mass_flow(self, v1, v2, N) :
        ratio = v1 / v2
        '''
        grad = (self.coeffs_eta_v[1] - self.coeffs_eta_v[0]) / (50 - 25)
        coeffs = self.coeffs_eta_v[0] + grad * (N - 25)
        '''
        coeffs = self.coeffs_eta_v
        eta_v = np.polyval(coeffs, ratio)
        mdot = eta_v * self.V_s * N/2 / v1
        return mdot
    
    def eta_elme(self, v1, v2, N) :
        ratio = v1 / v2
        '''
        grad = (self.coeffs_eta_elme[1] - self.coeffs_eta_elme[0]) / (50 - 25)
        coeffs = self.coeffs_eta_elme[0] + grad * (N - 25)
        '''
        coeffs = self.coeffs_eta_elme

        eta_elme = np.polyval(coeffs, ratio)
        return eta_elme
        
    def eta_is_max(self, N) :
        '''
        grad = (self.coeffs_eta_is_max[1] - self.coeffs_eta_is_max[0]) / (50 - 25)
        eta_is_max = self.coeffs_eta_is_max[0] + grad * (N - 25)
        '''
        eta_is_max = self.coeffs_eta_is_max
        return eta_is_max
    
    def Solve(self, state_in, p_2, N) :
        '''
        if state_in.Q != -1: 
            raise ValueError("Inlet state must be a saturated state (Q = -1) for this compressor model.")
        '''
        
        # Isentropic compression
        heos = state_in.heos
        heos.update(CoolProp.HmassP_INPUTS, state_in.h, state_in.p)
        v_1 = 1/heos.rhomass()
        v_ad = v_1 / self.BVR
        heos.update(CoolProp.DmassSmass_INPUTS, 1/v_ad, state_in.s)
        h_ad = heos.hmass()
        p_ad = heos.p()
        w_su_ad = h_ad - state_in.h

        #Isochoric compression 
        w_ad_ex = v_ad * (p_2 - p_ad)
        eta_is_max = self.eta_is_max(N)
        w_tot = (w_ad_ex + w_su_ad) / eta_is_max

        # Discharge state
        heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_tot, p_2)
        T_2 = heos.T()
        v_2 = 1/heos.rhomass()

        mdot = self.mass_flow(v_1, v_2, N)
        eta_elme = self.eta_elme(v_1, v_2, N)
        P_el = mdot * w_tot / eta_elme

        return T_2, P_el, mdot 
    
    def get_points_between(self, state_in, state_out, n_points=100):
        heos = state_in.heos
        # Compute the isentropic compression 
        p_2 = state_out.p
        heos.update(CoolProp.PSmass_INPUTS, p_2, state_in.s)
        h_2_is = heos.hmass()
        eta_is_mean = (h_2_is - state_in.h) / (state_out.h - state_in.h)

        p_min = state_in.p
        p_max = state_out.p
        p = np.linspace(p_min, p_max, n_points)
        T = np.zeros(n_points)
        s = np.zeros(n_points)
        h = np.zeros(n_points)
        for i, p_val in enumerate(p):
            try : 
                heos.update(CoolProp.PSmass_INPUTS, p_val, state_in.s)
                h_is = heos.hmass()
                h_val = state_in.h + (h_is - state_in.h) / eta_is_mean
                heos.update(CoolProp.HmassP_INPUTS, h_val, p_val)
                s[i] = heos.smass()
                T[i] = heos.T()
                h[i] = h_val
            except ValueError as e:
                T[i] = float('nan')
                s[i] = float('nan')
                h[i] = float('nan')
        return T, s, p, h


class Compressor_3_params():

    def __init__(self, BVR, eta_v, eta_is_max, eta_elme = 0.98):
        self.BVR = BVR
        self.eta_v = eta_v
        self.eta_is_max = eta_is_max
        self.eta_elme = eta_elme

    
    def Solve(self, state_in, p_ex, mdot_wf):
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
        
        heos = state_in.heos
        
        # Isentropic compression
        heos.update(CoolProp.PT_INPUTS, state_in.p, state_in.T)
        v_su = 1/heos.rhomass()                                                    # Specific volume at inlet [m^3/kg]
        v_ad = v_su / self.BVR 
        heos.update(CoolProp.DmassSmass_INPUTS, 1/v_ad, state_in.s)                # Specific volume at outlet assuming isentropic compression [m^3/kg]
        h_ad = heos.hmass()                                                        # Enthalpy at outlet assuming isentropic compression [J/kg]
        p_ad = heos.p()                                                            # Outlet pressure assuming isentropic compression [Pa]
        w_su_ad = h_ad - state_in.h                                                # Specific work for isentropic compression [J/kg]

        # Isochoric compression with electrical power input
        w_ad_ex = v_ad * (p_ex - p_ad)                                             # Specific work for isentropic compression based on efficiency [J/kg]

        # Scaling 
        w_tot = (w_ad_ex + w_su_ad) / self.eta_is_max                              # Total specific work based on efficiency [J/kg]
        P_el = w_tot * mdot_wf / self.eta_elme / self.eta_v
        
        heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_tot, p_ex)
        T_ex = heos.T()                     # Mass flow rate based on electrical power input [kg/s]

        return P_el, T_ex 
    

class Compressor_2_param():
    def __init__(self, cycle, eta_v, eta_is_max, fluid = 'R290', eta_elme = 0.98):
        self.cycle = cycle
        self.eta_v = eta_v
        self.eta_is_max = eta_is_max
        self.eta_elme = eta_elme
        self.fluid = fluid

    def Solve(self, p_ex, state_in, mdot_wf=None, mode="Dimensional"):
        """
        2-parameter model for compressor outlet state calculation.
        
        Parameters:
        p_ex (float): The outlet pressure of the compressor in Pa.
        state_in (state): The inlet state of the fluid.
        mdot_wf (float): The mass flow rate of the working fluid in kg/s.
        mode (str): Calculation mode, either "Dimensional" or "Non-Dimensional".
        
        Returns:
        P_el (float): The electrical power input to the compressor in Watts.
        T_ex (float): The outlet temperature of the fluid after compression in K.
        """
        if state_in.p is None  or state_in.s is None:
            raise ValueError("Inlet pressure (p), enthalpy (h), and entropy (s) must be defined to calculate outlet state.")
        
        if mode not in ["Dimensional", "Non-Dimensional"]:
            raise ValueError("Mode must be either 'Dimensional' or 'Non-Dimensional'.")
        
        # Isentropic compression
        heos = state_in.heos
        heos.update(CoolProp.PSmass_INPUTS, p_ex, state_in.s)
        h_ex_s = heos.hmass()
        h_ex = (h_ex_s - state_in.h)/self.eta_is_max + state_in.h
        heos.update(CoolProp.HmassP_INPUTS, h_ex, p_ex)
        T_ex = heos.T()
        w_tot = h_ex - state_in.h

        if mode == "Dimensional" :
            P_el =  mdot_wf * self.eta_v * w_tot / self.eta_elme
        elif mode == "Non-Dimensional" :
            P_el = None

        w_real = w_tot * self.eta_v
        heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_real, p_ex)
        T_ex = heos.T()
        


        #self.cycle.transforms.append(Transform(label_in='Compressor Inlet', label_out='Compressor Outlet', type='comp'))                                               

        return P_el, T_ex
    
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
                T[i] = self.Solve(p_ex=p_val, state_in=state_in, mode="Non-Dimensional")[1]
                heos.update(CoolProp.PT_INPUTS, p_val, T[i])
                s[i] = heos.smass()
                h[i] = heos.hmass()
            except ValueError as e:
                T[i] = float('nan')
                s[i] = float('nan')
                h[i] = float('nan')
        return T, s, p, h
    
    def energy_analysis(self, P_el, state_in, state_out, mdot_wf) : 
        """
        Placeholder for energy analysis of the compressor.
        """
        P_mec = P_el * self.eta_elme

        dict_energy = {
            'P_{el}': P_el,
            'P_{loss}': P_el - P_mec
        }

        return dict_energy
    
    def exergy_analysis(self, T0, P0, P_el, state_in, state_out, mdot_wf) :

        """
        Placeholder for exergy analysis of the compressor.
        """
        P_mec = P_el * self.eta_elme
        P_irr = P_mec - mdot_wf * (state_out.exergy(T0, P0) - state_in.exergy(T0, P0))  

        dict_exergy = {
            'P_{el}': P_el,
            'P_{loss}': P_el - P_mec,
            'P_{irr}': P_irr
        }

        return dict_exergy


if __name__ == "__main__" :
    # Theoretical graph
    from state import State
    '''

    
    T = 300
    p = PropsSI('P', 'T', T, 'Q', 1, 'R290')
    state_in = State(T=T+3, p=p, fluid='R290')
    compressor = Compressor_2_param(eta_v=0.95, eta_is_max=0.65, eta_elme=0.98)
    mdot_wf, T_ex = compressor.Solve(P_el=5e3, p_ex=2*p, state_in=state_in)
    print(f'Mass flow rate: {mdot_wf} kg/s')
    print(f'Outlet temperature: {T_ex} K')
    print(PropsSI('Q', 'T', T_ex, 'P', 2*p, 'R290'))
    '''
    '''
    heos = CoolProp.AbstractState("HEOS", "R290")

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
            state_in = State(heos,T=T_in[k], p=p_in[k])
            compressor = Compressor_3_params(BVR=2.1260, eta_v=0.95, eta_is_max=0.65, eta_elme=0.98)
            mdot_wf= compressor.Solve(P_el=P_comp, p_ex=p_out[j, k], state_in=state_in)[0]
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

    BVR = np.linspace(2, 5, 4)
    pi = np.linspace(1, 5, 10)
    w_comp_BVR = np.zeros((len(pi), len(BVR)))
    p_in = 5e5 
    T_in = PropsSI('T', 'P', p_in, 'Q', 1, 'R290') + 3

    #p_out = 3 * p_in

    for i in range(len(pi)) :
        p_out = p_in * pi[i]
        for j in range(len(BVR)) :
            state_in = State(heos, T=T_in, p=p_in)
            compressor = Compressor_3_params(BVR=BVR[j], eta_v=0.95, eta_is_max=0.65, eta_elme=0.98)
            mdot_wf= compressor.Solve(P_el=P_comp, p_ex=p_out, state_in=state_in)[0]
            w_comp_BVR[i][j] = P_comp / mdot_wf / 1000 # in kJ/kg

    plt.figure(figsize=(8,6))
    for j in range(len(BVR)):
        plt.plot(pi, w_comp_BVR[:,j], marker='o', label=f'BVR: {BVR[j]:.0f}')
    plt.xlabel('Pressure Ratio')
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
        state_in = State(heos, T=T_in, p=p_in)
        compressor = Compressor_3_params(BVR=2.1260, eta_v=0.95, eta_is_max=eta_is_max[j], eta_elme=0.98)
        mdot_wf= compressor.Solve(P_el=P_comp, p_ex=p_out, state_in=state_in)[0]
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
        state_in = State(heos, T=T_in, p=p_in)
        compressor = Compressor_3_params(BVR=2.1260, eta_v=eta_v[j], eta_is_max=0.65, eta_elme=0.98)
        mdot_wf= compressor.Solve(P_el=P_comp, p_ex=p_out, state_in=state_in)[0]
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
    heos = CoolProp.AbstractState("HEOS", "R290")
    T_sup = 10
    T_evap = np.array([273.15 + 10, 273.15 + 15, 273.15 + 20, 273.15 + 25])  # K
    T_cond = np.linspace(273.15 + 25, 273.15 + 90, 40)  # K
    p_evap = PropsSI('P', 'T', T_evap, 'Q', 1, 'R290')
    p_cond = PropsSI('P', 'T', T_cond, 'Q', 0, 'R290')

    BVR = 3
    eta_is_max = 0.7
    V_s = 8.5e-5
    N = 25 

    comp = Compressor_3_params(BVR=BVR, eta_v=None, eta_is_max=eta_is_max, eta_elme=1)
    
    '''
    P_el = np.zeros((len(T_cond), len(T_evap)))
    T_2 = np.zeros((len(T_cond), len(T_evap)))
    volume_ratio = np.zeros((len(T_cond), len(T_evap)))
    pressure_ratio = np.zeros((len(T_cond), len(T_evap)))
    eta_is_exp = np.zeros((len(T_cond), len(T_evap)))
    m_dot = 10
    w_tot = np.zeros((len(T_cond), len(T_evap)))
    v_1 = np.zeros((len(T_cond), len(T_evap)))
    pressure_ratio = np.zeros((len(T_cond), len(T_evap)))
    p_ad = np.zeros(len(T_evap))

    for i, T_c in enumerate(T_cond):
        for j, T_e in enumerate(T_evap):
            state_in = State(heos, T=T_e + T_sup, p=p_evap[j])
            v_1[i,j] = 1/PropsSI('D', 'T', T_e + T_sup, 'P', p_evap[j], 'R290')
            eta_v = (N * V_s) / (v_1[i,j] * m_dot)
            comp.eta_v = eta_v
            specific_work, T_2[i,j], w_tot[i,j] = comp.Solve(state_in=state_in, p_ex=p_cond[i])
            P_el[i,j] = specific_work * m_dot
            v_2 = 1/PropsSI('D', 'T', T_2[i,j], 'P', p_cond[i], 'R290')
            volume_ratio[i,j] = v_1[i,j] / v_2
            pressure_ratio[i,j] = p_cond[i] / p_evap[j]
            h_1 = PropsSI('H', 'T', T_e + T_sup, 'P', p_evap[j], 'R290')
            h_2 = PropsSI('H', 'T', T_2[i,j], 'P', p_cond[i], 'R290')
            h_2is = PropsSI('H', 'P', p_cond[i], 'S', PropsSI('S', 'T', T_e + T_sup, 'P', p_evap[j], 'R290'), 'R290')
            eta_is_exp[i,j] = (h_2is - h_1) / (h_2 - h_1)
            pressure_ratio[i,j] = p_cond[i] / p_evap[j]
            p_ad[j] = PropsSI('P', 'D', 1/(v_1[i,j]/BVR), 'S', PropsSI('S', 'T', T_e + T_sup, 'P', p_evap[j], 'R290'), 'R290')
    
    plt.figure()
    for i in range(len(T_evap)):
        plt.plot(volume_ratio[:,i], eta_is_exp[:,i], label='$T_{evap} =$ '+str(T_evap[i]-273.15)+'°C')
        #plt.plot(pressure_ratio[:,i], eta_is_exp[:,i], label='T_evap = '+str(T_evap[i]-273.15)+'°C')
    plt.axvline(x=BVR, color='black', linestyle='dotted', label='BVR')
    plt.axhline(y=eta_is_max, color='black', linestyle='dashed', label='Max Isentropic Efficiency')
    plt.xlabel(r'$v_1 / v_2$')
    plt.ylabel(r'$\eta_{is, exp}$')
    plt.legend()
    plt.tight_layout()
    #plt.show()

    
    plt.figure()
    for i in range(len(T_evap)):
        plt.plot(volume_ratio[:,i], P_el[:,i]/1e3, label='$T_{evap} =$ '+str(T_evap[i]-273.15)+'°C')
        #plt.plot(v_2[:,i], label='v_2')
    plt.xlabel(r'$v_1 / v_2$')
    plt.ylabel(r'$P_{el}$ [kW]')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    

    plt.figure()
    for i in range(len(T_evap)):
        plt.plot(volume_ratio[:,i], T_2[:,i]-273.15, label='$T_{evap} =$ '+str(T_evap[i]-273.15)+'°C')
        #plt.plot(v_2[:,i], label='v_2')
    plt.xlabel(r'$v_1 / v_2$')
    plt.ylabel(r'$T_2$ [°C]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    '''
    state_in = State(heos, T=300, p=5e5)
    p_ex = np.linspace(6e5, 40e5, 100)
    heos = state_in.heos
    eta_is_max = 0.4
    BVR = 3
    w_su_ad = np.zeros(len(p_ex))
    w_ad_ex = np.zeros(len(p_ex))
    w_tot = np.zeros(len(p_ex))
    ratio_volume = np.zeros(len(p_ex))
    eta_is = np.zeros(len(p_ex))
    w_is = np.zeros(len(p_ex))
    p_ad = 0

    for i in range(len(p_ex)) :

        # Isentropic compression
        heos.update(CoolProp.PT_INPUTS, state_in.p, state_in.T)
        v_su = 1/heos.rhomass()                                                    # Specific volume at inlet [m^3/kg]
        v_ad = v_su / BVR 
        heos.update(CoolProp.DmassSmass_INPUTS, 1/v_ad, state_in.s)                # Specific volume at outlet assuming isentropic compression [m^3/kg]
        h_ad = heos.hmass()                                                        # Enthalpy at outlet assuming isentropic compression [J/kg]
        p_ad = heos.p()                                                            # Outlet pressure assuming isentropic compression [Pa]
        w_su_ad[i] = h_ad - state_in.h                                                # Specific work for isentropic compression [J/kg]

        # Isochoric compression with electrical power input
        w_ad_ex[i] = v_ad * (p_ex[i] - p_ad)                                             # Specific work for isentropic compression based on efficiency [J/kg]

        # Scaling 
        w_tot[i] = (w_ad_ex[i] + w_su_ad[i]) / eta_is_max                              # Total specific work based on efficiency [J/kg]
        
        heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_tot[i], p_ex[i])
        T_ex = heos.T()                     # Mass flow rate based on electrical power input [kg/s]
        v_ex = 1/heos.rhomass()
        ratio_volume[i] = v_su / v_ex

        heos.update(CoolProp.PSmass_INPUTS, p_ex[i], state_in.s)
        hex_is = heos.hmass()
        w_is[i] = hex_is - state_in.h
        eta_is[i] = w_is[i] / w_tot[i]
    
    fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    axs[0].plot(ratio_volume, w_tot/1e3, label=r'$w_{tot}$')
    axs[0].plot(ratio_volume, w_su_ad/1e3, label=r'$w_{su->ad}$')
    axs[0].plot(ratio_volume, w_ad_ex/1e3, label=r'$w_{ad->ex}$')
    axs[0].plot(ratio_volume, w_is/1e3, label=r'$w_{is}$')
    axs[1].set_xlabel(r'$v_1 / v_2$')
    axs[0].set_ylabel(r'$w$ [kJ/kg]')
    axs[0].legend(frameon=False)
    axs[0].axvline(x=BVR, color='black', linestyle='dotted', label='BVR')

    axs[1].plot(ratio_volume, eta_is)
    axs[1].set_ylabel(r'$\eta_{is,exp}$')
    axs[1].axvline(x=BVR, color='black', linestyle='dotted', label='BVR')
    axs[1].axhline(y=eta_is_max, color='black', linestyle='dashed', label='Max Isentropic Efficiency')
    axs[1].legend(frameon=False)
    plt.tight_layout()

    fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)

    axs[0].plot(p_ex/state_in.p, w_tot/1e3, label=r'$w_{tot}$')
    axs[0].plot(p_ex/state_in.p, w_su_ad/1e3, label=r'$w_{su->ad}$')
    axs[0].plot(p_ex/state_in.p, w_ad_ex/1e3, label=r'$w_{ad->ex}$')
    axs[0].plot(p_ex/state_in.p, w_is/1e3, label=r'$w_{is}$')
    axs[1].set_xlabel(r'$p_2 / p_1$')
    axs[0].set_ylabel(r'$w$ [kJ/kg]')
    axs[0].legend(frameon=False)
    axs[0].axvline(x=p_ad/state_in.p, color='black', linestyle='dotted', label=r'$p_{ad}/p_{in}$')
    
    axs[1].plot(p_ex/state_in.p, eta_is)
    axs[1].set_ylabel(r'$\eta_{is,exp}$')
    axs[1].axvline(x=p_ad/state_in.p, color='black', linestyle='dotted', label=r'$p_{ad}/p_{in}$')
    axs[1].axhline(y=eta_is_max, color='black', linestyle='dashed', label='Max Isentropic Efficiency')
    axs[1].legend(frameon=False)
    plt.tight_layout()


    #plt.grid()
    plt.show()
    



