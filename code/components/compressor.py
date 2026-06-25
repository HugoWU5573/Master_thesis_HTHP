import CoolProp
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import CoolProp
from scipy.optimize import fsolve

class Compressor_LP():
    """
    Model of the Bitzer 4FESP-5Z-40S. The values of the parameters are computed in the fitting/compressor section.
    
    
    """
    
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
        Resolve the compressor outlet state given the inlet state, outlet pressure and rotational speed.

        Parameters:
            - state_in (State): The inlet thermodynamic state.
            - p_2 (float): The outlet pressure of the compressor in Pa.
            - N (float): The electrical frequency of the compressor in Hz.
        Returns:
            - h_2 (float): The enthalpy at the compressor outlet in J/kg.
            - P_el (float): The electrical power input to the compressor in Watts.
            - mdot (float): The mass flow rate of the working fluid in kg/s.

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
        h_2 = state_in.h + w_tot
        v_2 = 1/heos.rhomass()

        mdot = self.mass_flow(v_1, v_2, N)
        eta_elme = self.eta_elme(v_1, v_2, N)
        P_el = mdot * w_tot / eta_elme

        return h_2, P_el, mdot 
    
    def Solve_2(self, state_in, p_2, mdot) :
        
        '''
        Resolve the compressor outlet state given the inlet state, outlet pressure and mass flow rate.
        
        Parameters:
            - state_in (State): The inlet thermodynamic state.
            - p_2 (float): The outlet pressure of the compressor in Pa.
            - mdot (float): The mass flow rate of the working fluid in kg/s.
        Returns:
            - h_2 (float): The enthalpy at the compressor outlet in J/kg.
            - P_el (float): The electrical power input to the compressor in Watts.
            - N (float): The electrical frequency of the compressor in Hz.
        
        
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

        def objective(args, final=False): 
            #print(args)
            N, v_2 = args
            mdot_calc = self.mass_flow(v_1, v_2, N)
            
            #Isochoric compression 
            w_ad_ex = v_ad * (p_2 - p_ad)
            eta_is_max = self.eta_is_max(N)
            w_tot = (w_ad_ex + w_su_ad) / eta_is_max

            # Discharge state
            heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_tot, p_2)
            h_2 = state_in.h + w_tot
            v_2_calc = 1/heos.rhomass()

            #mdot = self.mass_flow(v_1, v_2_calc, N)
            eta_elme = self.eta_elme(v_1, v_2_calc, N)
            P_el = mdot * w_tot / eta_elme

            if final :
                residual = [(mdot_calc - mdot) / mdot, (v_2_calc - v_2) / v_2]
                return h_2, P_el, N
            else :
                residual = [(mdot_calc - mdot) / mdot, (v_2_calc - v_2) / v_2]
                return residual
            
        initial_guess = [50, v_1]
        N, v_2 = fsolve(objective, initial_guess)
        return objective([N, v_2], final=True)
        

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
    """
    Model of the Bitzer 4FME-9K-40S. The values of the parameters are computed in the fitting/compressor section.
    """
    def __init__(self) :
        # Geometric parameters
        self.BVR = 2.559206490261362
        self.bore = 41e-3
        self.stroke = 27.3e-3
        self.n_cylinders = 4 
        self.V_s = self.bore**2 * np.pi/4 * self.stroke * self.n_cylinders

        # Coefficients for efficiency models
        self.coeffs_eta_elme = np.array([-0.00899985,  0.0856807,  -0.24639968,  1.20386369])   #50 Hz
        self.grad_eta_elme = np.array([0.00071044, -0.00825923,  0.02806276, -0.02931855])
        self.coeffs_eta_v = np.array([-0.08511643151843135, 1.01003541])                        #50 Hz
        self.grad_eta_v = np.array([0.0, 0.00180874])
        self.coeffs_eta_is_max = 0.6823063620613579                                             #50 Hz
        self.grad_eta_is_max = -0.0016711973272213498
        
    def mass_flow(self, v1, v2, N) :
        ratio = v1 / v2
        coeffs = self.coeffs_eta_v + self.grad_eta_v * (N - 50)
        eta_v = np.polyval(coeffs, ratio)
        mdot = eta_v * self.V_s * N/2 / v1
        return mdot
    
    def eta_elme(self, v1, v2, N) :
        ratio = v1 / v2
        coeffs = self.coeffs_eta_elme + self.grad_eta_elme * (N - 50)
        eta_elme = np.polyval(coeffs, ratio)
        return eta_elme
        
    def eta_is_max(self, N) :
        eta_is_max = self.coeffs_eta_is_max + self.grad_eta_is_max * (N - 50)
        return eta_is_max
    
    def Solve(self, state_in, p_2, N) :
        '''
        Resolve the compressor outlet state given the inlet state, outlet pressure and rotational speed.

        Parameters:
            - state_in (State): The inlet thermodynamic state.
            - p_2 (float): The outlet pressure of the compressor in Pa.
            - N (float): The electrical frequency of the compressor in Hz.
        Returns:
            - h_2 (float): The enthalpy at the compressor outlet in J/kg.
            - P_el (float): The electrical power input to the compressor in Watts.
            - mdot (float): The mass flow rate of the working fluid in kg/s.

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
        h_2 = state_in.h + w_tot
        v_2 = 1/heos.rhomass()

        mdot = self.mass_flow(v_1, v_2, N)
        eta_elme = self.eta_elme(v_1, v_2, N)
        P_el = mdot * w_tot / eta_elme

        return h_2, P_el, mdot 
    
    def Solve_2(self, state_in, p_2, mdot) :
        '''
        Resolve the compressor outlet state given the inlet state, outlet pressure and mass flow rate.
        
        Parameters:
            - state_in (State): The inlet thermodynamic state.
            - p_2 (float): The outlet pressure of the compressor in Pa.
            - mdot (float): The mass flow rate of the working fluid in kg/s.
        Returns:
            - h_2 (float): The enthalpy at the compressor outlet in J/kg.
            - P_el (float): The electrical power input to the compressor in Watts.
            - N (float): The electrical frequency of the compressor in Hz.
        
        
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

        def objective(args, final=False): 
            #print(args)
            N, v_2 = args
            mdot_calc = self.mass_flow(v_1, v_2, N)
            
            #Isochoric compression 
            w_ad_ex = v_ad * (p_2 - p_ad)
            eta_is_max = self.eta_is_max(N)
            w_tot = (w_ad_ex + w_su_ad) / eta_is_max

            # Discharge state
            heos.update(CoolProp.HmassP_INPUTS, state_in.h + w_tot, p_2)
            h_2 = state_in.h + w_tot
            v_2_calc = 1/heos.rhomass()

            #mdot = self.mass_flow(v_1, v_2_calc, N)
            eta_elme = self.eta_elme(v_1, v_2_calc, N)
            P_el = mdot * w_tot / eta_elme

            if final :
                residual = [(mdot_calc - mdot) / mdot, (v_2_calc - v_2) / v_2]
                return h_2, P_el, N
            else :
                residual = [(mdot_calc - mdot) / mdot, (v_2_calc - v_2) / v_2]
                return residual
            
        initial_guess = [50, v_1]
        N, v_2 = fsolve(objective, initial_guess)
        return objective([N, v_2], final=True)
    
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
    

class Compressor_2_param():
    """
    2-parameter model for compressor outlet state calculation. Used in the thermodynamic analysis of the cycle.
    """

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
    



