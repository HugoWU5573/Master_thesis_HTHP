from CoolProp.CoolProp import PropsSI 
import numpy as np
from scipy.optimize import brentq  # Function used for iterative root finding 
import matplotlib.pyplot as plt
from state import State

class HEX():

    """
    Creates a COUNTER-FLOW heat exchanger object based on the following inputs:
        - state_in_c: State object representing the inlet state of the cold stream
        - state_in_h: State object representing the inlet state of the hot stream
        - mdot: list of mass flow rates [kg/s] for cold and hot streams
        - fluid: list of fluid names for cold and hot streams
        - N : number of plates in the heat exchanger
        - L : heat exchanger length [m] (default = 0.3 m)
        - W : heat exchanger width [m] (default = 0.1 m)
        - w : width of a single channel [m] (default = 1.5 mm)
        - beta : chevron angle [degrees] (default = 45 degrees)
        - Rcond: thermal resistance of the wall separating the two streams [m²K/W] (default = 0)

    """
    def __init__(self, state_in_c, state_in_h, mdot, fluid, N, L = 0.3, W=0.1, w=1.5e-3, beta=45, Rcond = 0):

        self.Tin_c = state_in_c.T
        self.Tin_h = state_in_h.T
        self.pin_c = state_in_c.p
        self.pin_h = state_in_h.p
        self.mdot_c = mdot[0]
        self.mdot_h = mdot[1]
        self.fluid_c = fluid[0]
        self.fluid_h = fluid[1]
        self.hin_c = state_in_c.h
        self.hin_h = state_in_h.h
        self.Nb_plates = N
        self.A = L * W * (self.Nb_plates - 2)        # We do not consider the first and last plate for heat transfer
        self.W = W
        self.w = w
        self.Aflow = self.W * self.w    # Cross-sectional flow area for one channel [m2]
        self.beta = beta
        self.Rcond = Rcond
        self.condensation_start = None  # The hot stream is going from vapor to 2 phase
        self.condensation_end = None    # The hot stream is going from 2 phase to liquid
        self.evaporation_start = None   # The cold stream is going from liquid to 2 phase
        self.evaporation_end = None     # The cold stream is going from 2 phase to vapor
        self.Q = None
        self.epsilon = None


    """
    This method returns a string representation of the heat exchanger object.

    """ 
    def __str__(self):
        result = "\n=========================== PHEX ===========================\n"
        if (self.condensation_start) and (self.condensation_end):
            result += "Hot stream condenses from vapor to liquid.\n"
        elif (self.condensation_start) and (not self.condensation_end):
            result += "Hot stream starts condensing from vapor to 2 phase.\n"
        elif (not self.condensation_start) and (self.condensation_end):
            result += "Hot stream finishes condensing from 2 phase to liquid.\n"
        else :
            result += "Hot stream does not condense.\n"
        
        if (self.evaporation_start) and (self.evaporation_end):
            result += "Cold stream evaporates from liquid to vapor.\n"
        elif (self.evaporation_start) and (not self.evaporation_end):
            result += "Cold stream starts evaporating from liquid to 2 phase.\n"
        elif (not self.evaporation_start) and (self.evaporation_end):
            result += "Cold stream finishes evaporating from 2 phase to vapor.\n"
        else :
            result += "Cold stream does not evaporate.\n"
        
        result += f"Heat exchanger effectiveness: {self.epsilon*100:.2f} %\n"
        result += f"Heat transfer rate: {self.Q/1000:.2f} kW,th\n"
        result += "============================================================"

        return result

    
    """
    This method calculates the maximum heat transfer rate based on the assumption of external pinching.

    """
    def _Qmax_ext(self):
        hout_h = PropsSI('H', 'T', self.Tin_c, 'P', self.pin_h, self.fluid_h)
        hout_c = PropsSI('H', 'T', self.Tin_h, 'P', self.pin_c, self.fluid_c)
        Qmax_h = self.mdot_h * (self.hin_h - hout_h)
        Qmax_c = self.mdot_c * (hout_c - self.hin_c)
        self.Qmax_ext = min(Qmax_h, Qmax_c)
        return self.Qmax_ext
    

    """
    This method implements the cell division algorithm to find the Enthalpy Vectors of both streams.
        - Input : Qmax (maximum heat transfer rate based on external pinching)
        - Output : Enthalpy Vectors of cold and hot streams

    """
    def _cell_division(self, Qmax):

        self.EnthalpyVector_h = np.array([self.hin_h - Qmax/self.mdot_h, self.hin_h])
        self.EnthalpyVector_c = np.array([self.hin_c, self.hin_c + Qmax/self.mdot_c])

        self.h_h_dew = PropsSI('H', 'P', self.pin_h, 'Q', 1, self.fluid_h)
        self.h_h_bub = PropsSI('H', 'P', self.pin_h, 'Q', 0, self.fluid_h)
        self.h_c_dew = PropsSI('H', 'P', self.pin_c, 'Q', 1, self.fluid_c)
        self.h_c_bub = PropsSI('H', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        self.N = 1 # Initial number of cells
        

        ## A. Insert phase transition enthalpies for the hot stream if applicable

            # 1. Check for potential phase transition 2 phase to Liquid (bubble point) 
        if (self.EnthalpyVector_h[0] < self.h_h_bub) and (self.EnthalpyVector_h[-1] > self.h_h_bub):
            self.EnthalpyVector_h = np.append(self.EnthalpyVector_h, self.h_h_bub)
            self.EnthalpyVector_h.sort()
            self.condensation_end = True
            self.N += 1
        else :
            self.condensation_end = False

            # 2. Check for potential phase transition Vapor to 2 phase (dew point)
        if (self.EnthalpyVector_h[0] < self.h_h_dew) and (self.EnthalpyVector_h[-1] > self.h_h_dew):
            self.EnthalpyVector_h = np.append(self.EnthalpyVector_h, self.h_h_dew)
            self.EnthalpyVector_h.sort()
            self.condensation_start = True
            self.N += 1
        else :
            self.condensation_start = False

        
        ## B. Insert phase transition enthalpies for the cold stream if applicable

            # 1. Check for potential phase transition Liquid to 2 phase (bubble point)
        if (self.EnthalpyVector_c[0] < self.h_c_bub) and (self.EnthalpyVector_c[-1] > self.h_c_bub):
            self.EnthalpyVector_c = np.append(self.EnthalpyVector_c, self.h_c_bub)
            self.EnthalpyVector_c.sort()
            self.evaporation_start = True
            self.N += 1
        else :
            self.evaporation_start = False

            # 2. Check for potential phase transition 2 phase to Vapor (dew point)
        if (self.EnthalpyVector_c[0] < self.h_c_dew) and (self.EnthalpyVector_c[-1] > self.h_c_dew):
            self.EnthalpyVector_c = np.append(self.EnthalpyVector_c, self.h_c_dew)
            self.EnthalpyVector_c.sort()
            self.evaporation_end = True
            self.N += 1
        else :
            self.evaporation_end = False


        ## C. Insert complementary phase transition enthalpies

        for j in range(self.N - 1):
            Qcell_h = self.mdot_h * (self.EnthalpyVector_h[j+1] - self.EnthalpyVector_h[j])
            Qcell_c = self.mdot_c * (self.EnthalpyVector_c[j+1] - self.EnthalpyVector_c[j])
            
            # We insert the complementary enthalpy in the enthalpy vector of the stream with the 
            # highest heat transfer in the cell (indicating that we need to split this cell)

            if Qcell_h > Qcell_c:   # In this case, h_h[j+1] is unknown -> we make an energy balance to find it
                new_h = self.mdot_c/self.mdot_h * (self.EnthalpyVector_c[j+1] - self.EnthalpyVector_c[j]) + self.EnthalpyVector_h[j]
                self.EnthalpyVector_h = np.append(self.EnthalpyVector_h, new_h)
                self.EnthalpyVector_h.sort()

            else :                  # In this case, h_c[j+1] is unknown -> we make an energy balance to find it
                new_h = self.mdot_h/self.mdot_c * (self.EnthalpyVector_h[j+1] - self.EnthalpyVector_h[j]) + self.EnthalpyVector_c[j]
                self.EnthalpyVector_c = np.append(self.EnthalpyVector_c, new_h)
                self.EnthalpyVector_c.sort()


        ## D. Verify that both vectors have the same length

        if (len(self.EnthalpyVector_c) != len(self.EnthalpyVector_h)):
            raise ValueError("Cell division algorithm failed: Enthalpy vectors have different lengths.")
        
        if (len(self.EnthalpyVector_h) != (self.N+1)):
            raise ValueError("Size of enthalpy vectors does not match the expected number of cells.")

        return self.EnthalpyVector_c, self.EnthalpyVector_h
    

    """
    This method calculates the maximum heat transfer rate based on the assumption of internal pinching.
    
    """
    def _Qmax_int(self):
        self.Qmax_int = self.Qmax_ext

        if (not self.evaporation_start) and (not self.evaporation_end) and (not self.condensation_start) and (not self.condensation_end):
            # No phase change in either stream -> no internal pinching possible
            return self.Qmax_int
        
        TemperatureVector_c = np.zeros(len(self.EnthalpyVector_c))
        TemperatureVector_h = np.zeros(len(self.EnthalpyVector_h))

        for i in range(len(self.EnthalpyVector_c)):
            TemperatureVector_c[i] = PropsSI('T', 'P', self.pin_c, 'H', self.EnthalpyVector_c[i], self.fluid_c)
            TemperatureVector_h[i] = PropsSI('T', 'P', self.pin_h, 'H', self.EnthalpyVector_h[i], self.fluid_h)
        
        for j in range(1, len(self.EnthalpyVector_c)-1):
            if (TemperatureVector_h[j] + 1e-6 < TemperatureVector_c[j]):
                # Internal pinching detected
                if (np.isclose(self.EnthalpyVector_c[j], self.h_c_bub, atol=1e-3)) :
                    # Pinch at the start of evaporation -> the cold stream sets the temperature
                    Tc_bub = PropsSI('T', 'P', self.pin_c, 'Q', 0, self.fluid_c)
                    TemperatureVector_h[j] = Tc_bub
                    self.EnthalpyVector_h[j] = PropsSI('H', 'P', self.pin_h, 'T', Tc_bub, self.fluid_h)
                
                elif (np.isclose(self.EnthalpyVector_c[j], self.h_c_dew, atol=1e-3)) :
                    # Pinch at the end of evaporation -> the cold stream sets the temperature
                    Tc_dew = PropsSI('T', 'P', self.pin_c, 'Q', 1, self.fluid_c)
                    TemperatureVector_h[j] = Tc_dew
                    self.EnthalpyVector_h[j] = PropsSI('H', 'P', self.pin_h, 'T', Tc_dew, self.fluid_h)
                
                elif (np.isclose(self.EnthalpyVector_h[j], self.h_h_dew, atol=1e-3)) :
                    # Pinch at the start of condensation -> the hot stream sets the temperature
                    Th_dew = PropsSI('T', 'P', self.pin_h, 'Q', 1, self.fluid_h)
                    TemperatureVector_c[j] = Th_dew
                    self.EnthalpyVector_c[j] = PropsSI('H', 'P', self.pin_c, 'T', Th_dew, self.fluid_c)
                
                elif (np.isclose(self.EnthalpyVector_h[j], self.h_h_bub, atol=1e-3)) :
                    # Pinch at the end of condensation -> the hot stream sets the temperature
                    Th_bub = PropsSI('T', 'P', self.pin_h, 'Q', 0, self.fluid_h)
                    TemperatureVector_c[j] = Th_bub
                    self.EnthalpyVector_c[j] = PropsSI('H', 'P', self.pin_c, 'T', Th_bub, self.fluid_c)

        # Now that we have updated the enthalpy vectors, we can calculate Qmax_int by summing the minimum heat transfer in each cell
        Qmax_int = 0
        for k in range(self.N):
            Qcell_h = self.mdot_h * (self.EnthalpyVector_h[k+1] - self.EnthalpyVector_h[k])
            Qcell_c = self.mdot_c * (self.EnthalpyVector_c[k+1] - self.EnthalpyVector_c[k])
            Qmax_int += min(Qcell_h, Qcell_c)

        self.Qmax_int = Qmax_int

        if (self.Qmax_int > self.Qmax_ext):
            raise ValueError("Qmax based on internal pinching cannot be higher than Qmax based on external pinching.")
        
        if (self.Qmax_int < 0):
            raise ValueError("Qmax based on internal pinching cannot be negative.")

        return self.Qmax_int
    

    """
    This method returns the fraction of the heat exchanger area used in cell "j".
        - Inputs :
            - cell_index : index of the cell for which we want to calculate the area fraction (j = 0, 1, ..., N-1)
            - alpha_h_j : overall heat transfer coefficient for the hot stream [W/m2K]
            - alpha_c_j : overall heat transfer coefficient for the cold stream [W/m2K]
        - Output : 
            - wj = Aj/A fraction of the heat exchanger area used in cell "j"
    
    """
    def _cell_analysis(self, cell_index, alpha_h_j, alpha_c_j):
        # Heat transfer rate in cell j
        Qj_h = self.mdot_h * (self.EnthalpyVector_h[cell_index+1] - self.EnthalpyVector_h[cell_index])
        Qj_c = self.mdot_c * (self.EnthalpyVector_c[cell_index+1] - self.EnthalpyVector_c[cell_index])
        if not (np.isclose(Qj_h, Qj_c, atol=1e-6)):
            raise ValueError("Heat transfer rates in cell j are not consistent.")
        else :
            Qj = (Qj_h + Qj_c)/2

        Th_j = PropsSI('T', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index], self.fluid_h)
        Th_j_plus_1 = PropsSI('T', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index+1], self.fluid_h)
        Tc_j = PropsSI('T', 'P', self.pin_c, 'H', self.EnthalpyVector_c[cell_index], self.fluid_c)
        Tc_j_plus_1 = PropsSI('T', 'P', self.pin_c, 'H', self.EnthalpyVector_c[cell_index+1], self.fluid_c)

        # Temperature differences at both ends of cell j
        DeltaTA_j = Th_j_plus_1 - Tc_j_plus_1
        DeltaTB_j = Th_j - Tc_j

        # Log Mean Temperature Difference (LMTD) for cell j
        if (np.isclose(DeltaTA_j, DeltaTB_j, atol=1e-6)):
            LMTDj = DeltaTA_j
        else :
            DeltaTA_j = max(DeltaTA_j, 1e-6)  # To avoid log(0) or log(negative)
            DeltaTB_j = max(DeltaTB_j, 1e-6)  # To avoid log(infinity) or log(negative)
            LMTDj = (DeltaTA_j - DeltaTB_j) / np.log(DeltaTA_j / DeltaTB_j)
        
        # Area fraction for cell j
        UArequired_j = Qj / LMTDj
        Uj = 1 / (1/alpha_h_j + self.Rcond + 1/alpha_c_j)  # Overall heat transfer coefficient for cell j (assuming no wall resistance)
        Arequired_j = UArequired_j / Uj
        wj = Arequired_j / self.A

        return wj
    

    """
    This method calculates the convective heat transfer coefficients for both streams in cell "cell_index".
        - Input : cell_index (index of the cell for which we want to calculate the heat transfer coefficients)
        - Outputs :
            - alpha_c : convective heat transfer coefficient for the cold stream [W/m2K]
            - alpha_h : convective heat transfer coefficient for the hot stream [W/m2K]
    
    """
    def _alpha(self, cell_index):

        alpha_c = None
        alpha_h = None

        ## Cold stream
        hc_start = self.EnthalpyVector_c[cell_index]
        hc_end = self.EnthalpyVector_c[cell_index+1]

        if hc_end <= self.h_c_bub or hc_start >= self.h_c_dew: 
            # Cold stream is single phase
            alpha_c = self._SinglePhase_Correlation(cell_index, "cold")
        else :
            # Cold stream is in 2 phase and evaporating
            alpha_c = self._Evaporation_Correlation(cell_index)

        ## Hot stream
        hh_start = self.EnthalpyVector_h[cell_index]
        hh_end = self.EnthalpyVector_h[cell_index+1]

        if hh_end <= self.h_h_bub or hh_start >= self.h_h_dew:
            # Hot stream is single phase liquid
            alpha_h = self._SinglePhase_Correlation(cell_index, "hot")
        else :
            # Hot stream is in 2 phase and condensing
            alpha_h = self._Condensation_Correlation(cell_index)

        return alpha_c, alpha_h
    

    """
    This method iteratively solves the heat exchanger model to find the outlet temperatures and heat transfer rate.
        - Outputs :
            - Tout_c : outlet temperature of the cold stream [K]
            - Tout_h : outlet temperature of the hot stream [K]
            - Q : heat transfer rate [W]
            - epsilon : effectiveness of the heat exchanger [-]
    
    """
    def Solve(self):
        Qmax_ext = self._Qmax_ext()         # STEP 1 : Calculate Qmax based on external pinching
        self._cell_division(Qmax_ext)       # STEP 2 : First cell division based on Qmax_ext
        Qmax_int = self._Qmax_int()         # STEP 3 : Calculate a derated Qmax based on internal pinching

        def iteration(Q):
            self._cell_division(Q)         
            w_total = 0
            self.wVector = np.zeros(self.N)
            for j in range(self.N):
                alpha_c_j, alpha_h_j = self._alpha(j)
                wj = self._cell_analysis(j, alpha_h_j, alpha_c_j)
                self.wVector[j] = wj
                w_total += wj
            
            return 1 - w_total
        
        try :
            self.Q = brentq(iteration, 0, Qmax_int) # STEP 4 : Find the real Q using the iterative Brent method between 0 and Qmax
        except ValueError as error:
            if error.args[0]=='f(a) and f(b) must have different signs':
                # The most probable reason for this error is that the number of plates is too high, meaning that the temperature
                # difference between the two streams at one of the internal pinch points close to 0 -> we set Q = Qmax_int
                iteration(Qmax_int)
                self.Q = Qmax_int
                print(f"\n[/!\ /!\ /!\ WARNING START /!\ /!\ /!\] \n Brent method did not converge.  This is "
                      f"likely due to an oversized exchanger (too many plates: N = {self.Nb_plates}).\n"
                      f"[/!\ /!\ /!\  WARNING END  /!\ /!\ /!\] ")
            else :
                raise error  # Re-raise the exception if it's a different error

        self.hout_c = self.EnthalpyVector_c[-1]
        self.hout_h = self.EnthalpyVector_h[0]
        self.Tout_c = PropsSI('T', 'P', self.pin_c, 'H', self.hout_c, self.fluid_c)
        self.Tout_h = PropsSI('T', 'P', self.pin_h, 'H', self.hout_h, self.fluid_h)
        self.epsilon = self.Q / self._Qmax_ext()

        return self.Tout_c, self.Tout_h, self.hout_c, self.hout_h, self.Q, self.epsilon


    """
    This method returns the normalized enthalpy vectors of both streams for further analysis.
        - Outputs :
            - Normalized_EnthalpyVector_c : normalized enthalpy vector of the cold stream
            - Normalized_EnthalpyVector_h : normalized enthalpy vector of the hot stream
        
        h_normalized = mdot * (h - h_min) / Q

    """
    def _get_Normalized_EnthalpyVectors(self):
        hc_min = self.EnthalpyVector_c[0]
        hh_min = self.EnthalpyVector_h[0]
        self.Normalized_EnthalpyVector_c = self.mdot_c * (self.EnthalpyVector_c - hc_min) / self.Q
        self.Normalized_EnthalpyVector_h = self.mdot_h * (self.EnthalpyVector_h - hh_min) / self.Q

        return self.Normalized_EnthalpyVector_c, self.Normalized_EnthalpyVector_h
    

    """
    This method plots the heat exchange on a T-h normalized diagram.
        - Inputs : With_SatLines (boolean) to indicate if saturation lines should be plotted or not
    
    """
    def _plot(self, With_SatLines = False):
        self._get_Normalized_EnthalpyVectors()
        TemperatureVector_c = np.zeros(len(self.EnthalpyVector_c))
        TemperatureVector_h = np.zeros(len(self.EnthalpyVector_h))
        Tsat_c = PropsSI('T', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        Tsat_h = PropsSI('T', 'P', self.pin_h, 'Q', 0, self.fluid_h)

        for i in range(len(self.EnthalpyVector_c)):
            TemperatureVector_c[i] = PropsSI('T', 'P', self.pin_c, 'H', self.EnthalpyVector_c[i], self.fluid_c)
            TemperatureVector_h[i] = PropsSI('T', 'P', self.pin_h, 'H', self.EnthalpyVector_h[i], self.fluid_h)
        
        plt.figure()
        plt.plot(self.Normalized_EnthalpyVector_c, TemperatureVector_c, marker='o', color="blue", label=self.fluid_c)
        plt.plot(self.Normalized_EnthalpyVector_h, TemperatureVector_h, marker='o', color="red", label=self.fluid_h)
        if With_SatLines:
            plt.axhline(y=Tsat_c, color='blue', linestyle='--')
            plt.axhline(y=Tsat_h, color='red', linestyle='--')
        # plt.legend()
        plt.xlabel(r"$\hat{h}[-]$")
        plt.xlim(0,1)
        plt.ylabel(r"$T[K]$")
        plt.show()


    """
    This method implements the Thonon and Bontemps (2002) correlation to estimate the convective heat transfer 
    coefficient of condensing fluid (ideally R290) inside a plate heat exchanger.
        - Input : cell_index (index of the cell for which we want to calculate the heat transfer coefficient)
        - Output : h (heat transfer coefficient) [W/m2K]
    
    """
    def _Condensation_Correlation(self, cell_index):
        x_start = PropsSI('Q', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index], self.fluid_h)
        x_end = PropsSI('Q', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index+1], self.fluid_h)
        x_m = (x_start + x_end) / 2

        mu_l = PropsSI('V', 'P', self.pin_h, 'Q', 0, self.fluid_h)
        mu_start = PropsSI('V', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index], self.fluid_h)
        mu_end = PropsSI('V', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index+1], self.fluid_h)
        mu_m = (mu_start + mu_end) / 2

        rho_l = PropsSI('D', 'P', self.pin_h, 'Q', 0, self.fluid_h)
        rho_v = PropsSI('D', 'P', self.pin_h, 'Q', 1, self.fluid_h)

        P = 2*(self.w + self.W)
        Dh = 4 * self.Aflow / P

        G = self.mdot_h / self.Aflow
        Geq = G*((1-x_m) + x_m*(rho_l/rho_v)**0.5)

        Pr_l = PropsSI("PRANDTL", 'Q', 0, 'P', self.pin_h, self.fluid_h)
        Pr_g = PropsSI("PRANDTL", 'Q', 1, 'P', self.pin_h, self.fluid_h)
        Pr = (1-x_m)*Pr_l + x_m*Pr_g

        k_l = PropsSI('L', 'P', self.pin_h, 'Q', 0, self.fluid_h)
        Re_eq = Geq * Dh / mu_l
        Re = Dh * G / mu_m
        #if (Re < 100) or (Re > 2000) :
        #    print("Warning: Reynolds number out of range for the Thonon and Bontemps correlation (condensation), Re = {:.2f}".format(Re))
        h_l0 = 0.347 * (k_l / Dh) * Re**0.653 * Pr**0.33
        h = 1564 * h_l0 * Re_eq**(-0.76)
        return h


    """
    This method implements the Khan et al. (2010) correlation to estimate the convective heat transfer
    coefficient of single phase fluid (ideally water) inside a plate heat exchanger.
        - Inputs :
            - cell_index : index of the cell for which we want to calculate the heat transfer coefficient
            - stream : "hot" or "cold" to indicate which stream we are considering
        - Output : h (heat transfer coefficient) [W/m2K]

    """
    def _SinglePhase_Correlation(self, cell_index, stream):

        beta_max = 60           # Maximum chevron angle [degrees]
        correction_factor = 1   # Correction factor for Nu when the fluid is not water but propane

        T_start_h = PropsSI('T', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index], self.fluid_h)
        T_end_h = PropsSI('T', 'P', self.pin_h, 'H', self.EnthalpyVector_h[cell_index+1], self.fluid_h)
        T_start_c = PropsSI('T', 'P', self.pin_c, 'H', self.EnthalpyVector_c[cell_index], self.fluid_c)
        T_end_c = PropsSI('T', 'P', self.pin_c, 'H', self.EnthalpyVector_c[cell_index+1], self.fluid_c)
        T_avg_h = (T_start_h + T_end_h) / 2 ; T_avg_c = (T_start_c + T_end_c) / 2
        Tw = (T_avg_h + T_avg_c) / 2 

        if stream == "hot":
            pressure = self.pin_h
            fluid = self.fluid_h
            T_start = T_start_h
            T_end = T_end_h
            mdot = self.mdot_h
        else :
            pressure = self.pin_c
            fluid = self.fluid_c
            T_start = T_start_c
            T_end = T_end_c
            mdot = self.mdot_c
        
        Tm = (T_start + T_end) / 2
        mu = PropsSI('V', 'P', pressure, 'T', Tm, fluid)
        mu_w = PropsSI('V', 'P', pressure, 'T', Tw, fluid)
        Pr = PropsSI("PRANDTL", 'T', Tm, 'P', pressure, fluid)

        P = 2*(self.w + self.W)
        Dh = 4 * self.Aflow / P
        G = mdot / self.Aflow
        Re = Dh * G / mu

        Nu = (0.0161*self.beta/beta_max+0.1298)*Re**(0.198*self.beta/beta_max+0.6398)*Pr**(0.35)*(mu/mu_w)**(0.14)
        if (fluid != "Water"):
            Nu *= correction_factor
        k = PropsSI('L', 'P', pressure, 'T', Tm, fluid)
        h = Nu * k / Dh

        return h


    """
    This method implements the Almalfi et al. (2015) correlation to estimate the convective heat transfer
    coefficient of evaporating fluid inside a plate heat exchanger.
        - Input : cell_index (index of the cell for which we want to calculate the heat transfer coefficient)
        - Output : h (heat transfer coefficient) [W/m2K]
    
    """
    def _Evaporation_Correlation(self, cell_index):

        beta_max = 70           # Maximum chevron angle [degrees]

        # Calculation of Bd
        rho_l = PropsSI('D', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        rho_v = PropsSI('D', 'P', self.pin_c, 'Q', 1, self.fluid_c)
        rho_m = (rho_l + rho_v) / 2
        P = 2*(self.w + self.W)
        Dh = 4 * self.Aflow / P
        g = 9.81
        x_start = PropsSI('Q', 'P', self.pin_c, 'H', self.EnthalpyVector_c[cell_index], self.fluid_c)
        x_end = PropsSI('Q', 'P', self.pin_c, 'H', self.EnthalpyVector_c[cell_index+1], self.fluid_c)
        x_m = (x_start + x_end) / 2
        sigma = PropsSI('SURFACE_TENSION', 'P', self.pin_c, 'Q', x_m, self.fluid_c)
        Bd = (g * (rho_l - rho_v) * Dh**2) / sigma

        # Calculation of Bo
        Q_cell = self.mdot_c * (self.EnthalpyVector_c[cell_index+1] - self.EnthalpyVector_c[cell_index])
        wj = self.wVector[cell_index]
        if (wj == 0): wj = 1
        q_prime_prime = Q_cell / (wj * self.A)  # Local heat flux [W/m2]
        G = self.mdot_c / self.Aflow
        hlv = PropsSI('H', 'P', self.pin_c, 'Q', 1, self.fluid_c) - PropsSI('H', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        Bo = q_prime_prime / (G * hlv)
        mu_l = PropsSI('V', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        mu_v = PropsSI('V', 'P', self.pin_c, 'Q', 1, self.fluid_c)

        # Calculation of h
        kl = PropsSI('L', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        if (Bd < 4) :
            h = 982*(kl/Dh)*(self.beta/beta_max)**(1.101)*(G*G*Dh/(rho_m*sigma))**(0.315)*(rho_l/rho_v)**(-0.224)*Bo**(0.320)
        else :
            h = 18.485*(kl/Dh)*(self.beta/beta_max)**(0.248)*(x_m*G*Dh/mu_v)**(0.135)*(G*Dh/mu_l)**(0.351)*(rho_l/rho_v)**(0.223)*Bd**(0.235)*Bo**(0.198)

        return h


# Examples of usage
if __name__=='__main__':

    # Example 1 : Evaporator with R290 and water
    state_in_c = State(T=275, p=5.5e5, fluid='R290')
    state_in_h = State(T=290, p=1e5, fluid='Water')
    Evaporator = HEX(state_in_c=state_in_c, state_in_h=state_in_h, mdot=[0.15, 0.4], fluid=['R290', 'Water'], N=10)
    Evaporator.Solve()
    print(Evaporator)
    Evaporator._plot()

    
    # Example 2 : Condenser with R290 and water
    state_in_c = State(T=320, p=1e5, fluid='Water') 
    state_in_h = State(T=350, p=25e5, fluid='R290')
    Condenser = HEX(state_in_c=state_in_c, state_in_h=state_in_h, mdot=[0.3, 0.15], fluid=['Water', 'R290'], N=10)
    Condenser.Solve()
    print(Condenser)
    Condenser._plot()


    

"""  

    Questions to be answered :
        - We assume no wall resistance at the moment (Rcond = t (wall thickness) /k (conductive heat transfer coeff)) -> is this valid?
        - How to handle supercritical operation ?
        - We assume no pressure drop at the moment -> is this valid -> if it has to be accounted for see p.89 of Rémi ?
        - We assume no fouling resistance at the moment
        - Impact of lubricant in the refrigerant is neglected

    Possible refinements :
        - Adding the possibility to increase the number of cells (divide each cell in 2 or more) to increase accuracy


"""