from CoolProp.CoolProp import PropsSI 
import numpy as np
from scipy.optimize import brentq  # Function used for iterative root finding 
import matplotlib.pyplot as plt

class HEX():

    """
    Creates a COUNTER-FLOW heat exchanger object based on the following inputs:
        - Tin: list of inlet temperatures [K] for cold and hot streams
        - pin: list of inlet pressures [Pa] for cold and hot streams
        - mdot: list of mass flow rates [kg/s] for cold and hot streams
        - fluid: list of fluid names for cold and hot streams
        - A: heat exchanger area [m2] (assumed to be equal for both streams (i.e., A_c = A_h))
        - Rcond: thermal resistance of the wall separating the two streams [m²K/W] (default = 0)

    """
    def __init__(self, Tin, pin, mdot, fluid, A, Rcond = 0):
        
        self.Tin_c = Tin[0]
        self.Tin_h = Tin[1]
        self.pin_c = pin[0]
        self.pin_h = pin[1]
        self.mdot_c = mdot[0]
        self.mdot_h = mdot[1]
        self.fluid_c = fluid[0]
        self.fluid_h = fluid[1]
        self.hin_c = PropsSI('H', 'T', self.Tin_c, 'P', self.pin_c, self.fluid_c)
        self.hin_h = PropsSI('H', 'T', self.Tin_h, 'P', self.pin_h, self.fluid_h)
        self.A = A
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
        result = ""
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

        """ To be completed with more information"""

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
    TO BE COMPLETED
    
    """
    def _alpha(self):

        """ TO BE COMPLETED"""

        return 1000
    


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
            for j in range(self.N):
                alpha_h_j = self._alpha()
                alpha_c_j = self._alpha()
                wj = self._cell_analysis(j, alpha_h_j, alpha_c_j)
                w_total += wj
            
            return 1 - w_total
        
        self.Q = brentq(iteration, 0, Qmax_int) # STEP 4 : Find the real Q using the iterative Brent method between 0 and Qmax

        self.hout_c = self.EnthalpyVector_c[-1]
        self.hout_h = self.EnthalpyVector_h[0]
        self.Tout_c = PropsSI('T', 'P', self.pin_c, 'H', self.hout_c, self.fluid_c)
        self.Tout_h = PropsSI('T', 'P', self.pin_h, 'H', self.hout_h, self.fluid_h)
        self.epsilon = self.Q / self._Qmax_ext()

        return self.Tout_c, self.Tout_h, self.Q, self.epsilon

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


# Example of usage
Evaporator_LT = HEX(Tin=[275, 283], pin=[5.5e5, 1e5], mdot=[0.2, 1], fluid=['R290', 'Water'], A=1)
Evaporator_LT.Solve()
print(Evaporator_LT)
Evaporator_LT._plot()



    

"""  

    What remains to be done:
        - Make functions to calculate alpha_h and alpha_c in each cell -> correlations are needed depending on the fluids and flow regimes


    Notes :
        - The functions wimposed are not needed in our case

    Questions to be answered :
        - We assume no wall resistance at the moment -> is this valid?
        - How to handle supercritical operation ?
        - We assume no pressure drop at the moment -> is this valid?


"""