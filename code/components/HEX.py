
from CoolProp.CoolProp import PropsSI 
import numpy as np

class HEX():

    """
    Creates a heat exchanger object based on the following inputs:
        - Tin: list of inlet temperatures [K] for cold and hot streams
        - pin: list of inlet pressures [Pa] for cold and hot streams
        - mdot: list of mass flow rates [kg/s] for cold and hot streams
        - fluid: list of fluid names for cold and hot streams

    """
    def __init__(self, Tin, pin, mdot, fluid):
        
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
        self.condensation_start = None  # The hot stream is going from vapor to 2 phase
        self.condensation_end = None    # The hot stream is going from 2 phase to liquid
        self.evaporation_start = None   # The cold stream is going from liquid to 2 phase
        self.evaporation_end = None     # The cold stream is going from 2 phase to vapor

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
    def _cell_division(self, Qmax_ext):

        EnthalpyVector_h = np.array([self.hin_h - Qmax_ext/self.mdot_h, self.hin_h])
        EnthalpyVector_c = np.array([self.hin_c, self.hin_c + Qmax_ext/self.mdot_c])

        h_h_dew = PropsSI('H', 'P', self.pin_h, 'Q', 0, self.fluid_h)
        h_h_bub = PropsSI('H', 'P', self.pin_h, 'Q', 1, self.fluid_h)
        h_c_dew = PropsSI('H', 'P', self.pin_c, 'Q', 0, self.fluid_c)
        h_c_bub = PropsSI('H', 'P', self.pin_c, 'Q', 1, self.fluid_c)
        N = 1 # Initial number of cells
        

        ## A. Insert phase transition enthalpies for the hot stream if applicable

            # 1. Check for potential phase transition 2 phase to Liquid (bubble point) 
        if (EnthalpyVector_h[0] < h_h_bub) and (EnthalpyVector_h[-1] > h_h_bub):
            EnthalpyVector_h = np.append(EnthalpyVector_h, h_h_bub)
            self.condensation_end = True
            N += 1
        else :
            self.condensation_end = False

            # 2. Check for potential phase transition Vapor to 2 phase (dew point)
        if (EnthalpyVector_h[0] < h_h_dew) and (EnthalpyVector_h[-1] > h_h_dew):
            EnthalpyVector_h = np.append(EnthalpyVector_h, 1, h_h_dew)
            self.condensation_start = True
            N += 1
        else :
            self.condensation_start = False

        EnthalpyVector_h.sort()
        
        ## B. Insert phase transition enthalpies for the cold stream if applicable

            # 1. Check for potential phase transition Liquid to 2 phase (bubble point)
        if (EnthalpyVector_c[0] < h_c_bub) and (EnthalpyVector_c[-1] > h_c_bub):
            EnthalpyVector_c = np.append(EnthalpyVector_c, h_c_bub)
            self.evaporation_start = True
            N += 1
        else :
            self.evaporation_start = False

            # 2. Check for potential phase transition 2 phase to Vapor (dew point)
        if (EnthalpyVector_c[0] < h_c_dew) and (EnthalpyVector_c[-1] > h_c_dew):
            EnthalpyVector_c = np.append(EnthalpyVector_c, h_c_dew)
            self.evaporation_end = True
            N += 1
        else :
            self.evaporation_end = False
        
        EnthalpyVector_c.sort()


        ## C. Insert complementary phase transition enthalpies

        for j in range(N):
            Qcell_h = self.mdot_h * (EnthalpyVector_h[j+1] - EnthalpyVector_h[j])
            Qcell_c = self.mdot_c * (EnthalpyVector_c[j+1] - EnthalpyVector_c[j])
            
            # We insert the complementary enthalpy in the enthalpy vector of the stream with the 
            # highest heat transfer in the cell (indicating that we need to split this cell)

            if Qcell_h > Qcell_c:   # In this case, h_h[j+1] is unknown -> we make an energy balance to find it
                new_h = self.mdot_c/self.mdot_h * (EnthalpyVector_c[j+1] - EnthalpyVector_c[j]) + EnthalpyVector_h[j]
                EnthalpyVector_h = np.append(EnthalpyVector_h, new_h)
                EnthalpyVector_h.sort()

            else :                  # In this case, h_c[j+1] is unknown -> we make an energy balance to find it
                new_h = self.mdot_h/self.mdot_c * (EnthalpyVector_h[j+1] - EnthalpyVector_h[j]) + EnthalpyVector_c[j]
                EnthalpyVector_c = np.append(EnthalpyVector_c, new_h)
                EnthalpyVector_c.sort()

        ## D. Verify that both vectors have the same length

        if (len(EnthalpyVector_c) != len(EnthalpyVector_h)):
            raise ValueError("Cell division algorithm failed: Enthalpy vectors have different lengths.")
        
        if (len(EnthalpyVector_h) != (N+1)):
            raise ValueError("Size of enthalpy vectors does not match the expected number of cells.")

        return EnthalpyVector_c, EnthalpyVector_h
    
    """
    This method calculates the maximum heat transfer rate based on the assumption of internal pinching.
        - Input : Enthalpy Vectors of the cold and hot streams
        - Output : Maximum heat transfer rate based on internal pinching
    
    """
    def _Qmax_int(self, EnthalpyVector_c, EnthalpyVector_h):
        Qmax_int_c = 0 ; Qmax_int_h = 0
        for i in range(len(EnthalpyVector_c)-1):
            Qmax_int_c += self.mdot_c * (EnthalpyVector_c[i+1] - EnthalpyVector_c[i])
            Qmax_int_h += self.mdot_h * (EnthalpyVector_h[i+1] - EnthalpyVector_h[i])
        
        if abs(Qmax_int_c - Qmax_int_h) > 1e-6:
            raise ValueError("Energy balance not satisfied in internal pinching calculation.")
        
        self.Qmax_int = Qmax_int_c

        return self.Qmax_int
    



"""  

    What remains to be done:
        - Derate corretly the Qmax using the temperature pinch -> Qmax_int is not correct because it is not derated
        - Find the real Q dot using the Brent method between 0 and Qmax_int


    Notes :
        - The functions wimposed are not needed in our case


"""