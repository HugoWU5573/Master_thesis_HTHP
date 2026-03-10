import CoolProp
import numpy as np
import matplotlib.pyplot as plt
import os

if not __name__ == '__main__':
    from components.state import State


"""  
MAIN ASSUMPTIONS :
    - Both streams are in counter-flow configuration
    - No fouling resistance
    - No wall resistance (Rcond = t (wall thickness) /k (conductive heat transfer coeff))
    - Impact of lubricant in the refrigerant is neglected
    - The pressure drop is only calculated in the design phase of the heat exchanger.  It is not considered in the 
      determination of the temperature profiles of both streams. 

"""    

class HEX_Design():

    """
    Creates a counter-flow heat exchanger object that will be modelled using the pinch model for design purposes.
    The parameters are:
        - states_in : list of State objects for the inlet states of both streams [State_c_in, State_h_in]
        - states_out : list of State objects for the outlet states of both streams [State_c_out, State_h_out]
        - mdot : list of mass flow rates for both streams [mdot_c, mdot_h] [kg/s] (see note below)
        - name : name of the heat exchanger (optional)
        - L : Length of the heat exchanger plates [m] (optional, default = 0.3 m)
        - W : Width of the heat exchanger plates [m] (optional, default = 0.1 m)
        - w : Distance between two plates (channel gap) [m] (optional, default = 2e-3 m)
        - phi : Surface enlargement factor due to the corrugation of the plates (optional, default = 1.2)
        - beta : Chevron angle of the plates [degrees] (optional, default = 45 degrees)
        - gamma : Aspect ratio of the corrugation (optional, default = 0.55)
        - Rcond : thermal resistance of the heat exchanger wall [m²K/W] (optional, default = 0)
        - mode : "Dimensional" or "Non-Dimensional" (optional, default = "Dimensional")
        - type : "Recuperator" or None (optional, default = None)  (see note below)
        - epsilon : Effectiveness of the heat exchanger (only for type = "Recuperator") (optional, default = None)
        - model : None or "ACH65" or "ACK18" or "ACK16" or "ACH30EQ" (optional, default = None) to indicate a specific model of Alfa Laval plate heat exchanger
        - corr_factors : dictionary of correction factors for the heat transfer coefficients and friction factors in the different flow regimes (optional, default = None, only used in the sensitivity analysis)
    
    Notes : 
        - In the "Dimensional" mode, at least one of the two mass flow rates must be provided (the other can be None).
        - In the "Non-Dimensional" mode, both mass flow rates must be set to None (default setting)
        - In the "Recuperator" type, the heat exchanger is modelled using a constant effectiveness (epsilon) approach.
        - In the "Recuperator" type, the mass flow rates and inlet states must be provided to calculate the outlet states.
    
    """
    def __init__(self, states_in, states_out, mdot=[None, None], name ="HEX", L=0.3, W=0.1, w=2e-3, 
                 beta=45, phi=1.2, gamma=0.55, Rcond=0, mode="Dimensional", type=None, epsilon=None, model=None, corr_factors=None):

        if model == "ACH65":
            L = 0.535
            W = 0.120
            w = 0.72e-3
            min_nb_plates = 10
            max_nb_plates = 120
        elif model == "ACK18":
            L = 0.315
            W = 0.073
            w = 1.63e-3
            min_nb_plates = 4
            max_nb_plates = 52
        elif model == "ACK16":
            L = 0.2095
            W = 0.0735
            w = 1.57e-3
            min_nb_plates = 4
            max_nb_plates = 60
        elif model == "ACH30EQ" :
            L = 0.323
            W = 0.095
            w = 0.91e-3
            min_nb_plates = 4
            max_nb_plates = 120
        elif model == "ACP70X":
            L = 0.526 
            W = 0.111
            w = 1.63e-3
            min_nb_plates = 4
            max_nb_plates = 124
        else :
            min_nb_plates = 3
            max_nb_plates = 1000

        self.state_in_c = states_in[0]
        self.state_in_h = states_in[1]
        self.state_out_c = states_out[0]
        self.state_out_h = states_out[1]
        self.mdot_c = mdot[0]
        self.mdot_h = mdot[1]
        self.Q = None
        self.HEOS_cold = self.state_in_c.heos 
        self.HEOS_hot = self.state_in_h.heos
        self.Tpinch = None
        self.name = name
        self.supercritical_hot_stream = False
        self.Rcond = Rcond
        self.L = L
        self.W = W
        self.w = w
        self.phi = phi
        self.Aflow = self.W * self.w                  # Cross-sectional flow area for one channel [m2]
        self.P = 2*(self.w + self.W)                  # Wetted perimeter for one channel [m]
        self.Dh = 4 * self.Aflow / self.P / self.phi  # Hydraulic diameter for one channel, corrected by the surface enlargement factor [m]
        self.beta = beta
        self.gamma = gamma
        self.A = None
        self.mode = mode
        self.type = type
        self.epsilon = epsilon
        self.Nb_plates = None           # Number of plates in the heat exchanger (to be computed in the Compute_Area method)
        self.min_Nb_plates = min_nb_plates
        self.max_Nb_plates = max_nb_plates
        self.model = model
        self.pressure_drop_cold = None
        self.pressure_drop_hot = None
        
        if corr_factors is None:
            self.corr_factors = {
                "h_single_phase": 1.0,
                "h_evaporation": 1.0,
                "h_condensation": 1.0,
                "h_supercritical": 1.0,
                "f_single_phase": 1.0,
                "f_evaporation": 1.0,
                "f_condensation": 1.0,
                "f_supercritical": 1.0,
            }
        else :
            self.corr_factors = corr_factors


    """
    This method returns a string representation of the HEX_Design object.
    
    """    
    def __str__(self):
        result = "\n=================================== {:<12} ===================================\n".format(self.name)

        # Streams summary table with all information
        stream_header = "+--------+-----------+----------+----------+------------+------------+-------------+"
        stream_table = [
            stream_header,
            "|Stream  |Fluid      |Tin [K]   |Tout [K]  |pin [bar]   |pout [bar]  |mdot [kg/s]  |",
            stream_header,
            "| Hot    | {:<9} | {:8.2f} | {:8.2f} | {:10.2f} | {:10.2f} | {:11.3f} |".format(
            self.state_in_h.fluid,
            self.state_in_h.T, self.state_out_h.T,
            self.state_in_h.p / 1e5, self.state_out_h.p / 1e5,
            self.mdot_h
            ),
            "| Cold   | {:<9} | {:8.2f} | {:8.2f} | {:10.2f} | {:10.2f} | {:11.3f} |".format(
            self.state_in_c.fluid,
            self.state_in_c.T, self.state_out_c.T,
            self.state_in_c.p / 1e5, self.state_out_c.p / 1e5,
            self.mdot_c
            ),
            stream_header,
        ]
        result += "\n" + "\n".join(stream_table) + "\n"
        # Summary of heat duty and pinch
        result += f"\nHeat transfer rate: {self.Q/1000:.2f} kW,th\n"
        result += f"Delta_T at pinch point: {self.Tpinch:.2f} K\n"
        if self.A == None:
            result += "Heat exchanger area: Not calculated yet\n"
        elif self.model == None:
            result += f"Heat exchanger area: {self.A:.3f} m2 ({self.Nb_plates} plates)\n"
        else :
            result += f"Heat exchanger area: {self.A:.3f} m2 ({self.Nb_plates} plates - model: {self.model})\n"
        if self.pressure_drop_hot is not None and self.pressure_drop_cold is not None:
            result += f"Pressure drops :\n"
            result += f"\t - Cold stream: {self.pressure_drop_cold/1e5:.2f} bar  ({self.pressure_drop_cold*100/self.state_in_c.p:.1f} %)\n"
            result += f"\t - Hot stream: {self.pressure_drop_hot/1e5:.2f} bar  ({self.pressure_drop_hot*100/self.state_in_h.p:.1f} %)\n"
        result += "===================================================================================="

        return result
    

    """
    This method determines the unknown mass flow rate(s) based on an energy balance.
        - Output : Q (heat transfer rate) [W]

    """
    def _determine_unknown_mass_flow(self):

        if self.mode == "Non-Dimensional":
            self.Q = 1 # In non-dimensional mode, we set Q = 1 W as a reference value
            Delta_h_c = self.state_out_c.h - self.state_in_c.h
            Delta_h_h = self.state_in_h.h - self.state_out_h.h
            self.mdot_c = self.Q / Delta_h_c
            self.mdot_h = self.Q / Delta_h_h
        
        elif self.mode == "Dimensional" :

            if self.mdot_c == None and self.mdot_h == None :
                raise ValueError("In 'Dimensional' mode, at least one mass flow rate must be provided.")

            elif self.mdot_c == None :    # Mass flow rate of the cold stream is unknown
                Qh = self.mdot_h * (self.state_in_h.h - self.state_out_h.h)
                self.Q = Qh 
                Delta_h_c = self.state_out_c.h - self.state_in_c.h
                self.mdot_c = Qh / Delta_h_c

            elif self.mdot_h == None :    # Mass flow rate of the hot stream is unknown
                Qc = self.mdot_c * (self.state_out_c.h - self.state_in_c.h)
                self.Q = Qc
                Delta_h_h = self.state_in_h.h - self.state_out_h.h
                self.mdot_h = Qc / Delta_h_h

            else :                        # Both mass flow rates are known
                Qc = self.mdot_c * (self.state_out_c.h - self.state_in_c.h)
                Qh = self.mdot_h * (self.state_in_h.h - self.state_out_h.h)
                if not np.isclose(Qc, Qh, atol=1e-3):
                    raise ValueError("Heat duty of both streams are not consistent.")
                else :
                    self.Q = (Qc + Qh) / 2
        
        else :
            raise ValueError("Invalid mode. Choose either 'Dimensional' or 'Non-Dimensional'.")

        return self.Q
    

    """
    This method determines the unknow outlet states of the heat exchanger based on its effectiveness
    using an iterative approach to accurately estimate cp_min.
        - Outputs :
            - state_out_c : outlet State of the cold stream
            - state_out_h : outlet State of the hot stream

    """
    def _determine_unknown_outlet_states(self):

        if self.mode == "Non-Dimensional":

            Delta_T_in = self.state_in_h.T - self.state_in_c.T

            # First estimate of cp_min based on the inlet states
            self.HEOS_cold.update(CoolProp.PT_INPUTS, self.state_in_c.p, self.state_in_c.T)
            cp_c = self.HEOS_cold.cpmass()
            self.HEOS_hot.update(CoolProp.PT_INPUTS, self.state_in_h.p, self.state_in_h.T)
            cp_h = self.HEOS_hot.cpmass()
            cp_min = min(cp_c, cp_h)

            # First estimate of state_out_c and state_out_h based on epsilon
            hout_h = self.state_in_h.h - self.epsilon * cp_min * Delta_T_in
            self.state_out_h = State(self.HEOS_hot,h=hout_h, p=self.state_in_h.p)
            hout_c = self.state_in_c.h + self.epsilon * cp_min * Delta_T_in
            self.state_out_c = State(self.HEOS_cold,h=hout_c, p=self.state_in_c.p)

            cp_min_old = 0.0
            cp_min_new = cp_min

            while not np.isclose(cp_min_old, cp_min_new, atol=1e-3):
                cp_min_old = cp_min_new

                # Compute cp_c and cp_h by averaging between inlet and outlet temperatures
                cp_c = self._Cp_average(self.HEOS_cold, self.state_in_c.p, self.state_in_c.T, self.state_out_c.T)
                cp_h = self._Cp_average(self.HEOS_hot, self.state_in_h.p, self.state_in_h.T, self.state_out_h.T)
                cp_min_new = min(cp_c, cp_h)

                # Update outlet States based on new cp_min
                hout_h = self.state_in_h.h - self.epsilon * cp_min_new * Delta_T_in
                self.state_out_h = State(self.HEOS_hot,h=hout_h, p=self.state_in_h.p)
                hout_c = self.state_in_c.h + self.epsilon * cp_min_new * Delta_T_in
                self.state_out_c = State(self.HEOS_cold,h=hout_c, p=self.state_in_c.p)

            self.Q = 1 # In non-dimensional mode, we set Q = 1 W as a reference value
        
        elif self.mode == "Dimensional" :
            # In the "Dimensional" mode, the outlet states are already known -> we only compute Q

            Q_h = self.mdot_h * (self.state_in_h.h - self.state_out_h.h)
            Q_c = self.mdot_c * (self.state_out_c.h - self.state_in_c.h)
            if not np.isclose(Q_h, Q_c, atol=1e-3):
                raise ValueError("Heat duty of both streams are not consistent after iteration.")
            else :
                self.Q = (Q_h + Q_c) / 2

        return self.state_out_c, self.state_out_h


    """
    This method divides the heat exchanger into cells based on potential phase changes in both streams.
        - Input : extra_cells (boolean to indicate if we want to add extra cells for better accuracy, especially useful for supercritical operation)
        - Output : EnthalpyVector_c, EnthalpyVector_h (enthalpy vectors of both streams)
    
    """
    def _cell_division(self, extra_cells):

        self.EnthalpyVector_h = np.array([self.state_out_h.h, self.state_in_h.h])
        self.EnthalpyVector_c = np.array([self.state_in_c.h, self.state_out_c.h])

        self.N = 1 # Initial number of cells

        ## A. Insert phase transition enthalpies for the hot stream if applicable

        pcrit_h = self.HEOS_hot.p_critical()

        if not (self.state_in_h.p > pcrit_h):  # If the hot stream pressure is higher than the critical pressure, no phase change can occur

            self.HEOS_hot.update(CoolProp.PQ_INPUTS, self.state_in_h.p, 0)
            self.h_h_bub = self.HEOS_hot.hmass()
            self.HEOS_hot.update(CoolProp.PQ_INPUTS, self.state_in_h.p, 1)
            self.h_h_dew = self.HEOS_hot.hmass()

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

        pcrit_c = self.HEOS_cold.p_critical()

        if not (self.state_in_c.p > pcrit_c):  # If the cold stream pressure is higher than the critical pressure, no phase change can occur

            self.HEOS_cold.update(CoolProp.PQ_INPUTS, self.state_in_c.p, 0)
            self.h_c_bub = self.HEOS_cold.hmass()
            self.HEOS_cold.update(CoolProp.PQ_INPUTS, self.state_in_c.p, 1)
            self.h_c_dew = self.HEOS_cold.hmass()

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


        ## D. Add extra cells if required (e.g. for supercritical operation)
        if extra_cells:

            final_nb_cells = 50
            total_delta_h = self.EnthalpyVector_c[-1] - self.EnthalpyVector_c[0]

            # Ensure that each cell will correspond to a similar enthalpy difference
            nb_cells_per_cell = np.zeros(self.N)
            for j in range(self.N):
                delta_h_cell_j = self.EnthalpyVector_c[j+1] - self.EnthalpyVector_c[j]
                proportion_j = delta_h_cell_j / total_delta_h
                nb_cells_per_cell[j] = max(1, round(proportion_j * final_nb_cells))
            
            # Create new enthalpy vectors with the extra cells
            EnthalpyVector_c_new = []
            EnthalpyVector_h_new = []
            for j in range(self.N):
                h_c_start = self.EnthalpyVector_c[j]
                h_c_end = self.EnthalpyVector_c[j+1]
                h_h_start = self.EnthalpyVector_h[j]
                h_h_end = self.EnthalpyVector_h[j+1]

                n_cells_j = int(nb_cells_per_cell[j])
                for k in range(n_cells_j):
                    h_c_new = h_c_start + k * (h_c_end - h_c_start) / n_cells_j
                    h_h_new = h_h_start + k * (h_h_end - h_h_start) / n_cells_j
                    EnthalpyVector_c_new.append(h_c_new)
                    EnthalpyVector_h_new.append(h_h_new)

            EnthalpyVector_c_new.append(self.EnthalpyVector_c[-1])
            EnthalpyVector_h_new.append(self.EnthalpyVector_h[-1])

            self.EnthalpyVector_c = np.array(EnthalpyVector_c_new)
            self.EnthalpyVector_h = np.array(EnthalpyVector_h_new)
            self.N = len(self.EnthalpyVector_c) - 1


        ## E. Verify that both vectors have the same length

        if (len(self.EnthalpyVector_c) != len(self.EnthalpyVector_h)):
            raise ValueError(f"Cell division algorithm failed: Enthalpy vectors have different lengths, len(EnthalpyVector_c) = {len(self.EnthalpyVector_c)}, len(EnthalpyVector_h) = {len(self.EnthalpyVector_h)}.")
        
        if (len(self.EnthalpyVector_h) != (self.N+1)):
            raise ValueError("Size of enthalpy vectors does not match the expected number of cells.")

        return self.EnthalpyVector_c, self.EnthalpyVector_h


    """
    This method computes the temperature difference at the pinch point of the heat exchanger.
        - Output : Tpinch (temperature difference at the pinch point) [K]
    
    """
    def Compute_Pinch(self):
        if self.type is None:
            self._determine_unknown_mass_flow()

        # Check if the hot stream is supercritical or not
        Tcrit_h = self.HEOS_hot.T_critical()
        pcrit_h = self.HEOS_hot.p_critical()
        if (self.state_in_h.p > pcrit_h) and (self.state_in_h.T > Tcrit_h):
            self.supercritical_hot_stream = True
            
        self._cell_division(extra_cells = self.supercritical_hot_stream) 

        self.TemperatureVector_c = np.zeros(len(self.EnthalpyVector_c))
        self.TemperatureVector_h = np.zeros(len(self.EnthalpyVector_h))

        Tpinch = np.inf
        for i in range(len(self.EnthalpyVector_c)):
            self.HEOS_cold.update(CoolProp.HmassP_INPUTS, self.EnthalpyVector_c[i], self.state_in_c.p)
            Tc = self.HEOS_cold.T()
            self.TemperatureVector_c[i] = Tc
            self.HEOS_hot.update(CoolProp.HmassP_INPUTS, self.EnthalpyVector_h[i], self.state_in_h.p)
            Th = self.HEOS_hot.T()
            self.TemperatureVector_h[i] = Th
            deltaT = Th - Tc
            # if deltaT < 0:
            #    raise ValueError("Heat exchanger model error: temperature difference between hot and cold streams is negative at some point.")
            Tpinch = min(Tpinch, deltaT)

        self.Tpinch = Tpinch

        return self.Tpinch
    

    """
    This method computes the outlet states of both streams for a Recuperator type heat exchanger using
    a constant effectiveness model.
        - Outputs :
            - state_out_c : State object for the outlet state of the cold stream
            - state_out_h : State object for the outlet state of the hot stream

    """
    def Solve_Recuperator(self):

        if self.type != "Recuperator":
            raise ValueError("This method is only applicable for recuperator type heat exchangers.")
        if self.epsilon is None:
            raise ValueError("Effectiveness (epsilon) must be provided for recuperator type heat exchangers.")

        self._determine_unknown_outlet_states()
        self.Compute_Pinch()    # We call Compute_Pinch to set up the enthalpy and temperature vectors (the actual pinch is not relevant here)

        return self.state_out_c, self.state_out_h


    """
    This method analyses a single cell of the heat exchanger and returns the required area for that cell.
        - Inputs :
            - cell_index : index of the cell to be analysed
            - alpha_h_j : convective heat transfer coefficient of the hot stream in cell j [W/m²K]
            - alpha_c_j : convective heat transfer coefficient of the cold stream in cell j [W/m²K]
        - Output :
            - Aj : required area for cell j [m²]
            - q_prime_prime_j : heat flux in cell j [W/m²]
    """
    def _cell_analysis(self, cell_index, alpha_h_j, alpha_c_j):

        # Heat transfer rate in cell "cell_index"
        Qj_h = self.mdot_h * (self.EnthalpyVector_h[cell_index+1] - self.EnthalpyVector_h[cell_index])
        Qj_c = self.mdot_c * (self.EnthalpyVector_c[cell_index+1] - self.EnthalpyVector_c[cell_index])
        if not (np.isclose(Qj_h, Qj_c, atol=1e-6)):
            raise ValueError("Heat transfer rates in cell j are not consistent.")
        else :
            Qj = (Qj_h + Qj_c)/2

        Th_j = self.TemperatureVector_h[cell_index]
        Th_j_plus_1 = self.TemperatureVector_h[cell_index+1]
        Tc_j = self.TemperatureVector_c[cell_index]
        Tc_j_plus_1 = self.TemperatureVector_c[cell_index+1]

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
        
        # Area for cell j
        UAj = Qj / LMTDj
        Uj = 1 / (1/alpha_h_j + self.Rcond + 1/alpha_c_j)  # Overall heat transfer coefficient for cell j
        Aj = UAj / Uj

        q_prime_prime_j = Qj / Aj  # Heat flux in cell j [W/m²]

        return Aj, q_prime_prime_j


    """
    This method evaluates the heat transfer area (geometric and required) and the pressure drops for both 
    streams for a given number of plates in the heat exchanger.
        - Input : nb_plates (number of plates in the heat exchanger)
        - Outputs :
            - A_required : minimum heat transfer area required to achieve the desired heat transfer with the given number of plates [m²]
            - A_geom : geometric heat transfer area available with the given number of plates [m²]
            - pressure_drop_cold : total pressure drop for the cold stream with the given number of plates [Pa]
            - pressure_drop_hot : total pressure drop for the hot stream with the given number of plates [Pa]

    """
    def _evaluate_design(self, nb_plates):

        # STEP 1 : Minimum required heat exchange area calculation
        Arequired_list = np.zeros(self.N)
        f_c_list = np.zeros(self.N)
        f_h_list = np.zeros(self.N)

        q_prime_prime_per_cell_old = self.q_prime_prime_per_cell.copy()
        error = np.inf

        while error > 1e-3:

            # Compute area & q'' per cell
            for cell_index in range(self.N):

                alpha_c, alpha_h, f_c, f_h = self._alpha_f(cell_index, nb_plates)
                Arequired_cell, q_prime_prime_j = self._cell_analysis(cell_index, alpha_h, alpha_c)
                Arequired_list[cell_index] = Arequired_cell
                self.q_prime_prime_per_cell[cell_index] = q_prime_prime_j
                f_c_list[cell_index] = f_c
                f_h_list[cell_index] = f_h

            # Check for convergence of q'' per cell
            relative_difference_q_prime_prime = np.abs(self.q_prime_prime_per_cell - q_prime_prime_per_cell_old) / self.q_prime_prime_per_cell
            error = np.max(relative_difference_q_prime_prime)
            q_prime_prime_per_cell_old = self.q_prime_prime_per_cell.copy()

        A_required = np.sum(Arequired_list)

        # STEP 2 : Total pressure drops for both streams calculation
        delta_p_c_list = np.zeros(self.N)
        delta_p_h_list = np.zeros(self.N)

        nb_channels = nb_plates - 1
        nb_channels_h = nb_channels // 2
        nb_channels_c = nb_channels - nb_channels_h

        for cell_index in range(self.N):

            f_c = f_c_list[cell_index]
            f_h = f_h_list[cell_index]

            L_real_cell =  Arequired_list[cell_index] / A_required * self.L
                
                # Cold stream characteristics
            h_mean_c = (self.EnthalpyVector_c[cell_index] + self.EnthalpyVector_c[cell_index+1]) / 2
            self.HEOS_cold.update(CoolProp.HmassP_INPUTS, h_mean_c, self.state_in_c.p)
            rho_c = self.HEOS_cold.rhomass()
            v_c = self.mdot_c / (rho_c * self.Aflow * nb_channels_c)  # Velocity in each channel of the cold stream

                # Hot stream characteristics
            h_mean_h = (self.EnthalpyVector_h[cell_index] + self.EnthalpyVector_h[cell_index+1]) / 2
            self.HEOS_hot.update(CoolProp.HmassP_INPUTS, h_mean_h, self.state_in_h.p)
            rho_h = self.HEOS_hot.rhomass()
            v_h = self.mdot_h / (rho_h * self.Aflow * nb_channels_h)  # Velocity in each channel of the hot stream

            delta_p_c_list[cell_index] = f_c * (L_real_cell/self.Dh) * (rho_c * v_c**2 / 2)
            delta_p_h_list[cell_index] = f_h * (L_real_cell/self.Dh) * (rho_h * v_h**2 / 2)

        pressure_drop_cold = np.sum(delta_p_c_list)
        pressure_drop_hot = np.sum(delta_p_h_list)
        A_geom = (nb_plates - 2) * self.W * self.L * self.phi

        return A_required, A_geom, pressure_drop_cold, pressure_drop_hot

    
    """
    This method computes the total heat exchanger area and number of plates required to achieve the desired 
    heat transfer rate while respecting the constraints on the maximum pressure drops for both streams.
        - Inputs :
            - plot : boolean to indicate if we want to plot the convergence process towards the solution (optional, default is False)
            - save : boolean to indicate if we want to save the plot (optional, default is False)
            - name_cycle : name of the cycle in which the HEX is integrated, used to save the plot in the correct folder (optional, default is None)
            - delta_P_hot_max : maximum allowed pressure drop for the hot stream (optional, default is None, meaning 10% of the inlet pressure)
            - delta_P_cold_max : maximum allowed pressure drop for the cold stream (optional, default is None, meaning 10% of the inlet pressure)
        - Outputs : 
            - A (total heat exchanger area) [m²], 
            - Nb_plates (number of plates required in the heat exchanger)

    """
    def Compute_Area(self, plot = False, save = False, name_cycle=None, delta_P_hot_max = None, delta_P_cold_max = None):

        # Default maximum pressure drop limits
        if delta_P_hot_max is None:
            delta_P_hot_max = np.min([0.5e5, 0.1 * self.state_in_h.p])   # Maximum pressure drop is 10% of the inlet pressure by default (with a maximum set to 0.5 bar)
        if delta_P_cold_max is None:
            delta_P_cold_max = np.min([0.5e5, 0.1 * self.state_in_c.p])

        # We initialize the heat flux per cell with a first guess based on the minimum number of plates
        A_min = (self.min_Nb_plates - 2) * self.W * self.L * self.phi
        q_prime_prime_first_guess = self.Q / (self.N * A_min)
        self.q_prime_prime_per_cell = np.ones(self.N) * q_prime_prime_first_guess

        # Preparation of some lists to store the convergence process if plot = True
        if plot:
            N_list = []
            Areq_list = []
            Ageom_list = []
            dPh_list = []
            dPc_list = []

        feasible_design_found = False

        # We loop over the number of plates until we find a feasible design that respects both thermal and hydraulic constraints
        for Nplates in range(self.min_Nb_plates, self.max_Nb_plates +1, 2):
            
            # Evaluate design with current number of plates
            A_req, A_geom, dP_c, dP_h = self._evaluate_design(Nplates)

            if plot:
                N_list.append(Nplates)
                Areq_list.append(A_req)
                Ageom_list.append(A_geom)
                dPh_list.append(dP_h)
                dPc_list.append(dP_c)

            # Check thermal and hydraulic feasibility
            thermal_ok = A_req <= A_geom
            hydraulic_ok = (dP_h <= delta_P_hot_max) and (dP_c <= delta_P_cold_max)

            if thermal_ok and hydraulic_ok and not feasible_design_found:
                # Feasible design found for the first time, we store the results
                feasible_design_found = True
                self.Nb_plates = Nplates
                self.pressure_drop_hot = dP_h
                self.pressure_drop_cold = dP_c

                # We evaluate once again the design with two plates less to determine a continuous area value (useful for the PCE used for sensitivity analysis)

                if Nplates - 2 < self.min_Nb_plates:
                    self.A = A_geom
                else :
                    A_req_previous, A_geom_previous, dP_c_previous, dP_h_previous = self._evaluate_design(Nplates - 2)
                    c1_prevous = A_req_previous/A_geom_previous ; c1_current = A_req/A_geom
                    c2_previous = dP_h_previous/delta_P_hot_max ; c2_current = dP_h/delta_P_hot_max
                    c3_previous = dP_c_previous/delta_P_cold_max ; c3_current = dP_c/delta_P_cold_max
                    
                    g_previous = max(c1_prevous, c2_previous, c3_previous)
                    g_current = max(c1_current, c2_current, c3_current)
                    t = (g_previous - 1) / (g_previous - g_current)
                    t = max(0, min(1, t))  # Ensure that t is between 0 and 1

                    self.A = (1-t) * A_geom_previous + t * A_geom

                if not plot:
                    # Do not break the loop immediately if plotting is enable to see the full curves
                    return self.A, self.Nb_plates

        # Plot results if requested
        if plot:
            self._plot_design_results(N_list, Areq_list, Ageom_list, dPh_list, dPc_list, delta_P_hot_max, delta_P_cold_max, save=save, name_cycle=name_cycle)
        
        if feasible_design_found:
            return self.A, self.Nb_plates
        else :
            raise ValueError(f"No feasible number of plates found within bounds for the {self.name} with the following model {self.model}.")



    """
    This method calculates the convective heat transfer coefficients and friction factors for both streams in cell "cell_index".
        - Input : 
            - cell_index (index of the cell for which we want to calculate the heat transfer coefficients)
            - nb_plates (number of plates in the heat exchanger)
        - Outputs :
            - alpha_c : convective heat transfer coefficient for the cold stream [W/m2K]
            - alpha_h : convective heat transfer coefficient for the hot stream [W/m2K]
            - f_c : Darcy friction factor for the cold stream [-]
            - f_h : Darcy friction factor for the hot stream [-]
    
    """
    def _alpha_f(self, cell_index, nb_plates):

        alpha_c = None
        alpha_h = None
        f_c = None
        f_h = None

        ## Cold stream
        hc_start = self.EnthalpyVector_c[cell_index]
        hc_end = self.EnthalpyVector_c[cell_index+1]

        if self.state_in_c.p > self.HEOS_cold.p_critical():
            # Cold stream is "supercritical"
            alpha_c, f_c = self._Supercritical_Correlation(cell_index, "cold", nb_plates)
        elif hc_end <= self.h_c_bub or hc_start >= self.h_c_dew: 
            # Cold stream is single phase
            alpha_c, f_c = self._SinglePhase_Correlation(cell_index, "cold", nb_plates)
        else :
            # Cold stream is in 2 phase and evaporating
            alpha_c, f_c = self._Evaporation_Correlation(cell_index, nb_plates)

        ## Hot stream
        hh_start = self.EnthalpyVector_h[cell_index]
        hh_end = self.EnthalpyVector_h[cell_index+1]

        if self.state_in_h.p > self.HEOS_hot.p_critical():
            # Hot stream is "supercritical"
            alpha_h, f_h = self._Supercritical_Correlation(cell_index, "hot", nb_plates)
        elif hh_end <= self.h_h_bub or hh_start >= self.h_h_dew:
            # Hot stream is single phase liquid
            alpha_h, f_h = self._SinglePhase_Correlation(cell_index, "hot", nb_plates)
        else :
            # Hot stream is in 2 phase and condensing
            alpha_h, f_h = self._Condensation_Correlation(cell_index, nb_plates)

        return alpha_c, alpha_h, f_c, f_h
    

    """
    This method implements the Zhang et al. (2020) correlation to estimate the convective heat transfer 
    coefficient of condensing fluid (ideally R290) inside a plate heat exchanger.
        - Input : 
            - cell_index (index of the cell for which we want to calculate the heat transfer coefficient)
            - nb_plates (number of plates in the heat exchanger)
        - Output : 
            - h (heat transfer coefficient) [W/m2K]
            - f (friction factor) [-]
    
    """
    def _Condensation_Correlation(self, cell_index, nb_plates):

        # Liquid properties
        self.HEOS_hot.update(CoolProp.PQ_INPUTS, self.state_in_h.p, 0)
        mu_l = self.HEOS_hot.viscosity()
        rho_l = self.HEOS_hot.rhomass()
        Pr_l = self.HEOS_hot.Prandtl()
        k_l = self.HEOS_hot.conductivity()

        # Vapor properties
        self.HEOS_hot.update(CoolProp.PQ_INPUTS, self.state_in_h.p, 1)
        rho_v = self.HEOS_hot.rhomass()

        self.HEOS_hot.update(CoolProp.HmassP_INPUTS, self.EnthalpyVector_h[cell_index], self.state_in_h.p)
        x_start = self.HEOS_hot.Q()
        self.HEOS_hot.update(CoolProp.HmassP_INPUTS, self.EnthalpyVector_h[cell_index+1], self.state_in_h.p)
        x_end = self.HEOS_hot.Q()
        x_m = (x_start + x_end) / 2

        # Equivalent Re calculation

        Nb_channels = nb_plates - 1
        Nb_channels_h = Nb_channels // 2
        G = self.mdot_h / Nb_channels_h / self.Aflow
        Geq = G*((1-x_m) + x_m*(rho_l/rho_v)**0.5)

        Re_eq = Geq * self.Dh / mu_l

        # Calculation of Bd

        self.HEOS_hot.update(CoolProp.PQ_INPUTS, self.state_in_h.p, 0)
        sigma = self.HEOS_hot.surface_tension()
        g = 9.81
        Bd = g * (rho_l - rho_v) * self.Dh**2 / sigma

        # Calculation of rho_star
        rho_star = rho_l / rho_v

        # Calculation of h

        h = 0.4703 * Re_eq**0.5221 * Pr_l**(1/3) * Bd**0.1674 * (rho_star)**(0.2126) * k_l / self.Dh
        h *= self.corr_factors["h_condensation"]

        # Calculation of friction factor
        f =  11557.62 * Re_eq**(-1.0041) * Bd**0.3002 * (rho_star)**(-0.4268) 
        f *= self.corr_factors["f_condensation"]

        return h, f


    """
    This method implements the Yang et al. (2016) correlation to estimate the convective heat transfer
    coefficient of single phase fluid (ideally water) inside a brazed plate heat exchanger and the 
    Kim and Park (2017) correlation to estimate the friction factor.
        - Inputs :
            - cell_index : index of the cell for which we want to calculate the heat transfer coefficient
            - stream : "hot" or "cold" to indicate which stream we are considering
            - nb_plates : number of plates in the heat exchanger
        - Output : 
            - h (heat transfer coefficient) [W/m2K]
            - f (friction factor) [-]

    """
    def _SinglePhase_Correlation(self, cell_index, stream, nb_plates):

        T_start_h = self.TemperatureVector_h[cell_index]
        T_end_h = self.TemperatureVector_h[cell_index+1]
        Tm_h = (T_start_h + T_end_h) / 2

        T_start_c = self.TemperatureVector_c[cell_index]
        T_end_c = self.TemperatureVector_c[cell_index+1]
        Tm_c = (T_start_c + T_end_c) / 2

        Tw = (Tm_h + Tm_c) / 2  # Wall temperature, we take it as the average of the mean temperatures of both streams

        Nb_channels = nb_plates - 1
        Nb_channels_h = Nb_channels // 2
        Nb_channels_c = Nb_channels - Nb_channels_h

        if stream == "hot":
            mdot = self.mdot_h
            Tm = Tm_h
            self.HEOS_hot.update(CoolProp.PT_INPUTS, self.state_in_h.p, Tm)
            mu = self.HEOS_hot.viscosity()
            Pr = self.HEOS_hot.Prandtl()
            k = self.HEOS_hot.conductivity()
            Nb_channels_stream = Nb_channels_h
            self.HEOS_hot.update(CoolProp.PT_INPUTS, self.state_in_h.p, Tw)
            mu_wall = self.HEOS_hot.viscosity()
        else :
            mdot = self.mdot_c
            Tm = Tm_c   
            self.HEOS_cold.update(CoolProp.PT_INPUTS,self.state_in_c.p, Tm)
            mu = self.HEOS_cold.viscosity()
            Pr = self.HEOS_cold.Prandtl()
            k = self.HEOS_cold.conductivity()
            Nb_channels_stream = Nb_channels_c
            self.HEOS_cold.update(CoolProp.PT_INPUTS, self.state_in_c.p, Tw)
            mu_wall = self.HEOS_cold.viscosity()

        # Calculation of Reynolds number
        G = mdot / Nb_channels_stream / self.Aflow
        Re_Yang = G * self.Dh * self.phi / mu
        Re = self.Dh * G / mu

        # Calculation of h
        Nu = (-1.342e-4 * self.beta**2 + 1.808e-2 * self.beta - 0.0075) * Re_Yang**(-7.956e-5 * self.beta**2 + 9.687e-3 * self.beta + 0.3155) * Re_Yang**(self.phi/self.beta) * Re_Yang**(self.gamma/self.beta) * Pr**(1/3) * (mu/mu_wall)**0.14
        h = Nu * k / (self.Dh * self.phi)
        h *= self.corr_factors["h_single_phase"]

        # Calculation of friction factor 
        f = 4 * self.phi**4 * (0.6796 * self.phi * Re**(-0.0551) + 0.2)
        f *= self.corr_factors["f_single_phase"]

        return h,f


    """
    This method implements the Amalfi et al. (2015) correlation to estimate the convective heat transfer
    coefficient of evaporating fluid inside a plate heat exchanger.
        - Input : 
            - cell_index : index of the cell for which we want to calculate the heat transfer coefficient
            - nb_plates : number of plates in the heat exchanger
        - Output : 
            - h (heat transfer coefficient) [W/m2K]
            - f (friction factor) [-]
    
    """
    def _Evaporation_Correlation(self, cell_index, nb_plates):

        beta_max = 70          # Maximum chevron angle [degrees]
        q_prime_prime = self.q_prime_prime_per_cell[cell_index]  # Heat flux in cell "cell_index" [W/m²]

        # Saturated liquid properties
        self.HEOS_cold.update(CoolProp.PQ_INPUTS, self.state_in_c.p, 0)
        rho_l = self.HEOS_cold.rhomass()
        kl = self.HEOS_cold.conductivity()
        mu_l = self.HEOS_cold.viscosity()
        sigma = self.HEOS_cold.surface_tension()
        h_l = self.HEOS_cold.hmass()

        # Saturated vapor properties
        self.HEOS_cold.update(CoolProp.PQ_INPUTS, self.state_in_c.p, 1)
        rho_v = self.HEOS_cold.rhomass()
        h_v = self.HEOS_cold.hmass()
        mu_v = self.HEOS_cold.viscosity()

        h_lv = h_v - h_l  # Latent heat of vaporization [J/kg]

        # Average quality in the cell
        self.HEOS_cold.update(CoolProp.HmassP_INPUTS, self.EnthalpyVector_c[cell_index], self.state_in_c.p)
        x_start = self.HEOS_cold.Q()
        self.HEOS_cold.update(CoolProp.HmassP_INPUTS, self.EnthalpyVector_c[cell_index+1], self.state_in_c.p)
        x_end = self.HEOS_cold.Q()
        x_m = (x_start + x_end) / 2

        # Average properties in the cell
        rho_m = 1 / (x_m/rho_v + (1-x_m)/rho_l)  # Mixture density using the ideal mixture rule

        # Number of channels for this stream
        Nb_channels = nb_plates - 1
        Nb_channels_h = Nb_channels // 2
        Nb_channels_c = Nb_channels - Nb_channels_h

        G = self.mdot_c / Nb_channels_c / self.Aflow

        # Calculation of Bd
        g = 9.81
        Bd = g * (rho_l - rho_v) * self.Dh**2 / sigma

        # Calculation of Bo
        Bo = q_prime_prime / (G * h_lv)

        # Calculation of C
        C = 2.125 * (self.beta/beta_max)**9.993 + 0.955

        # Calculation of We_m
        We_m = G**2 * self.Dh / (rho_m * sigma)

        # Calculation of rho_star
        rho_star = rho_l / rho_v

        # Calculation of Re_v
        Re_v = x_m * G * self.Dh / mu_v

        # Calculation of Re_lo
        Re_lo = G * self.Dh / mu_l

        # Calculation of h
        if Bd < 4 :
            h = 982 * (kl / self.Dh) * (self.beta/beta_max)**1.101 * We_m**0.315 * Bo**0.320 * rho_star**(-0.224)
        else :
            h = 18.495 * (kl / self.Dh) * (self.beta/beta_max)**0.248 * Re_v**0.135 * Re_lo**0.351 * Bd**0.235 * Bo**0.198 * rho_star**(-0.223)

        h *= self.corr_factors["h_evaporation"]

        # Calculation of f
        f = 4 * C * 15.698 * We_m**(-0.475) * Bd**(0.255) * rho_star**(-0.571)
        f *= self.corr_factors["f_evaporation"]

        return h, f
    

    """
    This method implements the Zendehboudi et al. (2021) correlation to estimate the convective heat transfer
    coefficient of supercritical fluid inside a brazed plate heat exchanger and the 
    Lee et al. (2020) correlation to estimate the friction factor.
        - Inputs :
            - cell_index : index of the cell for which we want to calculate the heat transfer coefficient
            - stream : "hot" or "cold" to indicate which stream we are considering
            - nb_plates : number of plates in the heat exchanger
        - Output : 
            - h (heat transfer coefficient) [W/m2K]
            - f (friction factor) [-]

    """
    def _Supercritical_Correlation(self, cell_index, stream, nb_plates):

        # Calculation of the bulk temperature
        T_start_h = self.TemperatureVector_h[cell_index]
        T_end_h = self.TemperatureVector_h[cell_index+1]
        Tm_h = (T_start_h + T_end_h) / 2
        
        T_start_c = self.TemperatureVector_c[cell_index]
        T_end_c = self.TemperatureVector_c[cell_index+1]
        Tm_c = (T_start_c + T_end_c) / 2

        Nb_channels = nb_plates - 1
        Nb_channels_h = Nb_channels // 2
        Nb_channels_c = Nb_channels - Nb_channels_h

        if stream == "hot":
            Tm = Tm_h
            heos = self.HEOS_hot
            mdot = self.mdot_h
            p = self.state_in_h.p
            Nb_channels_stream = Nb_channels_h
        else :
            Tm = Tm_c
            heos = self.HEOS_cold
            mdot = self.mdot_c
            p = self.state_in_c.p
            Nb_channels_stream = Nb_channels_c
        
        # Wall temperature estimation
        Tw = (Tm_h + Tm_c) / 2

        # Fluid properties at Tb
        heos.update(CoolProp.PT_INPUTS, p, Tm)
        rho_m = heos.rhomass()
        Cp_m = heos.cpmass()
        mu_m = heos.viscosity()
        k_m = heos.conductivity()
        h_m = heos.hmass()

        # Fluid properties at Tw
        heos.update(CoolProp.PT_INPUTS, p, Tw)
        rho_w = heos.rhomass()
        h_w = heos.hmass()

        cp_bar = (h_w - h_m) / (Tw - Tm)
        Pr_m = cp_bar * mu_m / k_m

        # Calculation of Re
        G = mdot / Nb_channels_stream / self.Aflow
        Re_m = self.Dh * G / mu_m

        # Calculation of Gr
        N = 5
        T_values = np.linspace(Tw, Tm, N)
        rho_values = np.zeros(len(T_values))
        for i, T in enumerate(T_values):
            heos.update(CoolProp.PT_INPUTS, p, T)
            rho_values[i] = heos.rhomass()
        
        rho_w_bar = np.abs(np.trapz(rho_values, T_values) / (Tm - Tw))
        g = 9.81
        Gr = g * self.Dh**3 * (rho_w_bar - rho_m) * rho_m / mu_m**2

        # Calculation of Nu and h
        Nu = 0.33 * Re_m**(0.804) * Pr_m**0.1 * (rho_w/rho_m)**(-0.1) * (cp_bar/Cp_m)**0.093 * (Gr/Re_m**2.7)**0.1 
        h = Nu * k_m / self.Dh
        h *= self.corr_factors["h_supercritical"]

        # Calculation of f
        f1 = 2.332e7 * Re_m**(-1.537)
        f2 = 1.129e6 * Re_m**(-1.075)
        if Re_m < 5000 :
            f = f1
        elif Re_m > 6500 :
            f = f2
        else :
            f = f1 + (f2 - f1) * (Re_m - 5000) / (6500 - 5000)  # Linear interpolation between the two regimes

        f *= self.corr_factors["f_supercritical"]

        return h, f


    """
    This method computes the average specific heat capacity of a fluid between two temperatures.
        - Inputs :
            - heos : CoolProp HEOS object of the fluid
            - T1 : first temperature [K]
            - T2 : second temperature [K]
        - Output :
            - Cp_avg : average specific heat capacity [J/kg/K]

    """
    def _Cp_average(self, heos, p, T1, T2):

        # Discretization of the temperature range
        N = 50
        T_values = np.linspace(T1, T2, N)

        # Calculation of Cp at each temperature
        Cp_values = np.zeros(len(T_values))
        for i, T in enumerate(T_values):
            heos.update(CoolProp.PT_INPUTS, p, T)
            Cp_values[i] = heos.cpmass()

        # Numerical integration using the trapezoidal rule
        Cp_avg = np.trapz(Cp_values, T_values) / (T2 - T1)

        return Cp_avg
    

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
        - Inputs : 
            - save (boolean) : to indicate if the plot should be saved as a PNG file
            - name_cycle (str) : name of the cycle in which the HEX is integrated (used for saving the plot in the right folder)
            - plot (boolean) : to indicate if the plot should be displayed or not
    
    """
    def _plot(self, save=False, name_cycle=None, plot=True):
        
        if plot is False and save is False:
            return

        self._get_Normalized_EnthalpyVectors()

        # Compute the yticks in °C
        Tmin_c = round(min(self.TemperatureVector_c) - 273.15, ndigits=1)
        Tmax_c = round(max(self.TemperatureVector_c) - 273.15, ndigits=1)
        Tmin_h = round(min(self.TemperatureVector_h) - 273.15, ndigits=1)
        Tmax_h = round(max(self.TemperatureVector_h) - 273.15, ndigits=1)
        Tmax = max(Tmax_c, Tmax_h)
        Tmin = min(Tmin_c, Tmin_h)
        yticks = np.array([Tmin, Tmax])
        yticks = np.unique(yticks)

        # Compute the xticks
        xticks = np.array([0, 1]) 
        
        # Create the plot
        plt.figure(self.name)
        plt.plot(self.Normalized_EnthalpyVector_c, self.TemperatureVector_c - 273.15, marker='o', color="blue", clip_on=False)
        plt.plot(self.Normalized_EnthalpyVector_h, self.TemperatureVector_h - 273.15, marker='o', color="red", clip_on=False)
        plt.xlabel(r"$\hat{h}$ [-]", fontsize=12)
        plt.xlim(0,1)
        plt.ylim(yticks[0], yticks[-1])

        # Customize the axes
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.set_title('Temperature [°C]', loc='left', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.spines['left'].set_position(('outward', 15))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        plt.tick_params(axis='x', rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
        plt.tight_layout()

        if save and (name_cycle is not None):
            fig_dir = f'code/Figures/{name_cycle}'
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(f'{fig_dir}/{self.name}.pdf')

        if plot : plt.show()


    """
    This method plots the convergence process of the design evaluation for different number of plates in the heat exchanger.
    More precisely, it plots the evolution of the area ratio (Ageom/Areq) and the pressure drop ratios (dP_hot/dP_hot_max and dP_cold/dP_cold_max) as a function of the number of plates.
        - Inputs :
            - N_list : list of the number of plates tested during the convergence process
            - Areq_list : list of the required heat transfer area for each number of plates
            - Ageom_list : list of the geometric heat transfer area for each number of plates
            - dPh_list : list of the pressure drop for the hot stream for each number of plates
            - dPc_list : list of the pressure drop for the cold stream for each number of plates
            - deltaP_hot_max : maximum allowed pressure drop for the hot stream (used for normalization)
            - deltaP_cold_max : maximum allowed pressure drop for the cold stream (used for normalization)
            - save (boolean) : to indicate if the plot should be saved as a PNG file
            - name_cycle (str) : name of the cycle in which the HEX is integrated (used for saving the plot in the right folder)
    """
    def _plot_design_results(self, N_list, Areq_list, Ageom_list, dPh_list, dPc_list, deltaP_hot_max, deltaP_cold_max, save=False,name_cycle=None):

        # Convert the lists to numpy arrays
        N_array = np.array(N_list)
        Areq_array = np.array(Areq_list)
        Ageom_array = np.array(Ageom_list)
        dPh_array = np.array(dPh_list)
        dPc_array = np.array(dPc_list)

        # Normalized quantities
        area_ratio = Ageom_array / Areq_array
        dPh_ratio = dPh_array / deltaP_hot_max
        dPc_ratio = dPc_array / deltaP_cold_max

        max_area_ratio = np.max(area_ratio)
        max_dPh_ratio = np.max(dPh_ratio)
        max_dPc_ratio = np.max(dPc_ratio)
        max_ratio = np.round(max(max_area_ratio, max_dPh_ratio, max_dPc_ratio))

        # Y ticks
        yticks = np.array([0, 1.0, min(5, max_ratio)])
        
            # Replace values above 5 with NaN to avoid plotting them
        area_ratio = np.where(area_ratio > yticks[-1], np.nan, area_ratio)
        dPh_ratio = np.where(dPh_ratio > yticks[-1], np.nan, dPh_ratio)
        dPc_ratio = np.where(dPc_ratio > yticks[-1], np.nan, dPc_ratio)

        # X ticks
        xticks = np.array([N_array[0], N_array[-1]])

        plt.figure(self.name + "_Design")
        plt.plot(N_array, area_ratio, color="black",marker=".", clip_on=False, label=r"$\frac{A_{\mathrm{geom}}}{A_{\mathrm{required}}}$")
        plt.plot(N_array, dPh_ratio, color="red",marker=".", clip_on=False, label=r"$\frac{\Delta p_{h}}{\Delta p_{h}^{\mathrm{max}}}$")
        plt.plot(N_array, dPc_ratio,color="blue",marker=".", clip_on=False, label=r"$\frac{\Delta p_{c}}{\Delta p_{c}^{\mathrm{max}}}$")

        # Constraint line
        plt.axhline(1.0, linestyle='--', linewidth=1, color='gray')

        plt.xlabel("Number of plates [-]", fontsize=12)
        plt.xlim(N_array[0], N_array[-1])
        plt.ylim(yticks[0], yticks[-1])

        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.set_title('Normalized design criteria [-]', loc='left', fontsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.spines['left'].set_position(('outward', 15))

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        plt.tick_params(axis='x', rotation=0)
        plt.tick_params(axis='both', which='major',
                        labelsize=11,
                        direction='in')

        plt.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black')
        plt.tight_layout()

        if save and (name_cycle is not None):
            fig_dir = f'code/Figures/{name_cycle}'
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(f'{fig_dir}/{self.name}_Design.pdf')

        plt.show()


    def energy_analysis(self, state_in, state_out, mdot_wf, mdot_secondary, state_in_secondary, state_out_secondary):

        P_wf = (state_out.h - state_in.h) * mdot_wf
        P_secondary = (state_out_secondary.h - state_in_secondary.h) * mdot_secondary
        P_loss = abs(abs(P_wf) - abs(P_secondary))

        dict_energy = {
            "P_{wf}": P_wf,
            "P_{secondary}": P_secondary,
            "P_{loss}": P_loss
        }
    
        return dict_energy
    

    def exergy_analysis(self, T0, P0, state_in, state_out, mdot_wf, mdot_secondary, state_in_secondary, state_out_secondary):

        P_wf = (state_out.exergy(T0, P0) - state_in.exergy(T0, P0)) * mdot_wf
        P_secondary = (state_out_secondary.exergy(T0, P0) - state_in_secondary.exergy(T0, P0)) * mdot_secondary
        P_loss = -(P_wf + P_secondary)

        dict_exergy = {
            "P_{wf}": P_wf,
            "P_{secondary}": P_secondary,
            "P_{irr}": P_loss
        }

        return dict_exergy