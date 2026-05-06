
############################################################
# Import libraries and modules
############################################################
import sys
from pathlib import Path

# Add the parent directory (code) to sys.path to enable relative imports
code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from components.transform import Transform
from components.state import State
from components.compressor import Compressor_LP, Compressor_HP
from components.valve import Valve, Valve_other
from components.HEX import HEX_Operational
from components.cycle import Cycle
import CoolProp
import numpy as np
from scipy.optimize import fsolve, root, minimize, least_squares
import time
import itertools
from multiprocessing import Pool, cpu_count, Lock

def CAD(Q_cond,compressors_inputs, valves_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = 'hybr', save_results = True, plot_figures = False) :

    # INPUTS
        # 1. Frequencies of the compressors
    N_LP = compressors_inputs['N_LP']
    N_HP = compressors_inputs['N_HP']

        # 3. Valve openings
    z_LP = valves_inputs['z_LP']
    z_HP = valves_inputs['z_HP']

    # PARAMETERS
        # 1. Component parameters
    compressor_LP = Compressor_LP()
    compressor_HP = Compressor_HP()
    valve_LP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
    valve_HP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
    evaporator_LP = None
    evaporator_HP = None
    condenser = None

        # 2. Cycle parameters
    fluid = 'R290'                                  # Working fluid
    heos_working_fluid = CoolProp.AbstractState("HEOS", fluid)
    cycle = Cycle("CAD")
        
        # 3. Heat sources parameters
        # a. LT source
    external_fluid_LT = 'Water'                                                                     # External fluid in the heat source
    T1_prime = external_fluid_LT_param['T1_prime']                                                  # Inlet temperature of the external fluid in the heat source [K] 
    p1_prime = external_fluid_LT_param['p1_prime']                                                  # Inlet pressure of the external fluid in the heat source [Pa]
    glide_external_fluid_LT = external_fluid_LT_param['glide_external_fluid_LT']                      # Temperature glide of the external fluid in the heat source [K]
    heos_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
    cycle.state_1_prime = State(heos = heos_external_fluid_LT, T = T1_prime, p = p1_prime)

        # b. MT source
    external_fluid_MT = 'Water'                                                                     # External fluid in the heat sink
    T3_prime = external_fluid_MT_param['T3_prime']                                                  # Inlet temperature of the external fluid in the heat sink [K] 
    p3_prime = external_fluid_MT_param['p3_prime']                                                  # Inlet pressure of the external fluid in the heat sink [Pa]
    glide_external_fluid_MT = external_fluid_MT_param['glide_external_fluid_MT']                      # Temperature glide of the external fluid in the heat sink [K]
    heos_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
    cycle.state_3_prime = State(heos = heos_external_fluid_MT, T = T3_prime, p = p3_prime)

        # c. HT source
    external_fluid_HT = 'Water'                                                                     # External fluid in the heat sink
    T5_prime = external_fluid_HT_param['T5_prime']                                                     # Inlet temperature of the external fluid in the heat sink [K] 
    p5_prime = external_fluid_HT_param['p5_prime']                                                     # Inlet pressure of the external fluid in the heat sink [Pa]
    glide_external_fluid_HT = external_fluid_HT_param['glide_external_fluid_HT']                         # Temperature glide of the external fluid in the heat sink [K]
    heos_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
    cycle.state_5_prime = State(heos = heos_external_fluid_HT, T = T5_prime, p = p5_prime)

    def objective(args) :
        args = np.exp(args)
        p_3, p_5, h_3, p_10, p_8, mdot_wf_bottom, mdot_wf_top, mdot_LT, mdot_MT, mdot_HT = args
        #print("Current guess:", args)
        
        #print(args)

        try : 
            cycle.state_3 = State(heos = heos_working_fluid, h = h_3, p = p_3)
            #print("State 3:", cycle.state_3)

            cycle.mdot_wf_top = mdot_wf_top

            # HP Compressor
            h_5, P_HP, N_HP_calc = compressor_HP.Solve_2(cycle.state_3, p_5, cycle.mdot_wf_top)
            cycle.state_5 = State(heos = heos_working_fluid, h = h_5, p = p_5)
            #print("State 5 :", cycle.state_5)

            # Condenser
            condenser = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], mdot = [mdot_HT, cycle.mdot_wf_top], name = 'Condenser', N = 86, model = "ACP70X")
            sol_HP = condenser.Solve()
            cycle.state_6_prime, cycle.state_7 = sol_HP[0]
            Q_cond_calc = sol_HP[1]
        

            glide_HT_calc = cycle.state_6_prime.T - cycle.state_5_prime.T
            #print("State 6:", cycle.state_6)

            # Flow separation
            cycle.mdot_wf_bottom = mdot_wf_bottom
            mdot_wf_MP = cycle.mdot_wf_top - cycle.mdot_wf_bottom
            
            # LP Valve
            h_10, z_LP_calc = valve_LP.Solve(cycle.state_7, p_10, cycle.mdot_wf_bottom)
            cycle.state_10 = State(heos = heos_working_fluid, h = h_10, p = p_10)
            #print("State 10:", cycle.state_10)

            # LP Evaporator
            evaporator_LP = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], mdot = [mdot_wf_bottom, mdot_LT], name = 'Evaporator_LP', N = 57, model = "ACP70X")
            sol_LP = evaporator_LP.Solve()
            cycle.state_1, cycle.state_2_prime = sol_LP[0]
            Q_evap_LP_calc = sol_LP[1]
            glide_LT_calc = cycle.state_1_prime.T - cycle.state_2_prime.T

            # LP compressor
            h_3_comp, P_LP_calc, N_LP_calc = compressor_LP.Solve_2(cycle.state_1, p_3, cycle.mdot_wf_bottom)
            cycle.state_3_comp = State(heos = heos_working_fluid, h = h_3_comp, p = p_3)

            # HP valve
            h_8, z_HP_calc = valve_HP.Solve(cycle.state_7, p_8, mdot_wf_MP)
            cycle.state_8 = State(heos = heos_working_fluid, h = h_8, p = p_8)

            # HP evaporator
            evaporator_HP = HEX_Operational(states_in=[cycle.state_8, cycle.state_3_prime], mdot = [mdot_wf_MP, mdot_MT], name = 'Evaporator_HP', N = 31, model = "ACP70X")
            sol_MP = evaporator_HP.Solve()
            cycle.state_3_evap, cycle.state_4_prime = sol_MP[0]
            Q_evap_HP_calc = sol_MP[1]
            glide_MT_calc = cycle.state_3_prime.T - cycle.state_4_prime.T

            # Isobaric mixing
            h_3_calc = (cycle.mdot_wf_bottom * h_3_comp + mdot_wf_MP * cycle.state_3_evap.h) / cycle.mdot_wf_top
            state_3_calc = State(heos = heos_working_fluid, h = h_3_calc, p = cycle.state_3_evap.p)
            heos_working_fluid.update(CoolProp.PQ_INPUTS, state_3_calc.p, 1)

            #alpha_calc = Q_HP_calc / (Q_HP_calc + Q_LP_calc)
            
            residual = [(state_3_calc.h - h_3) / h_3, (state_3_calc.p - p_3) / p_3, (Q_cond_calc - Q_cond) / Q_cond, (z_LP_calc - z_LP) / z_LP,\
                        (z_HP_calc - z_HP) / z_HP, (N_LP_calc - N_LP) / N_LP, (N_HP_calc - N_HP) / N_HP, (glide_LT_calc - glide_external_fluid_LT) / glide_external_fluid_LT, \
                        (glide_MT_calc - glide_external_fluid_MT) / glide_external_fluid_MT, (glide_HT_calc - glide_external_fluid_HT) / glide_external_fluid_HT ]
            print(residual)
            max_residual = np.max(np.abs(residual))
            #print(max_residual)
            if max_residual < 0.005:
                residual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
            return residual
            
        except Exception as e:
            print(f"Error in calculation: {e}")
            return [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]  # Return large residuals to indicate failure


    print("Starting simulation and optimization...")
    start = time.time()
    initial_guess = np.log(np.array([1170309.90, 2627387.70, 642609.11, 566474.68, 1173567.36, 0.0416, 0.0746, 0.4594, 0.4033, 1.1953]))
    method = 'hybr'  # You can also try 'lm' or 'trf'
    sol = np.exp(root(objective, initial_guess, method = method).x)
    p_3, p_5, h_3, p_10, p_8, mdot_wf_bottom, mdot_wf_top, mdot_LT, mdot_MT, mdot_HT = sol
    end = time.time()
    print(f"Simulation and optimization completed in {end - start:.2f} seconds.")

    if save_results :
        residual = objective(np.log(sol))
        with open(f"code/Figures/CAD/results.txt", "a") as f:
            #f.write("Q_cond \t N_LP \t N_HP \t z_LP \t z_HP \t p_3 \t p_5 \t h_3 \t p_10 \t p_8 \t mdot_wf_bottom \t mdot_wf_top \t mdot_LT \t mdot_MT \t mdot_HT \t Sum of resilduals \t time \n")
            f.write(f"{Q_cond:.2f} \t {N_LP:.2f} \t {N_HP:.2f} \t {z_LP:.2f} \t {z_HP:.2f} \t {p_3:.2f} \t {p_5:.2f} \t {h_3:.2f} \t {p_10:.2f} \t {p_8:.2f} \t {mdot_wf_bottom:.4f} \t {mdot_wf_top:.4f} \t {mdot_LT:.4f} \t {mdot_MT:.4f} \t {mdot_HT:.4f} \t {np.sum(np.abs(residual)):.2f} \t {end - start:.2f}\n")
    
    if plot_figures :
        evaporator_LP = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], mdot = [cycle.mdot_wf_bottom, mdot_LT], name = 'Evaporator_LP', N = 57, model = "ACP70X")
        Q_evap_LP = evaporator_LP.Solve()[1]
        evaporator_LP._plot()

        mdot_wf_MP = cycle.mdot_wf_top - cycle.mdot_wf_bottom
        evaporator_HP = HEX_Operational(states_in=[cycle.state_8, cycle.state_3_prime], mdot = [mdot_wf_MP, mdot_MT], name = 'Evaporator_HP', N = 31, model = "ACP70X")
        Q_evap_HP = evaporator_HP.Solve()[1]
        evaporator_HP._plot()
        #print(evaporator_HP)

        condenser = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], mdot = [mdot_HT, cycle.mdot_wf_top], name = 'Condenser', N = 86, model = "ACP70X")
        Q_cond = condenser.Solve()[1]
        condenser._plot()

        #print(Q_evap_LP, Q_evap_HP, Q_cond)

        cycle.transforms = [Transform('comp', '1', '3_comp', compressor_LP), 
                            Transform('hex', '10', '1', evaporator_LP, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                            Transform('adex', '7', '10', valve_LP), 
                            Transform('hex', '5', '7', condenser), 
                            Transform('adex', '7', '8', valve_HP),
                            Transform('hex', '8', '3_evap', evaporator_HP, label_in_secondary='3_prime', label_out_secondary='4_prime'), 
                            Transform('comp', '3', '5', compressor_HP), 
                            Transform('isobaric_mixing', '3_comp', '3_evap', None)]
        #print(cycle.mdot_wf_bottom)
        cycle.Ts_diagram(save = False)
        cycle.ph_diagram(save = False)

    return 

if __name__ == '__main__':

    ############################################################
    # Inputs 
    ############################################################

    # Condenser heat
    Q_cond = 27e3  # Condenser heat load [W]

    # Frequencies of the compressors 
    N_LP = 40
    N_HP = 60
    
    compressor_inputs = {'N_LP': N_LP, 'N_HP': N_HP}

    # Valve opening
    z_LP = 15.4 
    z_HP = 14.5

    valve_inputs = {'z_LP': z_LP, 'z_HP': z_HP}

    # Variation of the inputs

    # Condenser heat load variation
    dQ_cond = None
    #dQ_cond = 2e3

    # Compressor frequencies variation
    dN_LP = None
    #dN_LP = 5
    dN_HP = None
    #dN_HP = 5

    # Valve opening variation
    dz_LP = None
    #dz_LP = 2
    dz_HP = None
    #dz_HP = 2

    ############################################################
    # Parameters
    ############################################################
    # Heat sources parameters

        # 1. LT source
    T1_prime = 15 + 273.75                          # Inlet temperature of the external fluid in the heat source [K] 
    p1_prime = 3e5                                  # Inlet pressure of the external fluid in the heat source [Pa]
    glide_external_fluid_LT = 5                      # Mass flow rate of the external fluid in the heat source [kg/s] 
    external_fluid_LT_param = {'T1_prime': T1_prime, 'p1_prime': p1_prime, 'glide_external_fluid_LT': glide_external_fluid_LT}


        # 2. MT source
    T3_prime = 40 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
    p3_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
    glide_external_fluid_MT = 5                      # Mass flow rate of the external fluid in the heat sink [kg/s]
    external_fluid_MT_param = {'T3_prime': T3_prime, 'p3_prime': p3_prime, 'glide_external_fluid_MT': glide_external_fluid_MT}


    # Heat sink parameters
    T5_prime = 60 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
    p5_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
    glide_external_fluid_HT = 5                    # Mass flow rate of the external fluid in the heat sink [kg/s]
    external_fluid_HT_param = {'T5_prime': T5_prime, 'p5_prime': p5_prime, 'glide_external_fluid_HT': glide_external_fluid_HT}

initial_guess = np.log(np.array([1170309.90, 2627387.70, 642609.11, 566474.68, 1173567.36, 0.0416, 0.0746, 0.4594, 0.4033, 1.1953]))
method = 'hybr'  # You can also try 'lm'

if dQ_cond is None and dN_LP is None and dN_HP is None and dz_LP is None and dz_HP is None:
    CAD(Q_cond, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = True)

# Variation of Q_cond
if dQ_cond is not None:
    print("Starting variation of Q_cond...")
    Q_cond_up = Q_cond + dQ_cond
    Q_cond_low = Q_cond - dQ_cond
    CAD(Q_cond_low, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    CAD(Q_cond_up, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    print("Variation of Q_cond completed.")

# Variation of N_LP
if dN_LP is not None:
    print("Starting variation of N_LP...")
    N_LP_up = N_LP + dN_LP
    N_LP_low = N_LP - dN_LP
    compressor_inputs_up = {'N_LP': N_LP_up, 'N_HP': N_HP}
    compressor_inputs_low = {'N_LP': N_LP_low, 'N_HP': N_HP}
    CAD(Q_cond, compressor_inputs_low, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    CAD(Q_cond, compressor_inputs_up, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    print("Variation of N_LP completed.")

# Variation of N_HP
if dN_HP is not None:
    print("Starting variation of N_HP...")
    N_HP_up = N_HP + dN_HP
    N_HP_low = N_HP - dN_HP
    compressor_inputs_up = {'N_LP': N_LP, 'N_HP': N_HP_up}
    compressor_inputs_low = {'N_LP': N_LP, 'N_HP': N_HP_low}
    CAD(Q_cond, compressor_inputs_low, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    CAD(Q_cond, compressor_inputs_up, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    print("Variation of N_HP completed.")

# Variation of z_LP
if dz_LP is not None:
    print("Starting variation of z_LP...")
    z_LP_up = z_LP + dz_LP
    z_LP_low = z_LP - dz_LP
    valve_inputs_up = {'z_LP': z_LP_up, 'z_HP': z_HP}
    valve_inputs_low = {'z_LP': z_LP_low, 'z_HP': z_HP}
    CAD(Q_cond, compressor_inputs, valve_inputs_low, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    CAD(Q_cond, compressor_inputs, valve_inputs_up, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    print("Variation of z_LP completed.")

# Variation of z_HP
if dz_HP is not None:
    print("Starting variation of z_HP...")
    z_HP_up = z_HP + dz_HP
    z_HP_low = z_HP - dz_HP
    valve_inputs_up = {'z_LP': z_LP, 'z_HP': z_HP_up}
    valve_inputs_low = {'z_LP': z_LP, 'z_HP': z_HP_low}
    CAD(Q_cond, compressor_inputs, valve_inputs_low, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    CAD(Q_cond, compressor_inputs, valve_inputs_up, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, method = method, save_results = True, plot_figures = False)
    print("Variation of z_HP completed.")