
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


def CAD(compressors_inputs, valves_inputs, HP_evaporator, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, save_results = True, plot_figures = False) :
    # INPUTS
        # 1. Frequencies of the compressors
    N_LP = compressors_inputs['N_LP']
    N_HP = compressors_inputs['N_HP']

        # 2. Power of the compressors
    P_LP = compressors_inputs['P_LP']
    P_HP = compressors_inputs['P_HP']

        # 3. Valve openings
    z_LP = valves_inputs['z_LP']
    z_HP = valves_inputs['z_HP']

    # PARAMETERS
        # 1. Component parameters
    compressor_LP = Compressor_LP()
    compressor_HP = Compressor_HP()
    valve_LP = Valve([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
    valve_HP = Valve([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
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
    mdot_external_fluid_LT = external_fluid_LT_param['mdot_external_fluid_LT']                      # Mass flow rate of the external fluid in the heat source [kg/s] 
    heos_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
    cycle.state_1_prime = State(heos = heos_external_fluid_LT, T = T1_prime, p = p1_prime)

        # b. MT source
    external_fluid_MT = 'Water'                                                                     # External fluid in the heat sink
    T3_prime = external_fluid_MT_param['T3_prime']                                                  # Inlet temperature of the external fluid in the heat sink [K] 
    p3_prime = external_fluid_MT_param['p3_prime']                                                  # Inlet pressure of the external fluid in the heat sink [Pa]
    mdot_external_fluid_MT = external_fluid_MT_param['mdot_external_fluid_MT']                      # Mass flow rate of the external fluid in the heat sink [kg/s]
    heos_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
    cycle.state_3_prime = State(heos = heos_external_fluid_MT, T = T3_prime, p = p3_prime)

        # c. HT source
    external_fluid_HT = 'Water'                                                                     # External fluid in the heat sink
    T5_prime = external_fluid_HT_param['T5_prime']                                                     # Inlet temperature of the external fluid in the heat sink [K] 
    p5_prime = external_fluid_HT_param['p5_prime']                                                     # Inlet pressure of the external fluid in the heat sink [Pa]
    mdot_external_fluid_HT = external_fluid_HT_param['mdot_external_fluid_HT']                         # Mass flow rate of the external fluid in the heat sink [kg/s]
    heos_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
    cycle.state_5_prime = State(heos = heos_external_fluid_HT, T = T5_prime, p = p5_prime)



    def objective(args) :
        p_1, p_3, p_5, T_3, mdot_bottom, mdot_top = args
        p_1 = np.exp(p_1)
        p_3 = np.exp(p_3)
        p_5 = np.exp(p_5)
        T_3 = np.exp(T_3)
        mdot_bottom = np.exp(mdot_bottom)
        mdot_top = np.exp(mdot_top)
        #print(args)

        try : 
            cycle.mdot_wf_bottom = mdot_bottom
            cycle.mdot_wf_top = mdot_top
            cycle.state_3 = State(heos = heos_working_fluid, T = T_3, p = p_3)

            if P_HP == 0: 
                P_HP_calc = 0
                mdot_top_calc = 0
                cycle.state_5 = State(heos = heos_working_fluid, T = T_3, p = p_3)
            else:
                # HP Compressor
                T_5_calc, P_HP_calc, mdot_top_calc = compressor_HP.Solve(cycle.state_3, p_5, N_HP)
                cycle.state_5 = State(heos = heos_working_fluid, T = T_5_calc, p = p_5)
                #print("State 3:", cycle.state_3)
                #print("State 5:", cycle.state_5)

            # Condenser
            condenser = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], mdot = [mdot_external_fluid_HT, cycle.mdot_wf_top], name = 'Condenser', N = 86, model = "ACP70X")
            cycle.state_6_prime, cycle.state_6 = condenser.Solve()[0]
            #print("State 6:", cycle.state_6)
            
            # LP Valve
            p_10, h_10 = valve_LP.Solve(cycle.state_6, cycle.mdot_wf_bottom, z_LP)
            cycle.state_10 = State(heos = heos_working_fluid, p = p_10, h = h_10)
            #print("State 10:", cycle.state_10)

            # LP Evaporator
            evaporator_LP = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], mdot = [cycle.mdot_wf_bottom, mdot_external_fluid_LT], name = 'Evaporator_LP', N = 57, model = "ACP70X")
            cycle.state_1, cycle.state_2_prime = evaporator_LP.Solve()[0]
            #print("State 1:", cycle.state_1)

            # LP Compressor
            T_3_comp, P_LP_calc, mdot_bottom_calc = compressor_LP.Solve(cycle.state_1, p_3, N_LP)
            cycle.state_3_comp = State(heos = heos_working_fluid, T = T_3_comp, p = p_3)

            if HP_evaporator:
                # HP Valve
                p_8, h_8 = valve_HP.Solve(cycle.state_6, cycle.mdot_wf_top - cycle.mdot_wf_bottom, z_HP)
                cycle.state_8 = State(heos = heos_working_fluid, p = p_8, h = h_8)

                # HP Evaporator
                evaporator_HP = HEX_Operational(states_in=[cycle.state_8, cycle.state_3_prime], mdot = [cycle.mdot_wf_top - cycle.mdot_wf_bottom, mdot_external_fluid_MT], name = 'Evaporator_HP', N = 31, model = "ACP70X")
                cycle.state_3_evap, cycle.state_4_prime = evaporator_HP.Solve()[0]

                # Isenthalpic mixing
                p_3_calc = cycle.state_3_evap.p
                h_3 = ((cycle.mdot_wf_top - cycle.mdot_wf_bottom) * cycle.state_3_evap.h + cycle.mdot_wf_bottom * cycle.state_1.h) / (cycle.mdot_wf_top)
                cycle.state_3 = State(heos = heos_working_fluid, p = p_3, h = h_3)
                T_3_calc = cycle.state_3.T
            
            else: 
                p_3_calc = p_3
                T_3_calc = T_3_comp
            
            residual = [(P_HP_calc - P_HP) / P_HP, (P_LP_calc - P_LP) / P_LP, (mdot_top_calc - mdot_top) / mdot_top, (mdot_bottom_calc - mdot_bottom) / mdot_bottom, (p_3_calc - p_3) / p_3, (T_3_calc - T_3) / T_3]
            print(residual)
            if np.max(np.abs(residual)) < 5e-3:
                residual = [0, 0, 0, 0, 0, 0]
        
            return residual
            
        except Exception as e:
            print(f"Error in calculation: {e}")
            return [1e6, 1e6, 1e6, 1e6, 1e6, 1e6]  # Return large residuals to indicate failure

    
    print("Starting simulation and optimization...")
    start = time.time()
    for method in ['hybr', 'lm'] :
        p_1, p_3, p_5, T_3, mdot_bottom, mdot_top = root(objective, initial_guess, options={'col_deriv': True, 'maxfev': 50}, method = method).x
        residual = objective([p_1, p_3, p_5, T_3, mdot_bottom, mdot_top])
        if np.max(np.abs(residual)) < 5e-3:
            with lock:
                with open(f"code/Figures/CAD/results.txt", "a") as f:
                    f.write(f"{P_LP} \t {P_HP} \t {N_LP} \t {N_HP} \t {z_LP} \t {z_HP} \t {np.exp(p_1)} \t {np.exp(p_3)} \t {np.exp(p_5)} \t {np.exp(T_3)} \t {np.exp(mdot_bottom)} \t {np.exp(mdot_top)}\t {np.sqrt(np.sum(np.array(residual)**2))} \n")
                break
    end = time.time()
    print(f"Simulation and optimization completed in {end - start:.2f} seconds.")

    if plot_figures:
        evaporator_LP = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], mdot = [cycle.mdot_wf_bottom, mdot_external_fluid_LT], name = 'Evaporator_LP', N = 57, model = "ACP70X")
        evaporator_LP.Solve()
        condenser = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], mdot = [mdot_external_fluid_HT, cycle.mdot_wf_top], name = 'Condenser', N = 86, model = "ACP70X")
        print(condenser.Solve())
        cycle.transforms = [Transform('comp', '1', '3_comp', compressor_LP), 
                            Transform('comp', '3', '5', compressor_HP),
                            Transform('hex', '10', '1', evaporator_LP, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                            Transform('adex', '6', '10', valve_LP), 
                            Transform('hex', '5', '6', condenser)]

        cycle.Ts_diagram(save = False)
        cycle.ph_diagram(save = False)

    return 

if __name__ == '__main__':

    ############################################################
    # Inputs 
    ############################################################
    # Frequencies of the compressors 
    N_LP = [50]
    N_HP = 50

    # Power of the compressors
    P_LP = [3e3] #np.arange(2.5, 3.6, 0.5) * 1e3
    P_HP = [6e3] #np.arange(4.5, 7.5, 0.5) * 1e3

    compressor_inputs = {'N_LP': N_LP, 'N_HP': N_HP, 'P_LP': P_LP, 'P_HP': P_HP}

    # Valve opening
    z_LP = [40] #np.arange(35,46, 1)
    z_HP = [20]

    valve_inputs = {'z_LP': z_LP, 'z_HP': z_HP}

    # Manual Valve opening
    HP_evaporator = False

    ############################################################
    # Parameters
    ############################################################
    # Heat sources parameters

        # 1. LT source
    T1_prime = 25 + 273.75                          # Inlet temperature of the external fluid in the heat source [K] 
    p1_prime = 3e5                                  # Inlet pressure of the external fluid in the heat source [Pa]
    mdot_external_fluid_LT = 3                      # Mass flow rate of the external fluid in the heat source [kg/s] 
    external_fluid_LT_param = {'T1_prime': T1_prime, 'p1_prime': p1_prime, 'mdot_external_fluid_LT': mdot_external_fluid_LT}


        # 2. MT source
    T3_prime = 40 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
    p3_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
    mdot_external_fluid_MT = 1                      # Mass flow rate of the external fluid in the heat sink [kg/s]
    external_fluid_MT_param = {'T3_prime': T3_prime, 'p3_prime': p3_prime, 'mdot_external_fluid_MT': mdot_external_fluid_MT}


    # Heat sink parameters
    T5_prime = 45 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
    p5_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
    mdot_external_fluid_HT = 1.2                    # Mass flow rate of the external fluid in the heat sink [kg/s]
    external_fluid_HT_param = {'T5_prime': T5_prime, 'p5_prime': p5_prime, 'mdot_external_fluid_HT': mdot_external_fluid_HT}

    ##############################################################
    # Simulation and optimization
    ##############################################################

    #initial_guess = [np.log(5e5), np.log(10e5), np.log(20e5), np.log(330), np.log(0.08), np.log(0.08)]
    initial_guess = [13.122363377404328, 13.951199380764326, 14.59544959400266, 5.774448123746895, -2.4951621337422076, -2.268615953214233]

    # Create a pool of workers
    pool = Pool(processes=cpu_count())
    lock = Lock()

    list_inputs = []

    for nLP, pLP, pHP, zLP in itertools.product(N_LP, P_LP, P_HP, z_LP):
        print(f"Simulating for N_LP={nLP}, P_LP={pLP}, P_HP={pHP}, z_LP={zLP}, z_HP={z_HP}")

        compressor_inputs = {
            'N_LP': nLP,
            'N_HP': N_HP,
            'P_LP': pLP,
            'P_HP': pHP
        }

        valve_inputs = {
            'z_LP': zLP,
            'z_HP': z_HP
        }

        list_inputs.append((
            compressor_inputs,
            valve_inputs,
            HP_evaporator,
            external_fluid_LT_param,
            external_fluid_MT_param,
            external_fluid_HT_param,
            initial_guess
        ))

    # Use starmap for multiple arguments
    results = pool.starmap(CAD, list_inputs)

    pool.close()
    pool.join()


    



