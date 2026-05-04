
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
from components.compressor import Compressor_2_param
from components.HEX import HEX_Design
from components.cycle import Cycle
import CoolProp
import numpy as np
from scipy.optimize import fsolve, minimize, root, differential_evolution
from time import time
import pandas as pd

"""
    This function is inspired from the TC2R.py script.
 
"""
def run_study_cycle_dual_evaporator(alpha, glide_HT, T_MT, verbose=False, print_results=False, plot_results=False, save_results=False, warnings=False):

    start = time()

    ############################################################
    # Parameters
    ############################################################

    # Technological parameters
    T_pinch = 3                             # Minimum temperature difference in heat exchangers [K]
    eta_v = 1                               # Volumetric efficiency
    eta_is = 0.70                           # Isentropic efficiency of the compressor
    eta_elme = 0.95                         # Electrical-mechanical efficiency
    recuperator_effectiveness = 0.8         # Effectiveness of the recuperator

    # Cycle parameters
    working_fluid = 'R290'              # Working fluid
    Q = 25000                           # Power output at the condenser [W]

    # Heat sources parameters

        # 1. LT source
    external_fluid_LT = 'Water'                             # External fluid in the LT heat source
    T1_prime = 15 + 273.75                                  # Inlet temperature of the external fluid in the LT heat source [K]
    glide_LT = 5                                            # Temperature glide of the external fluid in the LT heat source [K]
    T2_prime = T1_prime - glide_LT                          # Outlet temperature of the external fluid in the LT heat source [K]
    p1_prime = 3e5                                          # Inlet pressure of the external fluid in the LT heat source [Pa]

        # 2. MT source
    external_fluid_MT = 'Water'                             # External fluid in the MT heat source
    T3_prime = T_MT                                         # Inlet temperature of the external fluid in the MT heat source [K]
    glide_MT = 5                                            # Temperature glide of the external fluid in the MT heat source [K]
    T4_prime = T3_prime - glide_MT                          # Outlet temperature of the external fluid in the MT heat source [K]
    p3_prime = 3e5                                          # Inlet pressure of the external fluid in the MT heat source [Pa]

    # Heat sink parameters
    external_fluid_HT = 'Water'                             # External fluid in the heat sink
    T6_prime = 120 + 273.15                                 # Inlet temperature of the external fluid in the heat sink [K]
    T5_prime = T6_prime - glide_HT                          # Outlet temperature of the external fluid in the heat sink [K]
    p5_prime = 5e5                                          # Inlet pressure of the external fluid in the heat sink [Pa]

    # Correction factors for the correlations
    corr_factors = {
        "h_single_phase": 1.0,
        "h_evaporation": 1.0,
        "h_condensation": 1.0,
        "h_supercritical": 1.0,
        "f_single_phase": 1.0,
        "f_evaporation": 1.0,
        "f_condensation": 1.0,
        "f_supercritical": 1.0
    }

    # Geometric parameters of the BPHEs
    beta = 45.0
    phi   = 1.2
    gamma = 0.55

    # Bounds for the optimization parameters
    T_6_min = T5_prime + T_pinch    # Minimum outlet temperature of the gas cooler [K]
    T_6_max = T_6_min + 10          # Maximum outlet temperature of the gas cooler [K]
    T_sup_1_min = 1                 # Minimum superheating at point 1 [K]
    T_sup_1_max = 8                 # Maximum superheating at point 1 [K]
    T_sup_3_min = 1                 # Minimum superheating at point 3 [K]
    T_sup_3_max = 20                # Maximum superheating at point 3 [K]


    ############################################################
    # Instantiate objects
    ############################################################

    # CoolProp low-level interface for all the fluids

    HEOS_type = "HEOS"  # Choose from "HEOS", "TTSE&HEOS"

    HEOS_external_fluid_LT = CoolProp.AbstractState(HEOS_type, external_fluid_LT)
    HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
    HEOS_external_fluid_HT = CoolProp.AbstractState(HEOS_type, external_fluid_HT)
    HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

    # Cycle with its fixed states and mass flow rates
    Study = Cycle("Study_Dual_Evaporator")
    Study.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
    Study.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
    Study.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
    Study.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
    Study.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
    Study.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

    # Compressors
    Study.Compressor_1 = Compressor_2_param(cycle=Study, eta_v=eta_v, eta_is_max=eta_is, fluid=working_fluid, eta_elme=eta_elme)
    Study.Compressor_2 = Compressor_2_param(cycle=Study, eta_v=eta_v, eta_is_max=eta_is, fluid=working_fluid, eta_elme=eta_elme)

    # Ratio between the two evaporators
    Study.alpha = alpha
    ratio_evaporators = alpha / (1 - alpha)


    ############################################################
    # Define the functions for the optimization
    ############################################################

    def iterative_process(p_gess, T_6_current, T_sup_current_1, T_sup_current_3): 
        p1_guess = p_gess[0] ; p3_guess = p_gess[1] ; p5_guess = p_gess[2]

        if p1_guess < 0 or p3_guess < 0 or p5_guess < 0 :
            return np.array([1e6, 1e6, 1e6])  # Return large residuals if any pressure guess is negative
        
        if p1_guess > 200e5 or p3_guess > 200e5 or p5_guess > 200e5 :
            return np.array([1e6, 1e6, 1e6])  # Return large residuals if any pressure guess is unreasonably high

        # STEP 1 : Compute the states based on the guesses values

            # Compute guessed state 1 (saturated vapor)
        HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
        Tsat_1 = HEOS_working_fluid.T()
        Study.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current_1, p = p1_guess)

            # Compute guessed state 3 (saturated vapor)
        HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
        Tsat_3 = HEOS_working_fluid.T()
        Study.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current_3, p = p3_guess)

            # Compute guessed state 6 (subcooled liquid at the GasCooler outlet)
        Study.state_6 = State(HEOS_working_fluid, T=T_6_current, p=p5_guess)

        # STEP 2 : solve the recuperator MT to get states 4, 7, 8 
        Study.Recuperator_MT = HEX_Design(states_in=[Study.state_3, Study.state_6], states_out=[None, None], 
                                        mdot=[None, None], name="Recuperator_MT", mode = "Non-Dimensional", 
                                        type = "Recuperator", epsilon=recuperator_effectiveness)
        Study.Recuperator_MT.Solve_Recuperator()
        Study.state_4 = Study.Recuperator_MT.state_out_c
        Study.state_7 = Study.Recuperator_MT.state_out_h

            # Compute guessed state 8 
        h8 = Study.state_7.h
        Study.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

        # STEP 3 : solve the recuperator LT to get states 2, 9, 10
        Study.Recuperator_LT = HEX_Design(states_in=[Study.state_1, Study.state_7], states_out=[None, None], 
                                        mdot=[None, None], name="Recuperator_LT", mode = "Non-Dimensional", 
                                        type = "Recuperator", epsilon=recuperator_effectiveness)
        Study.Recuperator_LT.Solve_Recuperator()
        Study.state_2 = Study.Recuperator_LT.state_out_c
        Study.state_9 = Study.Recuperator_LT.state_out_h

            # Compute guessed state 10
        h10 = Study.state_9.h
        Study.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

        # STEP 4 : Compute the residual for the first evaporator
        Study.Evaporator_LT = HEX_Design(states_in=[Study.state_10, Study.state_1_prime], states_out=[Study.state_1, Study.state_2_prime], 
                                    name="Evaporator_LT", mode="Non-Dimensional")
        Tpinch_real = Study.Evaporator_LT.Compute_Pinch()
        res_evap_LT = Tpinch_real - T_pinch

        # STEP 5 : Compute the residual for the second evaporator
            
            # Compute guessed state 3_comp (exit of first compressor)
        T_3_comp = Study.Compressor_1.Solve(p_ex=p3_guess, state_in=Study.state_2, mode = 'Non-Dimensional')[1]
        Study.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

        # Compute guessed state 3_evap
        r = ratio_evaporators * (Study.state_1.h - Study.state_10.h)
        h3_evap = (r*Study.state_3.h - Study.state_8.h * (Study.state_3.h - Study.state_3_comp.h)) / (r + Study.state_3_comp.h - Study.state_3.h)

        Study.state_3_evap = State(HEOS_working_fluid, h=h3_evap, p=p3_guess)

        Study.Evaporator_MT = HEX_Design(states_in=[Study.state_8, Study.state_3_prime], states_out=[Study.state_3_evap, Study.state_4_prime], 
                                        name="Evaporator_MT", mode="Non-Dimensional")
        Tpinch_real = Study.Evaporator_MT.Compute_Pinch()
        res_evap_MT = Tpinch_real - T_pinch

        # STEP 6 : Compute the residual for the GasCooler

            # Compute guessed state 5 (exit of second compressor)
        T_5 = Study.Compressor_2.Solve(p_ex=p5_guess, state_in=Study.state_4, mode = 'Non-Dimensional')[1]
        Study.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

        Study.GasCooler = HEX_Design(states_in=[Study.state_5_prime, Study.state_5], states_out=[Study.state_6_prime, Study.state_6],
                                    name="Gas Cooler", mode="Non-Dimensional")
        Tpinch_real = Study.GasCooler.Compute_Pinch()
        res_cond = Tpinch_real - T_pinch

        # STEP 7 : Assemble the residuals

        residuals = np.array([res_evap_LT, res_evap_MT, res_cond])
        return residuals


    def objective_function(optimization_vars):

        # Unpack optimization variables
        T_6_current = optimization_vars[0]
        T_sup_current_1 = optimization_vars[1]
        T_sup_current_3 = optimization_vars[2]

        # Initial guesses for the pressures
        p1_guess = 5e5 ; p3_guess = 10e5 ; p5_guess = 44e5
        p_guess = np.array([p1_guess, p3_guess, p5_guess])

        # Find the pressures that satisfy the pinch constraints
        try :
            p_solution = root(iterative_process, p_guess, args=(T_6_current, T_sup_current_1, T_sup_current_3))
            if not p_solution.success:
                if verbose:
                    print("Root finding did not converge: ", p_solution.message)
                return 1e6  # Return a large penalty if root finding fails
            p3_solution = p_solution.x[1]
            p5_solution = p_solution.x[2]
        except Exception as e:
            if verbose:
                print("An error occurred during fsolve: ", e)
            return 1e6  # Return a large penalty if fsolve fails

        # Compute the COP for the current cycle
        Delta_h_GasCooler = Study.state_5.h - Study.state_6.h
        Study.mdot_wf_top = Q / Delta_h_GasCooler
        Study.P_comp_top = Study.Compressor_2.Solve(p_ex=p5_solution, state_in=Study.state_4, mdot_wf=Study.mdot_wf_top, mode="Dimensional")[0]

        Study.mdot_wf_bottom = Study.mdot_wf_top / (1 + ratio_evaporators * (Study.state_1.h - Study.state_10.h) / (Study.state_3_evap.h - Study.state_8.h))
        Study.P_comp_bottom = Study.Compressor_1.Solve(p_ex=p3_solution, state_in=Study.state_2, mdot_wf=Study.mdot_wf_bottom, mode="Dimensional")[0]
        COP = Q / (Study.P_comp_top + Study.P_comp_bottom)

        if Study.P_comp_top < 0 or Study.P_comp_bottom < 0 :
            return 1e6  # Return a large penalty if any compressor power are negative

        # Print the current cycle performance if verbose
        if verbose:
            print(f"  - Current cycle with T6={T_6_current:.2f} K, T_sup1={T_sup_current_1:.2f} K, T_sup3={T_sup_current_3:.2f} K has COP = {COP:.4f}")

        # We want to maximize the COP, so we minimize its negative value
        return -COP    


    ############################################################
    # Optimization procedure
    ############################################################  

    # Initial guess and bounds for optimization variables
    optimization_vars_guess = np.array([(T_6_min + T_6_max) / 2, (T_sup_1_min + T_sup_1_max) / 2, (T_sup_3_min + T_sup_3_max) / 2])
    bounds = [(T_6_min, T_6_max), (T_sup_1_min, T_sup_1_max), (T_sup_3_min, T_sup_3_max)]

    result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Nelder-Mead")

    # Extract the best parameters
    T_6_best = result.x[0]
    T_sup_best_1 = result.x[1]
    T_sup_best_3 = result.x[2]
    if print_results:
        print("\nBest cycle found with parameters :")
        print(f"  - Outlet temperature of gas cooler : {T_6_best:.2f} K")
        print(f"  - Superheating at point 1 : {T_sup_best_1:.2f} K")
        print(f"  - Superheating at point 3 : {T_sup_best_3:.2f} K")

    # Recompute the best cycle states (for safety)
    p1_guess = 5e5 ; p3_guess = 10e5 ; p5_guess = 44e5
    p_guess = np.array([p1_guess, p3_guess, p5_guess])
    p_best = fsolve(iterative_process, p_guess, args=(T_6_best, T_sup_best_1, T_sup_best_3))
    p1_best = p_best[0] ; p3_best = p_best[1] ; p5_best = p_best[2]

    # Compute heat exchangers and compressor with dimensional mode
    Delta_h_GasCooler = Study.state_5.h - Study.state_6.h
    Study.mdot_wf_top = Q / Delta_h_GasCooler
    Study.P_comp_top = Study.Compressor_2.Solve(p_ex=p5_best, state_in=Study.state_4, mdot_wf=Study.mdot_wf_top, mode="Dimensional")[0]
    Study.mdot_wf_bottom = Study.mdot_wf_top / (1 + ratio_evaporators * (Study.state_1.h - Study.state_10.h) / (Study.state_3_evap.h - Study.state_8.h))
    Study.P_comp_bottom = Study.Compressor_1.Solve(p_ex=p3_best, state_in=Study.state_2, mdot_wf=Study.mdot_wf_bottom, mode="Dimensional")[0]
    Study.Evaporator_LT = HEX_Design(states_in=[Study.state_10, Study.state_1_prime], states_out=[Study.state_1, Study.state_2_prime], 
                                    mdot = [Study.mdot_wf_bottom, None], name="Evaporator_LT", beta=beta, phi=phi, gamma=gamma,
                                    mode="Dimensional", model="ACP70X", corr_factors=corr_factors)
    T_pinch_evap_LT = Study.Evaporator_LT.Compute_Pinch()
    Study.mdot_LT = Study.Evaporator_LT.mdot_h
    Study.Evaporator_MT = HEX_Design(states_in=[Study.state_8, Study.state_3_prime], states_out=[Study.state_3_evap, Study.state_4_prime],
                                    mdot = [Study.mdot_wf_top - Study.mdot_wf_bottom, None], name="Evaporator_MT", beta=beta, phi=phi, gamma=gamma,
                                    mode="Dimensional", model="ACP70X", corr_factors=corr_factors)
    T_pinch_evap_MT = Study.Evaporator_MT.Compute_Pinch()
    Study.mdot_MT = Study.Evaporator_MT.mdot_h
    Study.GasCooler = HEX_Design(states_in=[Study.state_5_prime, Study.state_5], states_out=[Study.state_6_prime, Study.state_6], 
                                mdot=[None, Study.mdot_wf_top], name="GasCooler", beta=beta, phi=phi, gamma=gamma,
                                mode="Dimensional", model="ACP70X", corr_factors=corr_factors)
    T_pinch_gas_cooler = Study.GasCooler.Compute_Pinch()
    Study.mdot_HT = Study.GasCooler.mdot_c
    Study.Recuperator_MT = HEX_Design(states_in=[Study.state_3, Study.state_6], states_out=[Study.state_4, Study.state_7], 
                                    mdot=[Study.mdot_wf_top, Study.mdot_wf_top], name="Recuperator_MT", beta=beta, phi=phi, gamma=gamma,
                                    mode="Dimensional", type="Recuperator", epsilon=recuperator_effectiveness, 
                                    model="ACK18", corr_factors=corr_factors)
    Study.Recuperator_MT.Solve_Recuperator()
    Study.Recuperator_LT = HEX_Design(states_in=[Study.state_1, Study.state_7], states_out=[Study.state_2, Study.state_9], 
                                    mdot=[Study.mdot_wf_bottom, Study.mdot_wf_bottom], name="Recuperator_LT", beta=beta, phi=phi, gamma=gamma,
                                    mode="Dimensional", type="Recuperator", epsilon=recuperator_effectiveness, 
                                    model="ACK18", corr_factors=corr_factors)
    Study.Recuperator_LT.Solve_Recuperator()

    # Compute cycle performance
    Study.COP = Study.GasCooler.Q / (Study.P_comp_top + Study.P_comp_bottom)

    # Limit the highest pressure of the cycle to 55 bars
    if warnings and (Study.state_5.p > 5.5e6) :
        raise ValueError("The highest pressure of the cycle exceeds 55 bars. Please adjust the input parameters.")

    # Raise error if pinch points are not satisfied
    if not (np.isclose(T_pinch_evap_LT, T_pinch, atol=1e-4) and np.isclose(T_pinch_evap_MT, T_pinch, atol=1e-4) and np.isclose(T_pinch_gas_cooler, T_pinch, atol=1e-4)):
        raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

    end = time()
    if print_results : print(f"\nOptimization completed in {end - start:.2f} seconds.\n")


    ############################################################
    # Plot the results
    ############################################################

    if plot_results or save_results:

        # Define the transforms 
        Study.transforms = [Transform('isobaric_mixing', '3_comp', '3_evap', None),
                           Transform('comp', '2', '3_comp', Study.Compressor_1),
                           Transform('comp', '4', '5', Study.Compressor_2),
                           Transform('hex', '5', '6', Study.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                           Transform('adex', '7', '8', None),
                           Transform('adex', '9', '10', None),
                           Transform('hex', '8', '3_evap',Study.Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                           Transform('hex', '10', '1',Study.Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'),
                           Transform('hex', '3', '4',Study.Recuperator_MT, label_in_secondary='6', label_out_secondary='7'),
                           Transform('hex', '1', '2',Study.Recuperator_LT, label_in_secondary='7', label_out_secondary='9')]

        # Plot T-s diagram with saturation curve
        Study.Ts_diagram(n=100, plot=plot_results, save=save_results, external_circuits=True)
        Study.ph_diagram(n=100, plot=plot_results, save=save_results)

        # Plot energy and exergy charts
        Study.energy_chart(plot=plot_results, save=save_results)
        Study.exergy_chart(T0=293.15, p0 = 1e5, plot=plot_results, save=save_results)

        # Plot heat exchangers diagrams
        Study.Evaporator_LT._plot(save=save_results, name_cycle=Study.name, plot=plot_results)
        Study.Evaporator_MT._plot(save=save_results, name_cycle=Study.name, plot=plot_results)
        Study.GasCooler._plot(save=save_results, name_cycle=Study.name, plot=plot_results)
        Study.Recuperator_LT._plot(save=save_results, name_cycle=Study.name, plot=plot_results)
        Study.Recuperator_MT._plot(save=save_results, name_cycle=Study.name, plot=plot_results)

    ############################################################
    # Print the results
    ############################################################

    if print_results:
        print(Study)
        print(Study.Evaporator_LT)
        print(Study.Evaporator_MT)
        print(Study.GasCooler)
        print(Study.Recuperator_LT)
        print(Study.Recuperator_MT)

    # Save prints to a text file
    if save_results:
        output_file = Path(__file__).parent.parent / "Figures" / Study.name / f"{Study.name}_results.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(str(Study) + '\n')
            f.write('\n' + str(Study.Evaporator_LT) + '\n')
            f.write('\n' + str(Study.Evaporator_MT) + '\n')
            f.write('\n' + str(Study.GasCooler) + '\n')
            f.write('\n' + str(Study.Recuperator_LT) + '\n')
            f.write('\n' + str(Study.Recuperator_MT) + '\n')

    # Prepare the results to be returned.  The COP Carnot is computed based on the method of Arpagaus et al. (2016)
    COP = Study.COP
    COP_LT_Carnot = Study.state_6_prime.T / (Study.state_6_prime.T - Study.state_1_prime.T)
    COP_MT_Carnot = Study.state_6_prime.T / (Study.state_6_prime.T - Study.state_3_prime.T)

    LM_HT = (Study.state_6_prime.T - Study.state_5_prime.T) / np.log(Study.state_6_prime.T / Study.state_5_prime.T)
    LM_MT = (Study.state_3_prime.T - Study.state_4_prime.T) / np.log(Study.state_3_prime.T / Study.state_4_prime.T)
    LM_LT = (Study.state_1_prime.T - Study.state_2_prime.T) / np.log(Study.state_1_prime.T / Study.state_2_prime.T)

    COP_LT_Lorenz = LM_HT / (LM_HT - LM_LT)
    COP_MT_Lorenz = LM_HT / (LM_HT - LM_MT)

    xsi = alpha/(1-alpha) * (COP_LT_Carnot - 1)/(COP_MT_Carnot - 1)
    COP_Carnot = 1/(1 + xsi) * COP_LT_Carnot + xsi/(1 + xsi) * COP_MT_Carnot

    xsi_Lorenz = alpha/(1-alpha) * (COP_LT_Lorenz - 1)/(COP_MT_Lorenz - 1)
    COP_Lorenz = 1/(1 + xsi_Lorenz) * COP_LT_Lorenz + xsi_Lorenz/(1 + xsi_Lorenz) * COP_MT_Lorenz

    second_law_efficiency = Study.COP / COP_Carnot
    second_law_efficiency_Lorenz = Study.COP / COP_Lorenz

    return COP, COP_Carnot, COP_Lorenz, second_law_efficiency, second_law_efficiency_Lorenz


"""
    This function is inspired from the TC1R.py script.

"""
def run_study_cycle_single_evaporator(glide_HT, T_MT, verbose=False, print_results=False, plot_results=False, save_results=False, warnings=False):

    start = time()

    ############################################################
    # Parameters
    ############################################################

    # Technological parameters
    T_pinch = 3                             # Minimum temperature difference in heat exchangers [K]
    eta_v = 1                               # Volumetric efficiency
    eta_is = 0.70                           # Isentropic efficiency of the compressor
    eta_elme = 0.95                         # Electrical-mechanical efficiency
    recuperator_effectiveness = 0.8         # Effectiveness of the recuperator

    # Cycle parameters
    working_fluid = 'R290'              # Working fluid
    Q = 25000                           # Power output at the condenser [W]

    # Heat source parameters
    external_fluid_MT = 'Water'                 # External fluid in the heat source
    T3_prime = T_MT                             # Inlet temperature of the external fluid in the heat source [K]
    glide_MT = 5                                # Temperature glide of the external fluid in the heat source [K]
    T4_prime = T3_prime - glide_MT              # Outlet temperature of the external fluid in the heat source [K]
    p3_prime = 3e5                              # Inlet pressure of the external fluid in the heat source [Pa]

    # Heat sink parameters
    external_fluid_HT = 'Water'                 # External fluid in the heat sink
    T6_prime = 120 + 273.15                     # Inlet temperature of the external fluid in the heat sink [K]
    T5_prime = T6_prime - glide_HT              # Outlet temperature of the external fluid in the heat sink [K]
    p5_prime = 5e5                              # Inlet pressure of the external fluid in the heat sink [Pa]

    # Correction factors for the correlations
    corr_factors = {
        "h_single_phase": 1.0,
        "h_evaporation": 1.0,
        "h_condensation": 1.0,
        "h_supercritical": 1.0,
        "f_single_phase": 1.0,
        "f_evaporation": 1.0,
        "f_condensation": 1.0,
        "f_supercritical": 1.0
    }

    # Geometric parameters of the BPHEs
    beta = 45.0
    phi   = 1.2
    gamma = 0.55

    # Bounds for the optimization parameters
    T_sup_min = 1                   # Minimum superheating at the exit of the evaporator [K]
    T_sup_max = 15                  # Maximum superheating at the exit of the evaporator [K]
    T_6_min = T5_prime + T_pinch    # Minimum outlet temperature of the gas cooler [K]
    T_6_max = T_6_min + 20          # Maximum outlet temperature of the gas cooler [K]


    ############################################################
    # Instantiate objects
    ############################################################

    # CoolProp low-level interface for all the fluids

    HEOS_type = "HEOS"  # Choose from "HEOS", "TTSE&HEOS"

    HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
    HEOS_external_fluid_HT = CoolProp.AbstractState(HEOS_type, external_fluid_HT)
    HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

    # Cycle with its fixed states and mass flow rates
    Study = Cycle("Study_Single_Evaporator")
    Study.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
    Study.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
    Study.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
    Study.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

    # Compressor
    Study.Compressor = Compressor_2_param(cycle=Study, eta_v=eta_v, eta_is_max=eta_is, fluid=working_fluid, eta_elme=eta_elme)


    ############################################################
    # Define the functions for the optimization
    ############################################################

    def iterative_process(args) :
            
        try :
        
            p3_guess = args[0]*1e5
            p5_guess = args[1]*1e5
            T_sup_current = args[2]
            T_6_current = args[3]

            # STEP 1 : Compute the states based on T_sup_current and T_6_current

                # Compute guessed state 3
            HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
            Tsat_3 = HEOS_working_fluid.T()
            Study.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current, p=p3_guess)

                # Compute guessed state 6
            Study.state_6 = State(HEOS_working_fluid, T=T_6_current, p=p5_guess)

            # STEP 2 : Solve the Recuperator to get states 4, 7 and 8

                # Compute guessed states 4 and 7
            Study.Recuperator = HEX_Design(states_in=[Study.state_3, Study.state_6], states_out=[None, None], mdot = [None, None], name = 'Recuperator', mode="Non-Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness)
            Study.Recuperator.Solve_Recuperator()
            Study.state_4 = Study.Recuperator.state_out_c
            Study.state_7 = Study.Recuperator.state_out_h

                # Compute guessed state 8
            h8 = Study.state_7.h
            Study.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

            # STEP 3 : Solve the Compressor to get state 5 and mass flow rate

                # Compute guessed state 5
            T_5 = Study.Compressor.Solve(p_ex=p5_guess, state_in=Study.state_4, mode="Non-Dimensional")[1]
            Study.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

            # STEP 4 : Compute the residual for the evaporator

            Study.Evaporator = HEX_Design(states_in=[Study.state_8, Study.state_3_prime], states_out=[Study.state_3, Study.state_4_prime], name="Evaporator", mode="Non-Dimensional")
            Tpinch_real = Study.Evaporator.Compute_Pinch()
            res_evap = (Tpinch_real - T_pinch)/T_pinch

            # STEP 5 : Compute the residual for the gas cooler

            Study.GasCooler = HEX_Design(states_in=[Study.state_5_prime, Study.state_5], states_out=[Study.state_6_prime, Study.state_6], name="Gas Cooler", mode="Non-Dimensional")
            Tpinch_real = Study.GasCooler.Compute_Pinch()
            res_gas_cooler = (Tpinch_real - T_pinch)/T_pinch

            # STEP 6 : Compute the COP
            Delta_h_GasCooler = Study.state_5.h - Study.state_6.h
            Study.mdot_wf_top = Q / Delta_h_GasCooler
            Study.P_comp_top = Study.Compressor.Solve(p_ex=p5_guess, state_in=Study.state_4, mdot_wf=Study.mdot_wf_top, mode="Dimensional")[0]
            COP = Q / Study.P_comp_top

            # STEP 7 : Assemble the residuals

                # We want to maximise the COP while satisfying the pinch constraints
            
            result = -COP + 100 * (res_evap**2 + res_gas_cooler**2)  # Add a penalty for the pinch constraints violations

            if verbose:
                print(f"Current cycle with p3 = {p3_guess/1e5:.2f} bar, p5 = {p5_guess/1e5:.2f} bar, T_sup = {T_sup_current:.2f} K and T_6 = {T_6_current:.2f} K has COP = {COP:.4f} and residual = {result:.4f}")
        
        except:

            if verbose:
                print(f"Current cycle with p3 = {args[0]:.2f} bar, p5 = {args[1]:.2f} bar, T_sup = {args[2]:.2f} K and T_6 = {args[3]:.2f} K has an error.")
            
            result = 1e6 # Return a large penalty if any error occurs during the computation of the cycle states or performance

        return result

    ############################################################
    # Optimization procedure
    ############################################################
            
    # Initial guess and bounds for optimization variables
    bounds = [(3, 25), 
              (42.6, 60),
              (T_sup_min, T_sup_max), 
              (T_6_min, T_6_max)]

    result_global = differential_evolution(iterative_process,
                                    bounds=bounds,
                                    strategy='best2bin',
                                    maxiter=1000,
                                    popsize=40,
                                    polish=False,
                                    disp=False,
                                    seed=42)
    
    if verbose: print(f"Global search completed with best parameters : p3 = {result_global.x[0]:.2f} bar, p5 = {result_global.x[1]:.2f} bar, T_sup = {result_global.x[2]:.2f} K and T_6 = {result_global.x[3]:.2f} K.")
    
    result = minimize(iterative_process, result_global.x, bounds=bounds, method="Nelder-Mead")

    # Extract the best parameters
    p3_best = result.x[0]*1e5
    p5_best = result.x[1]*1e5
    T_sup_best = result.x[2]
    T_6_best = result.x[3]
    if print_results:
        print("\nBest cycle found with parameters :")
        print(f"  - Pressure p3 at the evaporator outlet : {p3_best/1e5:.2f} bar")
        print(f"  - Pressure p5 at the gas cooler outlet : {p5_best/1e5:.2f} bar")
        print(f"  - Superheating at evaporator outlet : {T_sup_best:.2f} K")
        print(f"  - Outlet temperature of gas cooler : {T_6_best:.2f} K")

    # Recompute the best cycle states (for safety)
    iterative_process([p3_best/1e5, p5_best/1e5, T_sup_best, T_6_best])

    # Compute heat exchangers and compressor with dimensional mode
    Delta_h_GasCooler = Study.state_5.h - Study.state_6.h
    Study.mdot_wf_top = Q / Delta_h_GasCooler
    Study.P_comp_top = Study.Compressor.Solve(p_ex=p5_best, state_in=Study.state_4, mdot_wf=Study.mdot_wf_top, mode="Dimensional")[0]
    Study.Evaporator = HEX_Design(states_in=[Study.state_8, Study.state_3_prime], states_out=[Study.state_3, Study.state_4_prime], 
                                 mdot=[Study.mdot_wf_top, None], name="Evaporator", beta=beta, phi=phi, gamma=gamma,
                                 mode="Dimensional",  model="ACP70X", corr_factors=corr_factors)
    T_pinch_evap = Study.Evaporator.Compute_Pinch()
    Study.mdot_MT = Study.Evaporator.mdot_h
    Study.GasCooler = HEX_Design(states_in=[Study.state_5_prime, Study.state_5], states_out=[Study.state_6_prime, Study.state_6], 
                                mdot=[None, Study.mdot_wf_top], name="Gas Cooler", beta=beta, phi=phi, gamma=gamma,
                                mode="Dimensional", model="ACP70X", corr_factors=corr_factors)
    T_pinch_gas_cooler = Study.GasCooler.Compute_Pinch()
    Study.mdot_HT = Study.GasCooler.mdot_c
    Study.Recuperator = HEX_Design(states_in=[Study.state_3, Study.state_6], states_out=[Study.state_4, Study.state_7], 
                                  mdot=[Study.mdot_wf_top, Study.mdot_wf_top], name = 'Recuperator', beta=beta, phi=phi, gamma=gamma, 
                                  mode="Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness, 
                                  model="ACK18", corr_factors=corr_factors)
    Study.Recuperator.Solve_Recuperator()

    # Compute cycle performance
    Study.COP = Study.GasCooler.Q / Study.P_comp_top

    # Limit the highest pressure of the cycle to 55 bars
    if warnings and (Study.state_5.p > 5.5e6) :
        raise ValueError("The highest pressure of the cycle exceeds 55 bars. Please adjust the input parameters.")

    # Raise error if pinch points are not satisfied
    if not (np.isclose(T_pinch_evap, T_pinch, atol=1e-2) and np.isclose(T_pinch_gas_cooler, T_pinch, atol=1e-2)):
        raise ValueError(f"Pinch point constraints not satisfied in the best cycle found : T_pinch_evap = {T_pinch_evap:.4f} K, T_pinch_gas_cooler = {T_pinch_gas_cooler:.4f} K.")

    end = time()
    if print_results : print(f"\nOptimization completed in {end - start:.2f} seconds.\n")


    ############################################################
    # Plot the results
    ############################################################

    if plot_results or save_results:

        # Define the transforms 
        Study.transforms = [Transform('comp', '4', '5', Study.Compressor), 
                            Transform('hex', '5', '6', Study.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'), 
                            Transform('adex', '7', '8', None), 
                            Transform('hex', '8', '3', Study.Evaporator, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                            Transform('hex', '3', '4', Study.Recuperator, label_in_secondary='6', label_out_secondary='7')]

        # Plot T-s and p-h diagrams with saturation curve
        Study.Ts_diagram(n=100, plot=plot_results, save=save_results, external_circuits=True)
        Study.ph_diagram(n=100, plot=plot_results, save=save_results)

        # Plot energy and exergy charts
        Study.energy_chart(plot=plot_results, save=save_results)
        Study.exergy_chart(T0 = 293.15, p0 = 1e5, plot=plot_results, save=save_results)

        # Plot heat exchangers diagrams
        Study.Evaporator._plot(save=save_results, name_cycle=Study.name, plot=plot_results)
        Study.GasCooler._plot(save=save_results, name_cycle=Study.name, plot=plot_results)
        Study.Recuperator._plot(save=save_results, name_cycle=Study.name, plot=plot_results)

    ############################################################
    # Print the results
    ############################################################

    if print_results:
        print(Study)
        print(Study.Evaporator)
        print(Study.GasCooler)
        print(Study.Recuperator)

    # Save prints to a text file
    if save_results:
        output_file = Path(__file__).parent.parent / "Figures" / Study.name / f"{Study.name}_results.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(str(Study) + '\n')
            f.write('\n' + str(Study.Evaporator) + '\n')
            f.write('\n' + str(Study.GasCooler) + '\n')
            f.write('\n' + str(Study.Recuperator) + '\n')


    # Prepare the results to be returned.
    COP = Study.COP
    COP_Carnot = Study.state_6_prime.T / (Study.state_6_prime.T - Study.state_3_prime.T) 
    
    LM_HT = (Study.state_6_prime.T - Study.state_5_prime.T) / np.log(Study.state_6_prime.T / Study.state_5_prime.T)
    LM_MT = (Study.state_3_prime.T - Study.state_4_prime.T) / np.log(Study.state_3_prime.T / Study.state_4_prime.T)

    COP_Lorenz = LM_HT / (LM_HT - LM_MT)
    
    return COP, COP_Carnot, COP_Lorenz


if __name__ == "__main__":

    save_results = False

    run_single_evaporator_analysis = False
    run_dual_evaporator_analysis = False

    if run_single_evaporator_analysis:
        
        T_MT_values =  [15 + 273.15, 40 + 273.15, 60 +273.15]
        glide_values = [40, 60]

        results_no_alpha = []

        current_simulation = 1

        for T_MT in T_MT_values:
            for glide in glide_values:

                total_number_of_simulations = len(T_MT_values) * len(glide_values)
                progress_percentage = (current_simulation / total_number_of_simulations) * 100
                print(f"Progress: {progress_percentage:6.2f}%", end="\r")
                current_simulation += 1

                try:
                    cop, cop_carnot, cop_lorenz = run_study_cycle_single_evaporator(glide, T_MT, verbose=False, print_results=False, plot_results=False, save_results=False, warnings=False)
                    results_no_alpha.append({"glide_HT": glide, "T_MT": T_MT - 273.15, "COP": cop, "COP_Carnot": cop_carnot, "COP_Lorenz": cop_lorenz, "status": "success"})
                    print(f"Completed simulation for glide_HT={glide} and T_MT={T_MT - 273.15} °C: COP = {cop:.4f}")

                except Exception as exc:
                    print(f"An error occurred for glide_HT={glide} and T_MT={T_MT - 273.15} °C: {exc}")
                    results_no_alpha.append({"glide_HT": glide, "T_MT": T_MT - 273.15, "COP": None, "COP_Carnot": None, "COP_Lorenz": None, "status": f"error: {exc}"})

        # Once all the simulations are done, we can build the complete performance tables as a function of alpha

        results = []
        alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


        def update_results(COP_LT, COP_MT, COP_LT_Carnot, COP_MT_Carnot, COP_LT_Lorenz, COP_MT_Lorenz, glide_HT, T_MT, results) :

            for alpha in alpha_values:

                xsi = alpha/(1-alpha) * (COP_LT - 1)/(COP_MT - 1)
                COP = 1/(1 + xsi) * COP_LT + xsi/(1 + xsi) * COP_MT

                xsi_Carnot = alpha/(1-alpha) * (COP_LT_Carnot - 1)/(COP_MT_Carnot - 1)
                COP_Carnot = 1/(1 + xsi_Carnot) * COP_LT_Carnot + xsi_Carnot/(1 + xsi_Carnot) * COP_MT_Carnot

                xsi_Lorenz = alpha/(1-alpha) * (COP_LT_Lorenz - 1)/(COP_MT_Lorenz - 1)
                COP_Lorenz = 1/(1 + xsi_Lorenz) * COP_LT_Lorenz + xsi_Lorenz/(1 + xsi_Lorenz) * COP_MT_Lorenz

                second_law_efficiency = COP / COP_Carnot
                second_law_efficiency_Lorenz = COP / COP_Lorenz

                results.append({"alpha": alpha, "glide_HT": glide_HT, "T_MT": T_MT - 273.15, "COP": COP, "COP_Carnot": COP_Carnot, "COP_Lorenz": COP_Lorenz, "eta_2": second_law_efficiency, "eta_2_Lorenz": second_law_efficiency_Lorenz, "status": "success"})

            return results


            # CASE 1 : T_MT = 40 °C and glide_HT = 40 K
        case_1_LT = results_no_alpha[0]
        case_1_MT = results_no_alpha[2]

        COP_LT = case_1_LT["COP"]
        COP_MT = case_1_MT["COP"]

        print(f"CASE 1 : T_MT = 40 °C and glide_HT = 40 K : COP_LT = {COP_LT:.4f} and COP_MT = {COP_MT:.4f}")

        COP_LT_Carnot = case_1_LT["COP_Carnot"]
        COP_MT_Carnot = case_1_MT["COP_Carnot"]
        COP_LT_Lorenz = case_1_LT["COP_Lorenz"]
        COP_MT_Lorenz = case_1_MT["COP_Lorenz"]

        glide_HT = 40
        T_MT = 40 + 273.15

        update_results(COP_LT, COP_MT, COP_LT_Carnot, COP_MT_Carnot, COP_LT_Lorenz, COP_MT_Lorenz, glide_HT, T_MT, results)

            # CASE 2 : T_MT = 40 °C and glide_HT = 60 K
        case_2_LT = results_no_alpha[1]
        case_2_MT = results_no_alpha[3]

        COP_LT = case_2_LT["COP"]
        COP_MT = case_2_MT["COP"]

        print(f"CASE 2 : T_MT = 40 °C and glide_HT = 60 K : COP_LT = {COP_LT:.4f} and COP_MT = {COP_MT:.4f}")

        COP_LT_Carnot = case_2_LT["COP_Carnot"]
        COP_MT_Carnot = case_2_MT["COP_Carnot"]
        COP_LT_Lorenz = case_2_LT["COP_Lorenz"]
        COP_MT_Lorenz = case_2_MT["COP_Lorenz"]

        glide_HT = 60
        T_MT = 40 + 273.15

        update_results(COP_LT, COP_MT, COP_LT_Carnot, COP_MT_Carnot, COP_LT_Lorenz, COP_MT_Lorenz, glide_HT, T_MT, results)

            # CASE 3 : T_MT = 60 °C and glide_HT = 40 K
        case_3_LT = results_no_alpha[0]
        case_3_MT = results_no_alpha[4]

        COP_LT = case_3_LT["COP"]
        COP_MT = case_3_MT["COP"]

        print(f"CASE 3 : T_MT = 60 °C and glide_HT = 40 K : COP_LT = {COP_LT:.4f} and COP_MT = {COP_MT:.4f}")

        COP_LT_Carnot = case_3_LT["COP_Carnot"]
        COP_MT_Carnot = case_3_MT["COP_Carnot"]
        COP_LT_Lorenz = case_3_LT["COP_Lorenz"]
        COP_MT_Lorenz = case_3_MT["COP_Lorenz"]

        glide_HT = 40
        T_MT = 60 + 273.15

        update_results(COP_LT, COP_MT, COP_LT_Carnot, COP_MT_Carnot, COP_LT_Lorenz, COP_MT_Lorenz, glide_HT, T_MT, results)

            # CASE 4 : T_MT = 60 °C and glide_HT = 60 K
        case_4_LT = results_no_alpha[1]
        case_4_MT = results_no_alpha[5]

        COP_LT = case_4_LT["COP"]
        COP_MT = case_4_MT["COP"]

        print(f"CASE 4 : T_MT = 60 °C and glide_HT = 60 K : COP_LT = {COP_LT:.4f} and COP_MT = {COP_MT:.4f}")

        COP_LT_Carnot = case_4_LT["COP_Carnot"]
        COP_MT_Carnot = case_4_MT["COP_Carnot"]
        COP_LT_Lorenz = case_4_LT["COP_Lorenz"]
        COP_MT_Lorenz = case_4_MT["COP_Lorenz"]

        glide_HT = 60
        T_MT = 60 + 273.15

        update_results(COP_LT, COP_MT, COP_LT_Carnot, COP_MT_Carnot, COP_LT_Lorenz, COP_MT_Lorenz, glide_HT, T_MT, results)


        if save_results :
            results_df = pd.DataFrame(results)

            output_file = Path(__file__).parent.parent / "Figures" / "Case_Studies" / "Case_Study_single_results.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8", newline="") as f:
                results_df.to_csv(f, index=False)


    if run_dual_evaporator_analysis:

        alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        glide_values = [40, 60]
        T_MT_values =  [40 + 273.15, 60 +273.15] 

        results = []

        current_simulation = 1

        for T_MT in T_MT_values:
            for glide in glide_values:
                for alpha in alpha_values:

                    total_number_of_simulations = len(T_MT_values) * len(glide_values) * len(alpha_values)
                    progress_percentage = (current_simulation / total_number_of_simulations) * 100
                    print(f"Progress: {progress_percentage:6.2f}%", end="\r")
                    current_simulation += 1

                    try:
                        # We run the cycle for each combination of alpha and glide, and we store the results in a list of dictionaries
                        cop, cop_carnot, cop_lorenz, eta_2, eta_2_lorenz = run_study_cycle_dual_evaporator(alpha, glide, T_MT, verbose=False, print_results=False, plot_results=False, save_results=False, warnings=False)
                        results.append({"alpha": alpha, "glide_HT": glide, "T_MT": T_MT - 273.15, "COP": cop, "COP_Carnot": cop_carnot, "COP_Lorenz": cop_lorenz, "eta_2": eta_2, "eta_2_Lorenz": eta_2_lorenz, "status": "success"})

                    except Exception as exc:
                        # If an error occurs during the cycle simulation, we catch it and store the error message in the results list
                        print(f"An error occurred for alpha={alpha} and glide_HT={glide}: {exc}")
                        results.append({"alpha": alpha, "glide_HT": glide, "T_MT": T_MT - 273.15, "COP": None, "COP_Carnot": None, "COP_Lorenz": None, "eta_2": None, "eta_2_Lorenz": None, "status": f"error: {exc}"})

        if save_results :
            results_df = pd.DataFrame(results)

            output_file = Path(__file__).parent.parent / "Figures" / "Case_Studies" / "Case_Study_dual_results.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8", newline="") as f:
                results_df.to_csv(f, index=False)