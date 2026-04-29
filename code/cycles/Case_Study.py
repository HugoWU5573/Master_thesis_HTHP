
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
from scipy.optimize import fsolve, minimize, root
from time import time
import pandas as pd

def run_study_cycle(alpha, glide_HT, verbose=False, print_results=False, plot_results=False, save_results=False, warnings=False):

    start = time()

    ############################################################
    # Parameters
    ############################################################

    ## REMARK : Some parameters are fixed while others are extracted from the sample (sensitivity analysis)

    # Technological parameters
    T_pinch = 3                             # Minimum temperature difference in heat exchangers [K]
    eta_v = 1                               # Volumetric efficiency
    eta_is = 0.70                           # Isentropic efficiency of the compressor (can be varied in the sensitivity analysis)
    eta_elme = 0.95                         # Electrical-mechanical efficiency
    recuperator_effectiveness = 0.8         # Effectiveness of the recuperator

    # Cycle parameters
    working_fluid = 'R290'              # Working fluid
    Q = 25000                           # Power output at the condenser [W] (can be varied in the sensitivity analysis)

    # Heat sources parameters

        # 1. LT source
    external_fluid_LT = 'Water'                             # External fluid in the LT heat source
    T1_prime = 15 + 273.75                                  # Inlet temperature of the external fluid in the LT heat source [K]
    glide_LT = 5                                            # Temperature glide of the external fluid in the LT heat source [K]
    T2_prime = T1_prime - glide_LT                          # Outlet temperature of the external fluid in the LT heat source [K]
    p1_prime = 3e5                                          # Inlet pressure of the external fluid in the LT heat source [Pa]

        # 2. MT source
    external_fluid_MT = 'Water'                             # External fluid in the MT heat source
    T3_prime = 60 + 273.15                                  # Inlet temperature of the external fluid in the MT heat source [K]
    glide_MT = 5                                            # Temperature glide of the external fluid in the MT heat source [K]
    T4_prime = T3_prime - glide_MT                          # Outlet temperature of the external fluid in the MT heat source [K]
    p3_prime = 3e5                                          # Inlet pressure of the external fluid in the MT heat source [Pa]

    # Heat sink parameters
    external_fluid_HT = 'Water'                             # External fluid in the heat sink
    T6_prime = 120 + 273.15                                 # Inlet temperature of the external fluid in the heat sink [K]
    T5_prime = T6_prime - glide_HT                          # Outlet temperature of the external fluid in the heat sink [K]
    p5_prime = 5e5                                          # Inlet pressure of the external fluid in the heat sink [Pa]

    # Correction factors for the correlations (can be varied in the sensitivity analysis)
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

    # Geometric parameters of the BPHEs (can be varied in the sensitivity analysis)
    beta = 45.0
    phi   = 1.2
    gamma = 0.55

    # Bounds for the optimization parameters
    T_6_min = T5_prime + T_pinch    # Minimum outlet temperature of the gas cooler [K]
    T_6_max = T_6_min + 10          # Maximum outlet temperature of the gas cooler [K]
    T_sup_1_min = 1                 # Minimum superheating at point 1 [K]
    T_sup_1_max = 8                 # Maximum superheating at point 1 [K]
    T_sup_3_min = 3                 # Minimum superheating at point 3 [K]
    T_sup_3_max = 12                # Maximum superheating at point 3 [K]


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
    Study = Cycle("Study")
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

    result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Powell", options={'maxiter': 20})

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

    # Compute the HEX areas and plot the design criteria
    Study.Evaporator_LT.Compute_Area(plot=plot_results, save=save_results, name_cycle=Study.name)
    Study.Evaporator_MT.Compute_Area(plot=plot_results, save=save_results, name_cycle=Study.name)
    Study.GasCooler.Compute_Area(plot=plot_results, save=save_results, name_cycle=Study.name)
    Study.Recuperator_LT.Compute_Area(plot=plot_results, save=save_results, name_cycle=Study.name)
    Study.Recuperator_MT.Compute_Area(plot=plot_results, save=save_results, name_cycle=Study.name)

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


if __name__ == "__main__":

    save_results = False

    alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    glide_values = [40, 60]

    results = []

    for glide in glide_values:
        for alpha in alpha_values:

            print(f"\nRunning cycle for alpha={alpha} and glide_HT={glide} K...")

            try:
                # We run the cycle for each combination of alpha and glide, and we store the results in a list of dictionaries
                cop, cop_carnot, cop_lorenz, eta_2, eta_2_lorenz = run_study_cycle(alpha, glide, verbose=False, print_results=False, plot_results=False, save_results=False, warnings=False)
                results.append({"alpha": alpha, "glide_HT": glide, "COP": cop, "COP_Carnot": cop_carnot, "COP_Lorenz": cop_lorenz, "eta_2": eta_2, "eta_2_Lorenz": eta_2_lorenz, "status": "success"})

            except Exception as exc:
                # If an error occurs during the cycle simulation, we catch it and store the error message in the results list
                print(f"An error occurred for alpha={alpha} and glide_HT={glide}: {exc}")
                results.append({"alpha": alpha, "glide_HT": glide, "COP": None, "COP_Carnot": None, "COP_Lorenz": None, "eta_2": None, "eta_2_Lorenz": None, "status": f"error: {exc}"})

    if save_results :
        results_df = pd.DataFrame(results)

        output_file = Path(__file__).parent.parent / "Figures" / "Case_Studies" / "Case_Study_all_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            results_df.to_csv(f, index=False)
