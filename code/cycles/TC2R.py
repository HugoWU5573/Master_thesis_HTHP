
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
from scipy.optimize import fsolve, minimize
from time import time

start = time()

# Set verbosity
verbose = False

############################################################
# Parameters
############################################################

# Technological parameters
T_pinch = 3                       # Minimum temperature difference in heat exchangers [K]
eta_v = 1                         # Volumetric efficiency
eta_is_max = 0.7                  # Maximum isentropic efficiency
eta_elme = 0.95                   # Electrical-mechanical efficiency
recuperator_effectiveness = 0.8   # Effectiveness of the recuperator

# Cycle parameters
working_fluid = 'R290'            # Working fluid
Q = 30e3                          # Power output at the Gas Cooler [W]
beta = 0.5                        # beta = Q_evap_MT / (Q_evap_LT + Q_evap_MT) [-]

# Heat sources parameters

    # 1. LT source
external_fluid_LT = 'Water'     # External fluid in the heat source
T1_prime = 15 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
glide_LT = 5                    # Temperature glide of the external fluid in the heat source [K]
T2_prime = T1_prime - glide_LT  # Outlet temperature of the external fluid in the heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

    # 2. MT source
external_fluid_MT = 'Water'     # External fluid in the heat sink
T3_prime = 45 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_MT = 5                    # Temperature glide of the external fluid in the heat sink [K]
T4_prime = T3_prime - glide_MT  # Outlet temperature of the external fluid in the heat sink [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
T5_prime = 80 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_HT = 40                   # Temperature glide in the gas cooler [K]
T6_prime = T5_prime + glide_HT  # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 5e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_6_min = T5_prime + T_pinch    # Minimum outlet temperature of the gas cooler [K]
T_6_max = 380                   # Maximum outlet temperature of the gas cooler [K]
T_sup_1_min = 1                 # Minimum superheating at point 1 [K]
T_sup_1_max = 8                 # Maximum superheating at point 1 [K]
T_sup_3_min = 3                 # Minimum superheating at point 3 [K]
T_sup_3_max = 12                 # Maximum superheating at point 3 [K]


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
TC2R = Cycle("TC2R")
TC2R.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
TC2R.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
TC2R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC2R.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
TC2R.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC2R.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

# Compressors
TC2R.Compressor_1 = Compressor_2_param(cycle=TC2R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)
TC2R.Compressor_2 = Compressor_2_param(cycle=TC2R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)

# Ratio between the two evaporators
TC2R.beta = beta
ratio_evaporators = beta / (1 - beta)


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
    TC2R.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current_1, p = p1_guess)

        # Compute guessed state 3 (saturated vapor)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    TC2R.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current_3, p = p3_guess)

        # Compute guessed state 6 (subcooled liquid at the GasCooler outlet)
    TC2R.state_6 = State(HEOS_working_fluid, T=T_6_current, p=p5_guess)

    # STEP 2 : solve the recuperator MT to get states 4, 7, 8 
    TC2R.Recuperator_2 = HEX_Design(states_in=[TC2R.state_3, TC2R.state_6], states_out=[None, None], 
                                    mdot=[None, None], name="Recuperator_1", mode = "Non-Dimensional", 
                                    type = "Recuperator", epsilon=recuperator_effectiveness)
    TC2R.Recuperator_2.Solve_Recuperator()
    TC2R.state_4 = TC2R.Recuperator_2.state_out_c
    TC2R.state_7 = TC2R.Recuperator_2.state_out_h

        # Compute guessed state 8 
    h8 = TC2R.state_7.h
    TC2R.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    # STEP 3 : solve the recuperator LT to get states 2, 9, 10
    
    TC2R.Recuperator_1 = HEX_Design(states_in=[TC2R.state_1, TC2R.state_7], states_out=[None, None], 
                                    mdot=[None, None], name="Recuperator_2", mode = "Non-Dimensional", 
                                    type = "Recuperator", epsilon=recuperator_effectiveness)
    TC2R.Recuperator_1.Solve_Recuperator()
    TC2R.state_2 = TC2R.Recuperator_1.state_out_c
    TC2R.state_9 = TC2R.Recuperator_1.state_out_h

        # Compute guessed state 10
    h10 = TC2R.state_9.h
    TC2R.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

    # STEP 4 : Compute the residual for the first evaporator
    TC2R.Evaporator_LT = HEX_Design(states_in=[TC2R.state_10, TC2R.state_1_prime], states_out=[TC2R.state_1, TC2R.state_2_prime], 
                                 name="Evaporator_LT", mode="Non-Dimensional")
    Tpinch_real = TC2R.Evaporator_LT.Compute_Pinch()
    res_evap_LT = Tpinch_real - T_pinch

    # STEP 5 : Compute the residual for the second evaporator
        
        # Compute guessed state 3_comp (exit of first compressor)
    T_3_comp = TC2R.Compressor_1.Solve(p_ex=p3_guess, state_in=TC2R.state_2, mode = 'Non-Dimensional')[1]
    TC2R.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

    # Compute guessed state 3_evap
    r = ratio_evaporators * (TC2R.state_1.h - TC2R.state_10.h)
    h3_evap = (r*TC2R.state_3.h - TC2R.state_8.h * (TC2R.state_3.h - TC2R.state_3_comp.h)) / (r + TC2R.state_3_comp.h - TC2R.state_3.h)

    TC2R.state_3_evap = State(HEOS_working_fluid, h=h3_evap, p=p3_guess)

    TC2R.Evaporator_MT = HEX_Design(states_in=[TC2R.state_8, TC2R.state_3_prime], states_out=[TC2R.state_3_evap, TC2R.state_4_prime], 
                                    name="Evaporator_MT", mode="Non-Dimensional")
    Tpinch_real = TC2R.Evaporator_MT.Compute_Pinch()
    res_evap_MT = Tpinch_real - T_pinch

    # STEP 6 : Compute the residual for the GasCooler

        # Compute guessed state 5 (exit of second compressor)
    T_5 = TC2R.Compressor_2.Solve(p_ex=p5_guess, state_in=TC2R.state_4, mode = 'Non-Dimensional')[1]
    TC2R.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

    TC2R.GasCooler = HEX_Design(states_in=[TC2R.state_5_prime, TC2R.state_5], states_out=[TC2R.state_6_prime, TC2R.state_6],
                                 name="Gas Cooler", mode="Non-Dimensional")
    Tpinch_real = TC2R.GasCooler.Compute_Pinch()
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
    p1_guess = 5e5 ; p3_guess = 10e5 ; p5_guess = 42e5
    p_guess = np.array([p1_guess, p3_guess, p5_guess])

    # Find the pressures that satisfy the pinch constraints
    try :
        p_solution = fsolve(iterative_process, p_guess, args=(T_6_current, T_sup_current_1, T_sup_current_3))
        p3_solution = p_solution[1]
        p5_solution = p_solution[2]
    except :
        if verbose:
            print("Fsolve failed for the current set of optimization variables.")
        return 1e6  # Return a large penalty if fsolve fails

    # Compute the COP for the current cycle
    Delta_h_GasCooler = TC2R.state_5.h - TC2R.state_6.h
    TC2R.mdot_wf_top = Q / Delta_h_GasCooler
    TC2R.P_comp_top = TC2R.Compressor_2.Solve(p_ex=p5_solution, state_in=TC2R.state_4, mdot_wf=TC2R.mdot_wf_top, mode="Dimensional")[0]

    TC2R.mdot_wf_bottom = TC2R.mdot_wf_top / (1 + ratio_evaporators * (TC2R.state_1.h - TC2R.state_10.h) / (TC2R.state_3_evap.h - TC2R.state_8.h))
    TC2R.P_comp_bottom = TC2R.Compressor_1.Solve(p_ex=p3_solution, state_in=TC2R.state_2, mdot_wf=TC2R.mdot_wf_bottom, mode="Dimensional")[0]
    COP = Q / (TC2R.P_comp_top + TC2R.P_comp_bottom)

    if TC2R.P_comp_top < 0 or TC2R.P_comp_bottom < 0 :
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
print("\nBest cycle found with parameters :")
print(f"  - Outlet temperature of gas cooler : {T_6_best:.2f} K")
print(f"  - Superheating at point 1 : {T_sup_best_1:.2f} K")
print(f"  - Superheating at point 3 : {T_sup_best_3:.2f} K")

# Recompute the best cycle states (for safety)
p1_guess = 5e5 ; p3_guess = 10e5 ; p5_guess = 42e5
p_guess = np.array([p1_guess, p3_guess, p5_guess])
p_best = fsolve(iterative_process, p_guess, args=(T_6_best, T_sup_best_1, T_sup_best_3))
p1_best = p_best[0]
p3_best = p_best[1]
p5_best = p_best[2]

# Compute heat exchangers and compressor with dimensional mode
Delta_h_GasCooler = TC2R.state_5.h - TC2R.state_6.h
TC2R.mdot_wf_top = Q / Delta_h_GasCooler
TC2R.P_comp_top = TC2R.Compressor_2.Solve(p_ex=p5_best, state_in=TC2R.state_4, mdot_wf=TC2R.mdot_wf_top, mode="Dimensional")[0]
TC2R.mdot_wf_bottom = TC2R.mdot_wf_top / (1 + ratio_evaporators * (TC2R.state_1.h - TC2R.state_10.h) / (TC2R.state_3_evap.h - TC2R.state_8.h))
TC2R.P_comp_bottom = TC2R.Compressor_1.Solve(p_ex=p3_best, state_in=TC2R.state_2, mdot_wf=TC2R.mdot_wf_bottom, mode="Dimensional")[0]
TC2R.Evaporator_LT = HEX_Design(states_in=[TC2R.state_10, TC2R.state_1_prime], states_out=[TC2R.state_1, TC2R.state_2_prime], mdot = [TC2R.mdot_wf_bottom, None], name="Evaporator_LT", mode="Dimensional", model="ACH30EQ")
T_pinch_evap_LT = TC2R.Evaporator_LT.Compute_Pinch()
TC2R.mdot_LT = TC2R.Evaporator_LT.mdot_h
TC2R.Evaporator_MT = HEX_Design(states_in=[TC2R.state_8, TC2R.state_3_prime], states_out=[TC2R.state_3_evap, TC2R.state_4_prime],mdot = [TC2R.mdot_wf_top - TC2R.mdot_wf_bottom, None], name="Evaporator_MT", mode="Dimensional", model="ACH30EQ")
T_pinch_evap_MT = TC2R.Evaporator_MT.Compute_Pinch()
TC2R.mdot_MT = TC2R.Evaporator_MT.mdot_h
TC2R.GasCooler = HEX_Design(states_in=[TC2R.state_5_prime, TC2R.state_5], states_out=[TC2R.state_6_prime, TC2R.state_6], mdot=[None, TC2R.mdot_wf_top], name="GasCooler", mode="Dimensional", model="ACH65")
T_pinch_gas_cooler = TC2R.GasCooler.Compute_Pinch()
TC2R.mdot_HT = TC2R.GasCooler.mdot_c
TC2R.Recuperator_2 = HEX_Design(states_in=[TC2R.state_3, TC2R.state_6], states_out=[TC2R.state_4, TC2R.state_7], mdot=[TC2R.mdot_wf_top, TC2R.mdot_wf_top], name="Recuperator_1", mode="Dimensional", type="Recuperator", epsilon=recuperator_effectiveness, model="ACK16")
TC2R.Recuperator_2.Solve_Recuperator()
TC2R.Recuperator_1 = HEX_Design(states_in=[TC2R.state_1, TC2R.state_7], states_out=[TC2R.state_2, TC2R.state_9], mdot=[TC2R.mdot_wf_bottom, TC2R.mdot_wf_bottom], name="Recuperator_2", mode="Dimensional", type="Recuperator", epsilon=recuperator_effectiveness, model="ACK16")
TC2R.Recuperator_1.Solve_Recuperator()

# Compute cycle performance
TC2R.COP = TC2R.GasCooler.Q / (TC2R.P_comp_top + TC2R.P_comp_bottom)

# Limit the highest pressure of the cycle to 55 bars
if TC2R.state_5.p > 5.5e6 :
    raise ValueError("The highest pressure of the cycle exceeds 55 bars. Please adjust the input parameters.")

# Raise error if pinch points are not satisfied
if not (np.isclose(T_pinch_evap_LT, T_pinch, atol=1e-4) and np.isclose(T_pinch_evap_MT, T_pinch, atol=1e-4) and np.isclose(T_pinch_gas_cooler, T_pinch, atol=1e-4)):
    raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

end = time()
print(f"\nOptimization completed in {end - start:.2f} seconds.\n")


############################################################
# Plot the results
############################################################

full_details = False

if full_details:

    # Define the transforms 
    TC2R.transforms = [Transform('isobaric_mixing', '3_comp', '3_evap', None),
                    Transform('comp', '2', '3_comp', TC2R.Compressor_1),
                    Transform('comp', '4', '5', TC2R.Compressor_2),
                    Transform('hex', '5', '6', TC2R.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                    Transform('adex', '7', '8', None),
                    Transform('adex', '9', '10', None),
                    Transform('hex', '8', '3_evap',TC2R.Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                    Transform('hex', '10', '1',TC2R.Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'),
                    Transform('hex', '3', '4',TC2R.Recuperator_2, label_in_secondary='6', label_out_secondary='7'),
                    Transform('hex', '1', '2',TC2R.Recuperator_1, label_in_secondary='7', label_out_secondary='9')]

    # Plot T-s diagram with saturation curve
    TC2R.Ts_diagram(n=100, plot=True)
    TC2R.ph_diagram(n=100, plot=True)

    # Plot energy and exergy charts
    TC2R.energy_chart(plot=True)
    TC2R.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    TC2R.Evaporator_LT._plot(save=True, name_cycle=TC2R.name, plot=True)
    TC2R.Evaporator_MT._plot(save=True, name_cycle=TC2R.name, plot=True)
    TC2R.GasCooler._plot(save=True, name_cycle=TC2R.name, plot=True)
    TC2R.Recuperator_1._plot(save=True, name_cycle=TC2R.name, plot=True)
    TC2R.Recuperator_2._plot(save=True, name_cycle=TC2R.name, plot=True)


############################################################
# Print the results
############################################################

print(TC2R)

if full_details:
    TC2R.Evaporator_LT.Compute_Area()
    TC2R.Evaporator_MT.Compute_Area()
    TC2R.GasCooler.Compute_Area()
    TC2R.Recuperator_1.Compute_Area()
    TC2R.Recuperator_2.Compute_Area()
    print(TC2R.Evaporator_LT)
    print(TC2R.Evaporator_MT)
    print(TC2R.GasCooler)
    print(TC2R.Recuperator_1)
    print(TC2R.Recuperator_2)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / TC2R.name / f"{TC2R.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(TC2R) + '\n')
        f.write('\n' + str(TC2R.Evaporator_LT) + '\n')
        f.write('\n' + str(TC2R.Evaporator_MT) + '\n')
        f.write('\n' + str(TC2R.GasCooler) + '\n')
        f.write('\n' + str(TC2R.Recuperator_1) + '\n')
        f.write('\n' + str(TC2R.Recuperator_2) + '\n')