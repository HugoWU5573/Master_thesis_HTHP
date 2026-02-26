
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
working_fluid = 'R290'          # Working fluid
Q = 30e3                        # Power output at the gas cooler [W]

# Heat source parameters
external_fluid_MT = 'Water'     # External fluid in the heat source
T3_prime = 45 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
glide_MT = 5                    # Temperature glide of the external fluid in the heat source [K]
T4_prime = T3_prime - glide_MT  # Outlet temperature of the external fluid in the heat source [K]
p3_prime = 3e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
T5_prime = 80 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_HT = 40                   # Temperature glide in the gas cooler [K]
T6_prime = T5_prime + glide_HT  # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 5e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sup_min = 2                   # Minimum superheating at the exit of the evaporator [K]
T_sup_max = 8                   # Maximum superheating at the exit of the evaporator [K]
T_6_min = T5_prime + T_pinch    # Minimum outlet temperature of the gas cooler [K]
T_6_max = 380                   # Maximum outlet temperature of the gas cooler [K]

############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids

HEOS_type = "HEOS"  # Choose from "HEOS", "TTSE&HEOS"

HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState(HEOS_type, external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

# Cycle with its fixed states and mass flow rates
TC1R = Cycle("TC1R")
TC1R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC1R.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
TC1R.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC1R.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

# Compressor
TC1R.Compressor = Compressor_2_param(cycle=TC1R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Define the functions for the optimization
############################################################

def iterative_process(p_gess, T_sup_current, T_6_current) :
    p3_guess = p_gess[0] ; p5_guess = p_gess[1]

    # STEP 1 : Compute the states based on T_sup_current and T_6_current

        # Compute guessed state 3
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    TC1R.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current, p=p3_guess)

        # Compute guessed state 6
    TC1R.state_6 = State(HEOS_working_fluid, T=T_6_current, p=p5_guess)

    # STEP 2 : Solve the Recuperator to get states 4, 7 and 8

        # Compute guessed states 4 and 7
    TC1R.Recuperator = HEX_Design(states_in=[TC1R.state_3, TC1R.state_6], states_out=[None, None], mdot = [None, None], name = 'Recuperator', mode="Non-Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness)
    TC1R.Recuperator.Solve_Recuperator()
    TC1R.state_4 = TC1R.Recuperator.state_out_c
    TC1R.state_7 = TC1R.Recuperator.state_out_h

        # Compute guessed state 8
    h8 = TC1R.state_7.h
    TC1R.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    # STEP 3 : Solve the Compressor to get state 5 and mass flow rate

        # Compute guessed state 5
    T_5 = TC1R.Compressor.Solve(p_ex=p5_guess, state_in=TC1R.state_4, mode="Non-Dimensional")[1]
    TC1R.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

    # STEP 4 : Compute the residual for the evaporator

    TC1R.Evaporator = HEX_Design(states_in=[TC1R.state_8, TC1R.state_3_prime], states_out=[TC1R.state_3, TC1R.state_4_prime], name="Evaporator", mode="Non-Dimensional")
    Tpinch_real = TC1R.Evaporator.Compute_Pinch()
    res_evap = Tpinch_real - T_pinch

    # STEP 5 : Compute the residual for the gas cooler

    TC1R.GasCooler = HEX_Design(states_in=[TC1R.state_5_prime, TC1R.state_5], states_out=[TC1R.state_6_prime, TC1R.state_6], name="Gas Cooler", mode="Non-Dimensional")
    Tpinch_real = TC1R.GasCooler.Compute_Pinch()
    res_gas_cooler = Tpinch_real - T_pinch

    # STEP 6 : Assemble the residuals

    residuals = np.array([res_evap, res_gas_cooler])
    return residuals


def objective_function(optimization_vars) :

    # Unpack optimization variables
    T_sup_current = optimization_vars[0]
    T_6_current = optimization_vars[1]

    # Initial guesses for the pressures
    p3_guess = 10e5 ; p5_guess = 45e5
    p_guess = np.array([p3_guess, p5_guess])

    # Find the pressures that satisfy the pinch constraints
    try :
        p_solution = fsolve(iterative_process, p_guess, args=(T_sup_current, T_6_current))
        p5_solution = p_solution[1]
    except :
        if verbose:
            print(f"  - fsolve failed for T_sup = {T_sup_current:.2f} K and T_6 = {T_6_current:.2f} K. Assigning a large penalty.")
        return 1e6 # Return a large penalty if fsolve fails
    
    # Compute the COP for the current cycle
    Delta_h_GasCooler = TC1R.state_5.h - TC1R.state_6.h
    TC1R.mdot_wf_top = Q / Delta_h_GasCooler
    TC1R.P_comp_top = TC1R.Compressor.Solve(p_ex=p5_solution, state_in=TC1R.state_4, mdot_wf=TC1R.mdot_wf_top, mode="Dimensional")[0]
    COP = Q / TC1R.P_comp_top

    # Print the current cycle performance if verbose
    if verbose:
        print(f"  - Current cycle with T_sup = {T_sup_current:.2f} K and T_6 = {T_6_current:.2f} K has COP = {COP:.4f}")

    # We want to maximize the COP, so we minimize its negative value
    return -COP    


############################################################
# Optimization procedure
############################################################
        
# Initial guess and bounds for optimization variables
optimization_vars_guess = [(T_sup_min + T_sup_max) / 2.0, (T_6_min + T_6_max) / 2.0]
bounds = [(T_sup_min, T_sup_max), (T_6_min, T_6_max)]

result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Powell", options={'maxiter': 20})

# Extract the best parameters
T_sup_best = result.x[0]
T_6_best = result.x[1]
print("\nBest cycle found with parameters :")
print(f"  - Superheating at evaporator outlet : {T_sup_best:.2f} K")
print(f"  - Outlet temperature of gas cooler : {T_6_best:.2f} K")

# Recompute the best cycle states (for safety)
p3_guess = 10e5 ; p5_guess = 45e5
p_guess = np.array([p3_guess, p5_guess])
p_best = fsolve(iterative_process, p_guess, args=(T_sup_best, T_6_best))
p3_best = p_best[0]
p5_best = p_best[1]

# Compute heat exchangers and compressor with dimensional mode
Delta_h_GasCooler = TC1R.state_5.h - TC1R.state_6.h
TC1R.mdot_wf_top = Q / Delta_h_GasCooler
TC1R.P_comp_top = TC1R.Compressor.Solve(p_ex=p5_best, state_in=TC1R.state_4, mdot_wf=TC1R.mdot_wf_top, mode="Dimensional")[0]
TC1R.Evaporator = HEX_Design(states_in=[TC1R.state_8, TC1R.state_3_prime], states_out=[TC1R.state_3, TC1R.state_4_prime], mdot=[TC1R.mdot_wf_top, None], name="Evaporator", mode="Dimensional",  model="ACH30EQ")
T_pinch_evap = TC1R.Evaporator.Compute_Pinch()
TC1R.mdot_MT = TC1R.Evaporator.mdot_h
TC1R.GasCooler = HEX_Design(states_in=[TC1R.state_5_prime, TC1R.state_5], states_out=[TC1R.state_6_prime, TC1R.state_6], mdot=[None, TC1R.mdot_wf_top], name="Gas Cooler", mode="Dimensional", model="ACH65")
T_pinch_gas_cooler = TC1R.GasCooler.Compute_Pinch()
TC1R.mdot_HT = TC1R.GasCooler.mdot_c
TC1R.Recuperator = HEX_Design(states_in=[TC1R.state_3, TC1R.state_6], states_out=[TC1R.state_4, TC1R.state_7], mdot=[TC1R.mdot_wf_top, TC1R.mdot_wf_top], name = 'Recuperator', mode="Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness, model="ACK16")
TC1R.Recuperator.Solve_Recuperator()

# Compute cycle performance
TC1R.COP = TC1R.GasCooler.Q / TC1R.P_comp_top

# Limit the highest pressure of the cycle to 55 bars
if TC1R.state_5.p > 5.5e6 :
    raise ValueError("The highest pressure of the cycle exceeds 55 bars. Please adjust the input parameters.")

# Raise error if pinch points are not satisfied
if not (np.isclose(T_pinch_evap, T_pinch, atol=1e-4) and np.isclose(T_pinch_gas_cooler, T_pinch, atol=1e-4)):
    raise ValueError(f"Pinch point constraints not satisfied in the best cycle found : T_pinch_evap = {T_pinch_evap:.4f} K, T_pinch_gas_cooler = {T_pinch_gas_cooler:.4f} K.")

end = time()
print(f"\nOptimization completed in {end - start:.2f} seconds.\n")


############################################################
# Plot the results
############################################################

full_details = False

if full_details:

    # Define the transforms 
    TC1R.transforms = [Transform('comp', '4', '5', TC1R.Compressor), 
                    Transform('hex', '5', '6',TC1R.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'), 
                    Transform('adex', '7', '8', None), 
                    Transform('hex', '8', '3', TC1R.Evaporator, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                    Transform('hex', '3', '4', TC1R.Recuperator, label_in_secondary='6', label_out_secondary='7')]

    # Plot T-s and p-h diagrams with saturation curve
    TC1R.Ts_diagram(n=100, plot=True)
    TC1R.ph_diagram(n=100, plot=True)

    # Plot energy and exergy charts
    TC1R.energy_chart(plot=True)
    TC1R.exergy_chart(T0 = 293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    TC1R.Evaporator._plot(save=True, name_cycle=TC1R.name, plot=True)
    TC1R.GasCooler._plot(save=True, name_cycle=TC1R.name, plot=True)
    TC1R.Recuperator._plot(save=True, name_cycle=TC1R.name, plot=True)


############################################################
# Print the results
############################################################

print(TC1R)

if full_details:
    TC1R.Evaporator.Compute_Area(plot=True, save=True, name_cycle=TC1R.name)
    TC1R.GasCooler.Compute_Area(plot=True, save=True, name_cycle=TC1R.name)
    TC1R.Recuperator.Compute_Area(plot=True, save=True, name_cycle=TC1R.name)
    print(TC1R.Evaporator)
    print(TC1R.GasCooler)
    print(TC1R.Recuperator)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / TC1R.name / f"{TC1R.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(TC1R) + '\n')
        f.write('\n' + str(TC1R.Evaporator) + '\n')
        f.write('\n' + str(TC1R.GasCooler) + '\n')
        f.write('\n' + str(TC1R.Recuperator) + '\n')