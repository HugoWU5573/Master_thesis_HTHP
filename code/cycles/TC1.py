
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
T_pinch = 3                     # Minimum temperature difference in heat exchangers [K]
eta_v = 1                       # Volumetric efficiency
eta_is_max = 0.7                # Maximum isentropic efficiency
eta_elme = 0.95                 # Electrical-mechanical efficiency

# Cycle parameters
working_fluid = 'R290'          # Working fluid
Q = 30e3                        # Power output at the gas cooler [W]

# Heat source parameters
external_fluid_MT = 'Water'     # External fluid in the heat source
T3_prime = 45 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
glide_MT = 5                    # Temperature glide of the external fluid in the heat source [K]
T4_prime = T3_prime - glide_MT  # Outlet temperature of the external fluid in the heat source [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
T5_prime = 60 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_HT = 55                   # Temperature glide in the gas cooler [K]
T6_prime = T5_prime + glide_HT  # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sup_min = 2                   # Minimum superheating at the compressor inlet [K]
T_sup_max = 7                   # Maximum superheating at the compressor inlet [K]
T_7_min = T5_prime + T_pinch    # Minimum outlet temperature of the gas cooler [K]
T_7_max = 338                   # Maximum outlet temperature of the gas cooler [K]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids

HEOS_type = "TTSE&HEOS"  # Choose from "HEOS", "TTSE&HEOS"

HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState(HEOS_type, external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

# Cycle with its fixed states and mass flow rates
TC1 = Cycle("TC1")
TC1.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC1.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
TC1.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC1.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

# Compressor
TC1.Compressor = Compressor_2_param(cycle=TC1, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Define the functions for the optimization
############################################################

def iterative_process(p_gess, T_sup_current, T_7_current) :
    p3_guess = p_gess[0] ; p5_guess = p_gess[1]

    # STEP 1 : Compute the states based on T_sup_current and T_7_current

        # Compute guessed state 3
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    TC1.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current, p=p3_guess)

        # Compute guessed state 7
    TC1.state_7 = State(HEOS_working_fluid, T=T_7_current, p=p5_guess)

        # Compute guessed state 8
    h8 = TC1.state_7.h
    TC1.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    # STEP 2 : Solve the Compressor to get state 5 and mass flow rate

        # Compute guessed state 5
    T_5 = TC1.Compressor.Solve(p_ex=p5_guess, state_in=TC1.state_3, mode="Non-Dimensional")[1]
    TC1.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

    # STEP 3 : Compute the residual for the evaporator

    TC1.Evaporator = HEX_Design(states_in=[TC1.state_8, TC1.state_3_prime], states_out=[TC1.state_3, TC1.state_4_prime], name="Evaporator", mode="Non-Dimensional")
    Tpinch_real = TC1.Evaporator.Compute_Pinch()
    res_evap = Tpinch_real - T_pinch

    # STEP 4 : Compute the residual for the gas cooler

    TC1.GasCooler = HEX_Design(states_in=[TC1.state_5_prime, TC1.state_5], states_out=[TC1.state_6_prime, TC1.state_7], name="Gas Cooler", mode="Non-Dimensional")
    Tpinch_real = TC1.GasCooler.Compute_Pinch()
    res_gas_cooler = Tpinch_real - T_pinch

    # STEP 5 : Assemble the residuals

    residuals = np.array([res_evap, res_gas_cooler])
    return residuals


def objective_function(optimization_vars) :

    # Unpack optimization variables
    T_sup_current = optimization_vars[0]
    T_7_current = optimization_vars[1]

    # Initial guesses for pressures
    p3_guess = 10e5 ; p5_guess = 45e5
    p_guess = np.array([p3_guess, p5_guess])

    # Find the pressures that satisfy the pinch constraints
    try :
        p_solution = fsolve(iterative_process, p_guess, args=(T_sup_current, T_7_current))
        p5_solution = p_solution[1]
    except :
        return 1e6  # Return a large penalty value if fsolve fails
    
    # Compute the COP for the current cycle
    Delta_h_Condenser = TC1.state_5.h - TC1.state_7.h
    TC1.mdot_wf_top = Q / Delta_h_Condenser
    TC1.P_comp_top = TC1.Compressor.Solve(p_ex=p5_solution, state_in=TC1.state_3, mdot_wf=TC1.mdot_wf_top, mode="Dimensional")[0]
    COP = Q / TC1.P_comp_top

    # Print the current cycle performance if verbose
    if verbose:
        print(f"  - Current cycle with T_sup = {T_sup_current:.2f} K and T_7 = {T_7_current:.2f} K has COP = {COP:.4f}")

    # We want to maximize the COP, so we minimize its negative value
    return -COP


############################################################
# Optimization procedure
############################################################

# Initial guess and bounds for optimization variables
optimization_vars_guess = [(T_sup_min + T_sup_max) / 2.0, (T_7_min + T_7_max) / 2.0]
bounds = [(T_sup_min, T_sup_max), (T_7_min, T_7_max)]

result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Powell", options={'maxiter': 20})

# Extract the best parameters
T_sup_best = result.x[0]
T_7_best = result.x[1]
print("\nBest cycle found with parameters :")
print(f"  - Superheating at compressor inlet : {T_sup_best:.2f} K")
print(f"  - Outlet temperature of gas cooler : {T_7_best:.2f} K")

# Recompute the best cycle states (for safety)
p3_guess = 10e5 ; p5_guess = 45e5
p_guess = np.array([p3_guess, p5_guess])
p_best = fsolve(iterative_process, p_guess, args=(T_sup_best, T_7_best))
p3_best = p_best[0]
p5_best = p_best[1]

# Compute heat exchangers and compressor with dimensional mode
Delta_h_Condenser = TC1.state_5.h - TC1.state_7.h
TC1.mdot_wf_top = Q / Delta_h_Condenser
TC1.P_comp_top = TC1.Compressor.Solve(p_ex=p5_best, state_in=TC1.state_3, mdot_wf=TC1.mdot_wf_top, mode="Dimensional")[0]
TC1.Evaporator = HEX_Design(states_in=[TC1.state_8, TC1.state_3_prime], states_out=[TC1.state_3, TC1.state_4_prime], mdot=[TC1.mdot_wf_top, None], name="Evaporator", mode="Dimensional")
T_pinch_evap = TC1.Evaporator.Compute_Pinch()
TC1.mdot_MT = TC1.Evaporator.mdot_h
TC1.GasCooler = HEX_Design(states_in=[TC1.state_5_prime, TC1.state_5], states_out=[TC1.state_6_prime, TC1.state_7], mdot=[None, TC1.mdot_wf_top], name="Gas Cooler", mode="Dimensional")
T_pinch_gas_cooler = TC1.GasCooler.Compute_Pinch()
TC1.mdot_HT = TC1.GasCooler.mdot_c

# Compute cycle performance
TC1.COP = TC1.GasCooler.Q / TC1.P_comp_top

# Limit the highest pressure of the cycle to 50 bars
if TC1.state_5.p > 5e6 :
    raise ValueError("The highest pressure of the cycle exceeds 50 bars. Please adjust the input parameters.")

# Raise error if pinch points are not satisfied
if not (np.isclose(T_pinch_evap, T_pinch, atol=1e-4) and np.isclose(T_pinch_gas_cooler, T_pinch, atol=1e-4)):
    raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

end = time()
print(f"\nOptimization completed in {end - start:.2f} seconds.\n")


############################################################
# Plot the results
############################################################

full_details = False

if full_details:

    # Define the transforms 
    TC1.transforms = [Transform('comp', '3', '5', TC1.Compressor), 
                    Transform('hex', '5', '7',TC1.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'), 
                    Transform('adex', '7', '8', None), 
                    Transform('hex', '8', '3', TC1.Evaporator, label_in_secondary='3_prime', label_out_secondary='4_prime')]

    # Plot T-s and p-h diagrams with saturation curve
    TC1.Ts_diagram(n=100, plot=True)
    TC1.ph_diagram(n=100, plot=True)

    # Plot energy and exergy charts
    TC1.energy_chart(plot=True)
    TC1.exergy_chart(T0 = 293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    TC1.Evaporator._plot(save=True, name_cycle=TC1.name, plot=True)
    TC1.GasCooler._plot(save=True, name_cycle=TC1.name, plot=True)


############################################################
# Print the results
############################################################

print(TC1)

if full_details:
    TC1.Evaporator.Compute_Area()
    TC1.GasCooler.Compute_Area()
    print(TC1.Evaporator)
    print(TC1.GasCooler)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / TC1.name / f"{TC1.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(TC1) + '\n')
        f.write('\n' + str(TC1.Evaporator) + '\n')
        f.write('\n' + str(TC1.GasCooler) + '\n')