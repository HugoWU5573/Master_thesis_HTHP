
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
Q = 30e3                        # Power output at the condenser [W]

# Heat source parameters
external_fluid_LT = 'Water'     # External fluid in the heat source
T1_prime = 15 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
glide_LT = 5                    # Temperature glide of the external fluid in the heat source [K]
T2_prime = T1_prime - glide_LT  # Outlet temperature of the external fluid in the heat source [K]
p1_prime = 3e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_MT = 'Water'     # External fluid in the heat sink
T4_prime = 40 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_MT = 5                    # Temperature glide of the external fluid in the heat sink [K]
T3_prime = T4_prime + glide_MT  # Outlet temperature of the external fluid in the heat sink [K]
p4_prime = 3e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sub_min = 1                   # Minimum subcooling at the condenser outlet [K]
T_sub_max = 8                   # Maximum subcooling at the condenser outlet [K]
T_sup_min = 1                   # Minimum superheating at the compressor inlet [K]
T_sup_max = 8                   # Maximum superheating at the compressor inlet [K]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids

HEOS_type = "HEOS"  # Choose from "HEOS", "TTSE&HEOS"

HEOS_external_fluid_LT = CoolProp.AbstractState(HEOS_type, external_fluid_LT)
HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

# Cycle with its fixed states and mass flow rates
SC1 = Cycle("SC1")
SC1.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
SC1.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
SC1.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p4_prime)
SC1.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p4_prime)

# Compressor
SC1.Compressor = Compressor_2_param(cycle=SC1, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Define the functions for the optimization
############################################################

def iterative_process(p_gess, T_sub_current, T_sup_current) :
    p1_guess = p_gess[0] ; p3_guess = p_gess[1]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 1
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
    Tsat_1 = HEOS_working_fluid.T()
    SC1.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current, p=p1_guess)

        # Compute guessed state 3
    T_3 = SC1.Compressor.Solve(p_ex=p3_guess, state_in=SC1.state_1, mode="Non-Dimensional")[1]
    SC1.state_3 = State(HEOS_working_fluid, T=T_3, p=p3_guess)

        # Compute guessed state 9
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_9 = HEOS_working_fluid.T()
    SC1.state_9 = State(HEOS_working_fluid, T=Tsat_9 - T_sub_current, p=p3_guess)

        # Compute guessed state 10
    h10 = SC1.state_9.h
    SC1.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

    # STEP 2 : Compute the residual for the evaporator

    SC1.Evaporator = HEX_Design(states_in=[SC1.state_10, SC1.state_1_prime], states_out=[SC1.state_1, SC1.state_2_prime], name="Evaporator", mode="Non-Dimensional")
    Tpinch_real = SC1.Evaporator.Compute_Pinch()
    res_evap = Tpinch_real - T_pinch

    # STEP 3 : Compute the residual for the condenser

    SC1.Condenser = HEX_Design(states_in=[SC1.state_4_prime, SC1.state_3], states_out=[SC1.state_3_prime, SC1.state_9], name="Condenser", mode="Non-Dimensional")
    Tpinch_real = SC1.Condenser.Compute_Pinch()
    res_cond = Tpinch_real - T_pinch

    # STEP 4 : Assemble the residuals

    residuals = np.array([res_evap, res_cond])
    return residuals


def objective_function(optimization_vars) :

    # Unpack optimization variables
    T_sub_current = optimization_vars[0]
    T_sup_current = optimization_vars[1]

    # Initial guesses for pressures
    p1_guess = 5e5 ; p3_guess = 20e5
    p_guess = np.array([p1_guess, p3_guess])

    # Find the pressures that satisfy the pinch constraints
    try : 
        p_solution = fsolve(iterative_process, p_guess, args=(T_sub_current, T_sup_current))
        p3_solution = p_solution[1]
    except :
        if verbose:
            print("  - fsolve failed to converge.")
        return 1e6  # Return a large penalty value if fsolve fails

    # Compute the COP for the current cycle
    Delta_h_Condenser = SC1.state_3.h - SC1.state_9.h
    SC1.mdot_wf_bottom = Q / Delta_h_Condenser
    SC1.P_comp_bottom = SC1.Compressor.Solve(p_ex=p3_solution, state_in=SC1.state_1, mdot_wf=SC1.mdot_wf_bottom, mode="Dimensional")[0]
    COP = Q / SC1.P_comp_bottom
    
    # Print the current cycle performance if verbose
    if verbose:
        print(f"  - Current cycle with T_sub={T_sub_current:.2f} K and T_sup={T_sup_current:.2f} K has COP={COP:.4f}")

    # We want to maximize the COP, so we minimize its negative value
    return -COP


############################################################
# Optimization procedure
############################################################

# Initial guess and bounds for optimization variables
optimization_vars_guess = [(T_sub_min + T_sub_max) / 2 , (T_sup_min + T_sup_max) / 2]
bounds = [(T_sub_min, T_sub_max), (T_sup_min, T_sup_max)]

result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Powell", options={'maxiter': 20})

# Extract the best parameters
T_sub_best = result.x[0]
T_sup_best = result.x[1]
print("\nBest cycle found with parameters :")
print(f"  - Subcooling at condenser outlet : {T_sub_best:.2f} K")
print(f"  - Superheating at compressor inlet : {T_sup_best:.2f} K")

# Recompute the best cycle states (for safety)
p1_guess = 5e5 ; p3_guess = 20e5
p_guess = np.array([p1_guess, p3_guess])
p_best = fsolve(iterative_process, p_guess, args=(T_sub_best, T_sup_best))
p1_best = p_best[0]
p3_best = p_best[1]

# Compute heat exchangers and compressor with dimensional mode
Delta_h_Condenser = SC1.state_3.h - SC1.state_9.h
SC1.mdot_wf_bottom = Q / Delta_h_Condenser
SC1.P_comp_bottom = SC1.Compressor.Solve(p_ex=p3_best, state_in=SC1.state_1, mdot_wf=SC1.mdot_wf_bottom, mode="Dimensional")[0]
SC1.Evaporator = HEX_Design(states_in=[SC1.state_10, SC1.state_1_prime], states_out=[SC1.state_1, SC1.state_2_prime], mdot=[SC1.mdot_wf_bottom, None], name="Evaporator", mode="Dimensional", model="ACP70X")
T_pinch_evap = SC1.Evaporator.Compute_Pinch()
SC1.mdot_LT = SC1.Evaporator.mdot_h
SC1.Condenser = HEX_Design(states_in=[SC1.state_4_prime, SC1.state_3], states_out=[SC1.state_3_prime, SC1.state_9], mdot=[None, SC1.mdot_wf_bottom], name="Condenser", mode="Dimensional", model="ACP70X")
T_pinch_cond = SC1.Condenser.Compute_Pinch()
SC1.mdot_MT = SC1.Condenser.mdot_c

# Compute cycle performance
SC1.COP = SC1.Condenser.Q / SC1.P_comp_bottom

# Raise error if pinch points are not satisfied
if not (np.isclose(T_pinch_evap, T_pinch, atol=1e-4) and np.isclose(T_pinch_cond, T_pinch, atol=1e-4)):
    raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

end = time()
print(f"\nOptimization completed in {end - start:.2f} seconds.\n")


############################################################
# Plot the results
############################################################

full_details = False

if full_details:

    # Define the transforms 
    SC1.transforms = [Transform('comp', '1', '3', SC1.Compressor), 
                      Transform('hex', '10', '1',SC1.Evaporator, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                      Transform('adex', '9', '10', None), 
                      Transform('hex', '3', '9', SC1.Condenser, label_in_secondary='4_prime', label_out_secondary='3_prime')]

    # Plot T-s and p-h diagrams with saturation curve
    SC1.Ts_diagram(n=100, plot=True)
    SC1.ph_diagram(n=100, plot=True)

    # Plot energy and exergy charts
    SC1.energy_chart(plot=True)
    SC1.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    SC1.Evaporator._plot(save=True, name_cycle=SC1.name, plot=True)
    SC1.Condenser._plot(save=True, name_cycle=SC1.name, plot=True)


############################################################
# Print the results
############################################################

print(SC1)

if full_details:
    SC1.Evaporator.Compute_Area(plot=True, save=True, name_cycle=SC1.name)
    SC1.Condenser.Compute_Area(plot=True, save=True, name_cycle=SC1.name)
    print(SC1.Evaporator)
    print(SC1.Condenser)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / SC1.name / f"{SC1.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(SC1) + '\n')
        f.write('\n' + str(SC1.Evaporator) + '\n')
        f.write('\n' + str(SC1.Condenser) + '\n')