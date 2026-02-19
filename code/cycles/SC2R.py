
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
Q = 30e3                          # Power output at the condenser [W]
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
T3_prime = 40 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_MT = 5                    # Temperature glide of the external fluid in the heat sink [K]
T4_prime = T3_prime - glide_MT  # Outlet temperature of the external fluid in the heat sink [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
T5_prime = 55 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_HT = 5                    # Temperature glide in the gas cooler [K]
T6_prime = T5_prime + glide_HT  # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sub_min = 1                   # Minimum subcooling at the condenser outlet [K]
T_sub_max = 6                   # Maximum subcooling at the condenser outlet [K]
T_sup_1_min = 1                 # Minimum superheating at the exit of the first evaporator [K]
T_sup_1_max = 6                 # Maximum superheating at the exit of the first evaporator [K]
T_sup_3_min = 1                 # Minimum superheating at point 3 [K]
T_sup_3_max = 6                 # Maximum superheating at point 3 [K]


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
SC2R = Cycle("SC2R")
SC2R.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
SC2R.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
SC2R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
SC2R.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
SC2R.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
SC2R.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

# Compressors
SC2R.Compressor_1 = Compressor_2_param(cycle=SC2R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)
SC2R.Compressor_2 = Compressor_2_param(cycle=SC2R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)

# Ratio between the two evaporators
SC2R.beta = beta
ratio_evaporators = beta / (1 - beta)


############################################################
# Define the functions for the optimization
############################################################

def iterative_process(p_gess, T_sub_current, T_sup_current_1, T_sup_current_3): 
    p1_guess = p_gess[0] ; p3_guess = p_gess[1] ; p5_guess = p_gess[2]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 1 (saturated vapor)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
    Tsat_1 = HEOS_working_fluid.T()
    SC2R.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current_1, p = p1_guess)

        # Compute guessed state 3 (saturated vapor)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    SC2R.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current_3, p = p3_guess)

        # Compute guessed state 6 (subcooled liquid at the condenser outlet)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p5_guess, 0.0)
    Tsat_6 = HEOS_working_fluid.T()
    SC2R.state_6 = State(HEOS_working_fluid, T=Tsat_6 - T_sub_current, p=p5_guess)

    # STEP 2 : solve the recuperator MT to get states 4, 7, 8 
    SC2R.Recuperator_2 = HEX_Design(states_in=[SC2R.state_3, SC2R.state_6], states_out=[None, None], 
                                    mdot=[None, None], name="Recuperator_1", mode = "Non-Dimensional", 
                                    type = "Recuperator", epsilon=recuperator_effectiveness)
    SC2R.Recuperator_2.Solve_Recuperator()
    SC2R.state_4 = SC2R.Recuperator_2.state_out_c
    SC2R.state_7 = SC2R.Recuperator_2.state_out_h

        # Compute guessed state 8 
    h8 = SC2R.state_7.h
    SC2R.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    # STEP 3 : solve the recuperator LT to get states 2, 9, 10
    
    SC2R.Recuperator_1 = HEX_Design(states_in=[SC2R.state_1, SC2R.state_7], states_out=[None, None], 
                                    mdot=[None, None], name="Recuperator_2", mode = "Non-Dimensional", 
                                    type = "Recuperator", epsilon=recuperator_effectiveness)
    SC2R.Recuperator_1.Solve_Recuperator()
    SC2R.state_2 = SC2R.Recuperator_1.state_out_c
    SC2R.state_9 = SC2R.Recuperator_1.state_out_h

        # Compute guessed state 10
    h10 = SC2R.state_9.h
    SC2R.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

    # STEP 4 : Compute the residual for the first evaporator
    SC2R.Evaporator_LT = HEX_Design(states_in=[SC2R.state_10, SC2R.state_1_prime], states_out=[SC2R.state_1, SC2R.state_2_prime], 
                                 name="Evaporator_LT", mode="Non-Dimensional")
    Tpinch_real = SC2R.Evaporator_LT.Compute_Pinch()
    res_evap_LT = Tpinch_real - T_pinch

    # STEP 5 : Compute the residual for the second evaporator
        
        # Compute guessed state 3_comp (exit of first compressor)
    T_3_comp = SC2R.Compressor_1.Solve(p_ex=p3_guess, state_in=SC2R.state_2, mode = 'Non-Dimensional')[1]
    SC2R.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

    # Compute guessed state 3_evap
    def objective_h3_evap(h3_evap) :
        left = ratio_evaporators * (SC2R.state_1.h - SC2R.state_10.h) / (h3_evap - SC2R.state_8.h) * h3_evap + SC2R.state_3_comp.h
        right = (1 + ratio_evaporators * (SC2R.state_1.h - SC2R.state_10.h) / (h3_evap - SC2R.state_8.h)) * SC2R.state_3.h 
        return left - right
    
    h3_evap_guess = (SC2R.state_8.h + SC2R.state_3.h) / 2
    h3_evap = fsolve(objective_h3_evap, h3_evap_guess)[0]
    SC2R.state_3_evap = State(HEOS_working_fluid, h=h3_evap, p=p3_guess)

    SC2R.Evaporator_MT = HEX_Design(states_in=[SC2R.state_8, SC2R.state_3_prime], states_out=[SC2R.state_3_evap, SC2R.state_4_prime], 
                                    name="Evaporator_MT", mode="Non-Dimensional")
    Tpinch_real = SC2R.Evaporator_MT.Compute_Pinch()
    res_evap_MT = Tpinch_real - T_pinch

    # STEP 6 : Compute the residual for the condenser

        # Compute guessed state 5 (exit of second compressor)
    T_5 = SC2R.Compressor_2.Solve(p_ex=p5_guess, state_in=SC2R.state_4, mode = 'Non-Dimensional')[1]
    SC2R.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

    SC2R.Condenser = HEX_Design(states_in=[SC2R.state_5_prime, SC2R.state_5], states_out=[SC2R.state_6_prime, SC2R.state_6],
                                 name="Condenser", mode="Non-Dimensional")
    Tpinch_real = SC2R.Condenser.Compute_Pinch()
    res_cond = Tpinch_real - T_pinch

    # STEP 7 : Assemble the residuals

    residuals = np.array([res_evap_LT, res_evap_MT, res_cond])
    return residuals


def objective_function(optimization_vars) :

    # Unpack optimization variables
    T_sub_current = optimization_vars[0]
    T_sup_current_1 = optimization_vars[1]
    T_sup_current_3 = optimization_vars[2]

    # Initial guesses for pressures
    p1_guess = 5e5 ; p3_guess = 15e5 ; p5_guess = 25e5
    p_guess = np.array([p1_guess, p3_guess, p5_guess])

    # Find the pressures that satisfy the pinch constraints
    try :
        p_solution = fsolve(iterative_process, p_guess, args=(T_sub_current, T_sup_current_1, T_sup_current_3))
        p3_solution = p_solution[1]
        p5_solution = p_solution[2]
    except :
        if verbose:
            print("  - fsolve failed to converge.")
        return 1e6  # Return a large penalty value if fsolve fails        

    # Compute the COP for the current cycle
    Delta_h_Condenser = SC2R.state_5.h - SC2R.state_6.h
    SC2R.mdot_wf_top = Q / Delta_h_Condenser
    SC2R.P_comp_top = SC2R.Compressor_2.Solve(p_ex=p5_solution, state_in=SC2R.state_4, mdot_wf=SC2R.mdot_wf_top, mode="Dimensional")[0]

    SC2R.mdot_wf_bottom = SC2R.mdot_wf_top / (1 + ratio_evaporators * (SC2R.state_1.h - SC2R.state_10.h) / (SC2R.state_3_evap.h - SC2R.state_8.h))
    SC2R.P_comp_bottom = SC2R.Compressor_1.Solve(p_ex=p3_solution, state_in=SC2R.state_2, mdot_wf=SC2R.mdot_wf_bottom, mode="Dimensional")[0]
    COP = Q / (SC2R.P_comp_top + SC2R.P_comp_bottom)

    # Print the current cycle performance if verbose
    if verbose:
        print(f"  - Current cycle with T_sub={T_sub_current:.2f} K, T_sup_1={T_sup_current_1:.2f} K and T_sup_3={T_sup_current_3:.2f} K has COP={COP:.4f}")
    
    # We want to maximize the COP, so we minimize its negative value
    return -COP


############################################################
# Optimization procedure
############################################################

# Initial guess and bounds for optimization variables
optimization_vars_guess = [(T_sub_min + T_sub_max) / 2 , (T_sup_1_min + T_sup_1_max) / 2, (T_sup_3_min + T_sup_3_max) / 2]
bounds = [(T_sub_min, T_sub_max), (T_sup_1_min, T_sup_1_max), (T_sup_3_min, T_sup_3_max)]

result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Powell", options={'maxiter': 20})

# Extract the best parameters
T_sub_best = result.x[0]
T_sup_best_1 = result.x[1]
T_sup_best_3 = result.x[2]
print("\nBest cycle found with parameters :")
print(f"  - Subcooling at condenser outlet : {T_sub_best:.2f} K")
print(f"  - Superheating at point 1 : {T_sup_best_1:.2f} K")
print(f"  - Superheating at point 3 : {T_sup_best_3:.2f} K")

# Recompute the best cycle states (for satefy)
p1_guess = 5e5 ; p3_guess = 15e5 ; p5_guess = 25e5
p_guess = np.array([p1_guess, p3_guess, p5_guess])
p_best = fsolve(iterative_process, p_guess, args=(T_sub_best, T_sup_best_1, T_sup_best_3))
p1_best = p_best[0]
p3_best = p_best[1]
p5_best = p_best[2]

# Compute heat exchangers and compressor with dimensional mode
Delta_h_Condenser = SC2R.state_5.h - SC2R.state_6.h
SC2R.mdot_wf_top = Q / Delta_h_Condenser
SC2R.P_comp_top = SC2R.Compressor_2.Solve(p_ex=p5_best, state_in=SC2R.state_4, mdot_wf=SC2R.mdot_wf_top, mode="Dimensional")[0]
SC2R.mdot_wf_bottom = SC2R.mdot_wf_top / (1 + ratio_evaporators * (SC2R.state_1.h - SC2R.state_10.h) / (SC2R.state_3_evap.h - SC2R.state_8.h))
SC2R.P_comp_bottom = SC2R.Compressor_1.Solve(p_ex=p3_best, state_in=SC2R.state_2, mdot_wf=SC2R.mdot_wf_bottom, mode="Dimensional")[0]
SC2R.Evaporator_LT = HEX_Design(states_in=[SC2R.state_10, SC2R.state_1_prime], states_out=[SC2R.state_1, SC2R.state_2_prime], mdot = [SC2R.mdot_wf_bottom, None], name="Evaporator_LT", mode="Dimensional", model="ACH30EQ")
T_pinch_evap_LT = SC2R.Evaporator_LT.Compute_Pinch()
SC2R.mdot_LT = SC2R.Evaporator_LT.mdot_h
SC2R.Evaporator_MT = HEX_Design(states_in=[SC2R.state_8, SC2R.state_3_prime], states_out=[SC2R.state_3_evap, SC2R.state_4_prime],mdot = [SC2R.mdot_wf_top - SC2R.mdot_wf_bottom, None], name="Evaporator_MT", mode="Dimensional", model="ACH30EQ")
T_pinch_evap_MT = SC2R.Evaporator_MT.Compute_Pinch()
SC2R.mdot_MT = SC2R.Evaporator_MT.mdot_h
SC2R.Condenser = HEX_Design(states_in=[SC2R.state_5_prime, SC2R.state_5], states_out=[SC2R.state_6_prime, SC2R.state_6], mdot=[None, SC2R.mdot_wf_top], name="Condenser", mode="Dimensional", model="ACH65")
T_pinch_cond = SC2R.Condenser.Compute_Pinch()
SC2R.mdot_HT = SC2R.Condenser.mdot_c
SC2R.Recuperator_2 = HEX_Design(states_in=[SC2R.state_3, SC2R.state_6], states_out=[SC2R.state_4, SC2R.state_7], mdot=[SC2R.mdot_wf_top, SC2R.mdot_wf_top], name="Recuperator_1", mode="Dimensional", type="Recuperator", epsilon=recuperator_effectiveness, model="ACK16")
SC2R.Recuperator_2.Solve_Recuperator()
SC2R.Recuperator_1 = HEX_Design(states_in=[SC2R.state_1, SC2R.state_7], states_out=[SC2R.state_2, SC2R.state_9], mdot=[SC2R.mdot_wf_bottom, SC2R.mdot_wf_bottom], name="Recuperator_2", mode="Dimensional", type="Recuperator", epsilon=recuperator_effectiveness, model="ACK16")
SC2R.Recuperator_1.Solve_Recuperator()

# Compute cycle performance
SC2R.COP = SC2R.Condenser.Q / (SC2R.P_comp_top + SC2R.P_comp_bottom)

# Raise error if pinch points are not satisfied
if not (np.isclose(T_pinch_evap_LT, T_pinch, atol=1e-4) and np.isclose(T_pinch_evap_MT, T_pinch, atol=1e-4) and np.isclose(T_pinch_cond, T_pinch, atol=1e-4)):
    raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

end = time()
print(f"\nOptimization completed in {end - start:.2f} seconds.\n")

############################################################
# Plot the results
############################################################

full_details = False

if full_details:

    # Define the transforms 
    SC2R.transforms = [Transform('isobaric_mixing', '3_comp', '3_evap', None),
                    Transform('comp', '2', '3_comp', SC2R.Compressor_1),
                    Transform('comp', '4', '5', SC2R.Compressor_2),
                    Transform('hex', '5', '6', SC2R.Condenser, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                    Transform('adex', '7', '8', None),
                    Transform('adex', '9', '10', None),
                    Transform('hex', '8', '3_evap',SC2R.Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                    Transform('hex', '10', '1',SC2R.Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'),
                    Transform('hex', '3', '4',SC2R.Recuperator_2, label_in_secondary='6', label_out_secondary='7'),
                    Transform('hex', '1', '2',SC2R.Recuperator_1, label_in_secondary='7', label_out_secondary='9')]

    # Plot T-s diagram with saturation curve
    SC2R.Ts_diagram(n=100, plot=True)
    SC2R.ph_diagram(n=100, plot=True)

    # Plot energy and exergy charts
    SC2R.energy_chart(plot=True)
    SC2R.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    SC2R.Evaporator_LT._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Evaporator_MT._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Condenser._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Recuperator_1._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Recuperator_2._plot(save=True, name_cycle=SC2R.name, plot=True)


############################################################
# Print the results
############################################################

print(SC2R)

if full_details:
    SC2R.Evaporator_LT.Compute_Area()
    SC2R.Evaporator_MT.Compute_Area()
    SC2R.Condenser.Compute_Area()
    SC2R.Recuperator_1.Compute_Area()
    SC2R.Recuperator_2.Compute_Area()
    print(SC2R.Evaporator_LT)
    print(SC2R.Evaporator_MT)
    print(SC2R.Condenser)
    print(SC2R.Recuperator_1)
    print(SC2R.Recuperator_2)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / SC2R.name / f"{SC2R.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(SC2R) + '\n')
        f.write('\n' + str(SC2R.Evaporator_LT) + '\n')
        f.write('\n' + str(SC2R.Evaporator_MT) + '\n')
        f.write('\n' + str(SC2R.Condenser) + '\n')
        f.write('\n' + str(SC2R.Recuperator_1) + '\n')
        f.write('\n' + str(SC2R.Recuperator_2) + '\n')