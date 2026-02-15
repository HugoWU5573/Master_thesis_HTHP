
############################################################
# Import libraries and modules
############################################################

import sys
from pathlib import Path

# Add the parent directory (code) to sys.path to enable relative imports
code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from components.state import State
from components.compressor import Compressor_2_param
from components.HEX import HEX_Design
from components.cycle import Cycle
import CoolProp
import numpy as np
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt

# Set verbosity
verbose = True

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
Q = 30e3                        # Power output at the condenser [W]

# Heat source parameters
external_fluid_LT = 'Water'     # External fluid in the heat source
glide_LT = 5                    # Temperature glide of the external fluid in the heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_MT = 'Water'     # External fluid in the heat sink
glide_MT = 5                    # Temperature glide of the external fluid in the heat sink [K]
p4_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sub_min = 1                   # Minimum subcooling at the condenser outlet [K]
T_sub_max = 8                   # Maximum subcooling at the condenser outlet [K]
T_sup_min = 1                   # Minimum superheating at the evaporator outlet [K]
T_sup_max = 8                   # Maximum superheating at the evaporator outlet [K]


############################################################
# Define the range of operating conditions
############################################################

T1_prime_min = 10 + 273.15      # Minimum inlet temperature of the external fluid in the heat source [K]
T1_prime_max = 30 + 273.15      # Maximum inlet temperature of the external fluid in the heat source [K]
T4_prime_min = 30 + 273.15      # Minimum inlet temperature of the external fluid in the heat sink [K]
T4_prime_max = 50 + 273.15      # Maximum inlet temperature of the external fluid in the heat sink [K]

nb_steps = 11                   # Number of steps for each operating condition

T1_prime_range = np.linspace(T1_prime_min, T1_prime_max, nb_steps)
T4_prime_range = np.linspace(T4_prime_min, T4_prime_max, nb_steps)

Evaporator_areas = np.zeros((len(T1_prime_range), len(T4_prime_range)))
Condenser_areas = np.zeros((len(T1_prime_range), len(T4_prime_range)))
Recuperator_areas = np.zeros((len(T1_prime_range), len(T4_prime_range)))
COP_values = np.zeros((len(T1_prime_range), len(T4_prime_range)))

for i in range(len(T1_prime_range)):
    for j in range(len(T4_prime_range)):

        # Print progress
        print(f"Progress: {i*len(T1_prime_range) + j + 1}/{len(T1_prime_range) * len(T4_prime_range)}", end='\r')

        T1_prime = T1_prime_range[i]     # Inlet temperature of the external fluid in the heat source [K]
        T4_prime = T4_prime_range[j]     # Inlet temperature of the external fluid in the heat sink [K]
        if verbose:
            print(f"\nOperating conditions : T1'={T1_prime - 273.15:.2f} °C, T4'={T4_prime - 273.15:.2f} °C")

        T2_prime = T1_prime - glide_LT   # Outlet temperature of the external fluid in the heat source [K]
        T3_prime = T4_prime + glide_MT  # Outlet temperature of the external fluid in the heat sink [K]

        ############################################################
        # Instantiate objects
        ############################################################

        HEOS_type = "HEOS"  # Choose from "HEOS", "TTSE&HEOS"

        # CoolProp low-level interface for all the fluids
        HEOS_external_fluid_LT = CoolProp.AbstractState(HEOS_type, external_fluid_LT)
        HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
        HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

        # Cycle with its fixed states and mass flow rates
        SC1R = Cycle("SC1R")
        SC1R.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
        SC1R.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
        SC1R.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p4_prime)
        SC1R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p4_prime)

        # Compressor
        SC1R.Compressor = Compressor_2_param(cycle=SC1R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


        ############################################################
        # Define the functions for the optimization
        ############################################################

        def iterative_process(p_gess, T_sub_current, T_sup_current):
            p1_guess = p_gess[0] ; p3_guess = p_gess[1]

            # STEP 1 : Compute the states based on T_sup_current and T_sub_current

                # Compute guessed state 1
            HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
            Tsat_1 = HEOS_working_fluid.T()
            SC1R.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current, p=p1_guess)

                # Compute guessed state 8
            HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
            Tsat_8 = HEOS_working_fluid.T()
            SC1R.state_8 = State(HEOS_working_fluid, T=Tsat_8 - T_sub_current, p=p3_guess)

            # STEP 2 : Solve the Recuperator to get states 2, 9 and 10

                # Compute guessed states 2 and 9
            SC1R.Recuperator = HEX_Design(states_in=[SC1R.state_1, SC1R.state_8], states_out=[None, None], mdot = [None, None], name = 'Recuperator', mode="Non-Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness)
            SC1R.Recuperator.Solve_Recuperator()
            SC1R.state_2 = SC1R.Recuperator.state_out_c
            SC1R.state_9 = SC1R.Recuperator.state_out_h

                # Compute guessed state 10
            h10 = SC1R.state_9.h
            SC1R.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

            # STEP 3 : Solve the Compressor to get state 3 and mass flow rate

                # Compute guessed state 3
            T_3 = SC1R.Compressor.Solve(p_ex=p3_guess, state_in=SC1R.state_2, mode="Non-Dimensional")[1]
            SC1R.state_3 = State(HEOS_working_fluid, T=T_3, p=p3_guess)

            # STEP 4 : Compute the residual for the evaporator

            SC1R.Evaporator = HEX_Design(states_in=[SC1R.state_10, SC1R.state_1_prime], states_out=[SC1R.state_1, SC1R.state_2_prime], name="Evaporator", mode="Non-Dimensional")
            Tpinch_real = SC1R.Evaporator.Compute_Pinch()
            res_evap = Tpinch_real - T_pinch

            # STEP 5 : Compute the residual for the condenser

            SC1R.Condenser = HEX_Design(states_in=[SC1R.state_4_prime, SC1R.state_3], states_out=[SC1R.state_3_prime, SC1R.state_8], name="Condenser", mode="Non-Dimensional")
            Tpinch_real = SC1R.Condenser.Compute_Pinch()
            res_cond = Tpinch_real - T_pinch

            # STEP 6 : Assemble the residuals

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
                if verbose :
                    print("  - fsolve failed to converge.")
                return 1e6  # Return a large penalty value if fsolve fails

            # Compute the COP for the current cycle
            Delta_h_Condenser = SC1R.state_3.h - SC1R.state_8.h
            SC1R.mdot_wf_bottom = Q / Delta_h_Condenser
            SC1R.P_comp_bottom = SC1R.Compressor.Solve(p_ex=p3_solution, state_in=SC1R.state_2, mdot_wf=SC1R.mdot_wf_bottom, mode="Dimensional")[0]
            COP = Q / SC1R.P_comp_bottom

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

        # Recompute the best cycle states (for safety)
        p1_guess = 5e5 ; p3_guess = 20e5
        p_guess = np.array([p1_guess, p3_guess])
        p_best = fsolve(iterative_process, p_guess, args=(T_sub_best, T_sup_best))
        p1_best = p_best[0]
        p3_best = p_best[1]

        # Compute heat exchangers and compressor with dimensional mode
        Delta_h_Condenser = SC1R.state_3.h - SC1R.state_8.h
        SC1R.mdot_wf_bottom = Q / Delta_h_Condenser
        SC1R.P_comp_bottom = SC1R.Compressor.Solve(p_ex=p3_best, state_in=SC1R.state_2, mdot_wf=SC1R.mdot_wf_bottom, mode="Dimensional")[0]
        SC1R.Evaporator = HEX_Design(states_in=[SC1R.state_10, SC1R.state_1_prime], states_out=[SC1R.state_1, SC1R.state_2_prime], mdot=[SC1R.mdot_wf_bottom, None], name="Evaporator", mode="Dimensional")
        T_pinch_evap = SC1R.Evaporator.Compute_Pinch()
        SC1R.mdot_LT = SC1R.Evaporator.mdot_h
        SC1R.Condenser = HEX_Design(states_in=[SC1R.state_4_prime, SC1R.state_3], states_out=[SC1R.state_3_prime, SC1R.state_8], mdot=[None, SC1R.mdot_wf_bottom], name="Condenser", mode="Dimensional")
        T_pinch_cond = SC1R.Condenser.Compute_Pinch()
        SC1R.mdot_MT = SC1R.Condenser.mdot_c
        SC1R.Recuperator = HEX_Design(states_in=[SC1R.state_1, SC1R.state_8], states_out=[SC1R.state_2, SC1R.state_9], mdot=[SC1R.mdot_wf_bottom, SC1R.mdot_wf_bottom], name = 'Recuperator', mode="Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness)
        SC1R.Recuperator.Solve_Recuperator()

        # Compute cycle performance
        SC1R.COP = SC1R.Condenser.Q / SC1R.P_comp_bottom
        COP_values[i, j] = SC1R.COP

        # Raise error if pinch points are not satisfied
        if not (np.isclose(T_pinch_evap, T_pinch, atol=1e-2) and np.isclose(T_pinch_cond, T_pinch, atol=1e-2)):
            print(f"  - T_pinch_evap = {T_pinch_evap:.4f} K, T_pinch_cond = {T_pinch_cond:.4f} K")
            raise ValueError("Pinch point constraints not satisfied in the best cycle found.")
        
        ############################################################
        # Compute heat exchanger areas
        ############################################################

        SC1R.Evaporator.Compute_Area()
        SC1R.Condenser.Compute_Area()
        SC1R.Recuperator.Compute_Area()

        Evaporator_areas[i, j] = SC1R.Evaporator.A
        Condenser_areas[i, j] = SC1R.Condenser.A
        Recuperator_areas[i, j] = SC1R.Recuperator.A


############################################################
# Compute the statistics of the results
############################################################

Evaporators_areas_flat = Evaporator_areas.flatten()
Condenser_areas_flat = Condenser_areas.flatten()
Recuperator_areas_flat = Recuperator_areas.flatten()

# Compute the mean values
mean_evaporator_area = np.mean(Evaporators_areas_flat)
mean_condenser_area = np.mean(Condenser_areas_flat)
mean_recuperator_area = np.mean(Recuperator_areas_flat)

# Compute the standard deviations
std_evaporator_area = np.std(Evaporators_areas_flat)
std_condenser_area = np.std(Condenser_areas_flat)
std_recuperator_area = np.std(Recuperator_areas_flat)


############################################################
# Print the results
############################################################

if verbose:
    print("\nCOP Values:")
    print(COP_values)
    print("\nEvaporator Areas (m²):")
    print(Evaporator_areas)
    print("\nCondenser Areas (m²):")
    print(Condenser_areas)
    print("\nRecuperator Areas (m²):")
    print(Recuperator_areas)

print("\n\nStatistics of Evaporator Areas:")
print(f"  - Mean: {np.mean(Evaporators_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Evaporators_areas_flat):.4f} m², which is {100 * np.std(Evaporators_areas_flat) / np.mean(Evaporators_areas_flat):.2f}% of the mean")

print("\n\nStatistics of Condenser Areas:")
print(f"  - Mean: {np.mean(Condenser_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Condenser_areas_flat):.4f} m², which is {100 * np.std(Condenser_areas_flat) / np.mean(Condenser_areas_flat):.2f}% of the mean")

print("\n\nStatistics of Recuperator Areas:")
print(f"  - Mean: {np.mean(Recuperator_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Recuperator_areas_flat):.4f} m², which is {100 * np.std(Recuperator_areas_flat) / np.mean(Recuperator_areas_flat):.2f}% of the mean")

# Save prints to a text file
output_file = Path(__file__).parent.parent / "Figures" / SC1R.name / f"{SC1R.name}_statistics.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    f.write(f"Temperature ranges:\n")
    f.write(f"  - T1' range: {T1_prime_min - 273.15:.1f} C to {T1_prime_max - 273.15:.1f} C ({len(T1_prime_range)} values)\n")
    f.write(f"  - T4' range: {T4_prime_min - 273.15:.1f} C to {T4_prime_max - 273.15:.1f} C ({len(T4_prime_range)} values)\n")
    f.write(f"  - Total number of operating conditions: {len(T1_prime_range) * len(T4_prime_range)}\n\n")
    f.write("Evaporator Areas:\n")
    f.write(f"  - Mean: {np.mean(Evaporators_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Evaporators_areas_flat):.4f} m2, which is {100 * np.std(Evaporators_areas_flat) / np.mean(Evaporators_areas_flat):.2f}% of the mean\n\n")
    f.write("Condenser Areas:\n")
    f.write(f"  - Mean: {np.mean(Condenser_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Condenser_areas_flat):.4f} m2, which is {100 * np.std(Condenser_areas_flat) / np.mean(Condenser_areas_flat):.2f}% of the mean\n")
    f.write("\nRecuperator Areas:\n")
    f.write(f"  - Mean: {np.mean(Recuperator_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Recuperator_areas_flat):.4f} m2, which is {100 * np.std(Recuperator_areas_flat) / np.mean(Recuperator_areas_flat):.2f}% of the mean\n")

############################################################
# Plot the results
############################################################

## 1. Plot the evaporator areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Evaporators_areas_flat, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel(r'$Evaporator$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()

## 2. Plot the condenser areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Condenser_areas_flat, bins=10, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel(r'$Condenser$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()

## 3. Plot the recuperator areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Recuperator_areas_flat, bins=10, edgecolor='black', alpha=0.7, color='seagreen')
ax.set_xlabel(r'$Recuperator$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()