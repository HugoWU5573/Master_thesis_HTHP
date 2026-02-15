
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
Q = 30e3                        # Power output at the gas cooler [W]

# Heat source parameters
external_fluid_MT = 'Water'     # External fluid in the heat source
glide_MT = 5                    # Temperature glide of the external fluid in the heat source [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
glide_HT = 55                   # Temperature glide in the gas cooler [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sup_min = 2                   # Minimum superheating at the exit of the evaporator [K]
T_sup_max = 8                   # Maximum superheating at the exit of the evaporator [K]
T_6_max = 350                   # Maximum outlet temperature of the gas cooler [K]


############################################################
# Define the range of operating conditions
############################################################

T3_prime_min = 40 + 273.15      # Minimum inlet temperature of the external fluid in the heat source [K]
T3_prime_max = 50 + 273.15      # Maximum inlet temperature of the external fluid in the heat source [K]
T5_prime_min = 55 + 273.15      # Minimum inlet temperature of the external fluid in the heat sink [K]
T5_prime_max = 65 + 273.15      # Maximum inlet temperature of the external fluid in the heat sink [K]

nb_steps_1 = 6                   # Number of steps for each operating condition
nb_steps_2 = 6

T3_prime_range = np.linspace(T3_prime_min, T3_prime_max, nb_steps_1)
T5_prime_range = np.linspace(T5_prime_min, T5_prime_max, nb_steps_2)

Evaporator_areas = np.zeros((len(T3_prime_range), len(T5_prime_range)))
GasCooler_areas = np.zeros((len(T3_prime_range), len(T5_prime_range)))
Recuperator_areas = np.zeros((len(T3_prime_range), len(T5_prime_range)))
COP_values = np.zeros((len(T3_prime_range), len(T5_prime_range)))

T_sup_last = (T_sup_min + T_sup_max) / 2
T6_last = (T5_prime_min + T_pinch + T_6_max) / 2

for i in range(len(T3_prime_range)):
    for j in range(len(T5_prime_range)):

        # Print progress
        print(f"Progress: {i*len(T5_prime_range) + j + 1}/{len(T3_prime_range) * len(T5_prime_range)}", end='\r')

        T3_prime = T3_prime_range[i]     # Inlet temperature of the external fluid in the heat source [K]
        T5_prime = T5_prime_range[j]     # Inlet temperature of the external fluid in the heat sink [K]
        if verbose:
            print(f"\nOperating conditions : T3'={T3_prime - 273.15:.2f} °C, T5'={T5_prime - 273.15:.2f} °C")

        T4_prime = T3_prime - glide_MT   # Outlet temperature of the external fluid in the heat source [K]
        T6_prime = T5_prime + glide_HT   # Outlet temperature of the external fluid in the heat sink [K]


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
            p3_guess = 10e5 ; p5_guess = 41e5
            p_guess = np.array([p3_guess, p5_guess])

            # Find the pressures that satisfy the pinch constraints
            try :
                p_solution = fsolve(iterative_process, p_guess, args=(T_sup_current, T_6_current))
                p5_solution = p_solution[1]
            except :
                return 1e6 # Return a large penalty if fsolve fails
            
            # Compute the COP for the current cycle
            Delta_h_GasCooler = TC1R.state_5.h - TC1R.state_6.h
            TC1R.mdot_wf_top = Q / Delta_h_GasCooler
            TC1R.P_comp_top = TC1R.Compressor.Solve(p_ex=p5_solution, state_in=TC1R.state_4, mdot_wf=TC1R.mdot_wf_top, mode="Dimensional")[0]
            COP = Q / TC1R.P_comp_top

            # We want to maximize the COP, so we minimize its negative value
            return -COP    


        ############################################################
        # Optimization procedure
        ############################################################

        T_6_min = T5_prime + T_pinch  # Minimum outlet temperature of the gas cooler [K]
                
        # Initial guess and bounds for optimization variables
        optimization_vars_guess = [T_sup_last, T6_last]
        bounds = [(T_sup_min, T_sup_max), (T_6_min, T_6_max)]

        result = minimize(objective_function, optimization_vars_guess, bounds=bounds, method="Powell", options={'maxiter': 20})

        # Extract the best parameters
        T_sup_best = result.x[0]
        T_6_best = result.x[1]
        T_sup_last = T_sup_best
        T6_last = T_6_best

        # Recompute the best cycle states (for safety)
        p3_guess = 10e5 ; p5_guess = 41e5
        p_guess = np.array([p3_guess, p5_guess])
        p_best = fsolve(iterative_process, p_guess, args=(T_sup_best, T_6_best))
        p3_best = p_best[0]
        p5_best = p_best[1]

        # Compute heat exchangers and compressor with dimensional mode
        Delta_h_GasCooler = TC1R.state_5.h - TC1R.state_6.h
        TC1R.mdot_wf_top = Q / Delta_h_GasCooler
        TC1R.P_comp_top = TC1R.Compressor.Solve(p_ex=p5_best, state_in=TC1R.state_4, mdot_wf=TC1R.mdot_wf_top, mode="Dimensional")[0]
        TC1R.Evaporator = HEX_Design(states_in=[TC1R.state_8, TC1R.state_3_prime], states_out=[TC1R.state_3, TC1R.state_4_prime], mdot=[TC1R.mdot_wf_top, None], name="Evaporator", mode="Dimensional")
        T_pinch_evap = TC1R.Evaporator.Compute_Pinch()
        TC1R.mdot_MT = TC1R.Evaporator.mdot_h
        TC1R.GasCooler = HEX_Design(states_in=[TC1R.state_5_prime, TC1R.state_5], states_out=[TC1R.state_6_prime, TC1R.state_6], mdot=[None, TC1R.mdot_wf_top], name="Gas Cooler", mode="Dimensional")
        T_pinch_gas_cooler = TC1R.GasCooler.Compute_Pinch()
        TC1R.mdot_HT = TC1R.GasCooler.mdot_c
        TC1R.Recuperator = HEX_Design(states_in=[TC1R.state_3, TC1R.state_6], states_out=[TC1R.state_4, TC1R.state_7], mdot=[TC1R.mdot_wf_top, TC1R.mdot_wf_top], name = 'Recuperator', mode="Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness)
        TC1R.Recuperator.Solve_Recuperator()

        # Compute cycle performance
        TC1R.COP = TC1R.GasCooler.Q / TC1R.P_comp_top
        COP_values[i, j] = TC1R.COP

        # Warning message if the highest pressure of the cycle is lower than the critical pressure (42.5 bar for R290)
        if TC1R.state_5.p < HEOS_working_fluid.p_critical() :
            print("WARNING : The highest pressure of the cycle is lower than the critical pressure of the working fluid ({:.2f} bar < {:.2f} bar)".format(TC1R.state_5.p / 1e5, HEOS_working_fluid.p_critical() / 1e5))


        # Warning message if the highest pressure of the cycle exceeds 50 bar
        if TC1R.state_5.p > 5e6 :
            print("WARNING : The highest pressure of the cycle exceeds 50 bar ({:.2f} bar) -> not too problematic for the current analysis.".format(TC1R.state_5.p / 1e5))

        # Raise error if pinch points are not satisfied
        if not (np.isclose(T_pinch_evap, T_pinch, atol=1e-4) and np.isclose(T_pinch_gas_cooler, T_pinch, atol=1e-4)):
            raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

        ############################################################
        # Compute heat exchanger areas
        ############################################################

        TC1R.Evaporator.Compute_Area()
        TC1R.GasCooler.Compute_Area()
        TC1R.Recuperator.Compute_Area()

        Evaporator_areas[i, j] = TC1R.Evaporator.A
        GasCooler_areas[i, j] = TC1R.GasCooler.A
        Recuperator_areas[i, j] = TC1R.Recuperator.A


############################################################
# Compute the statistics of the results
############################################################

Evaporators_areas_flat = Evaporator_areas.flatten()
GasCooler_areas_flat = GasCooler_areas.flatten()
Recuperators_areas_flat = Recuperator_areas.flatten()

# Compute the mean values
mean_evaporator_area = np.mean(Evaporators_areas_flat)
mean_gascooler_area = np.mean(GasCooler_areas_flat)
mean_recuperator_area = np.mean(Recuperators_areas_flat)

# Compute the standard deviations
std_evaporator_area = np.std(Evaporators_areas_flat)
std_gascooler_area = np.std(GasCooler_areas_flat)
std_recuperator_area = np.std(Recuperators_areas_flat)


############################################################
# Print the results
############################################################

if verbose:
    print("\nCOP Values:")
    print(COP_values)
    print("\nEvaporator Areas (m²):")
    print(Evaporator_areas)
    print("\nGasCooler Areas (m²):")
    print(GasCooler_areas)
    print("\nRecuperator Areas (m²):")
    print(Recuperator_areas)

print("\n\nStatistics of Evaporator Areas:")
print(f"  - Mean: {np.mean(Evaporators_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Evaporators_areas_flat):.4f} m², which is {100 * np.std(Evaporators_areas_flat) / np.mean(Evaporators_areas_flat):.2f}% of the mean")

print("\n\nStatistics of GasCooler Areas:")
print(f"  - Mean: {np.mean(GasCooler_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(GasCooler_areas_flat):.4f} m², which is {100 * np.std(GasCooler_areas_flat) / np.mean(GasCooler_areas_flat):.2f}% of the mean")

print("\n\nStatistics of Recuperator Areas:")
print(f"  - Mean: {np.mean(Recuperators_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Recuperators_areas_flat):.4f} m², which is {100 * np.std(Recuperators_areas_flat) / np.mean(Recuperators_areas_flat):.2f}% of the mean")

# Save prints to a text file
output_file = Path(__file__).parent.parent / "Figures" / TC1R.name / f"{TC1R.name}_statistics.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    f.write(f"Temperature ranges:\n")
    f.write(f"  - T3' range: {T3_prime_min - 273.15:.1f} C to {T3_prime_max - 273.15:.1f} C ({len(T3_prime_range)} values)\n")
    f.write(f"  - T5' range: {T5_prime_min - 273.15:.1f} C to {T5_prime_max - 273.15:.1f} C ({len(T5_prime_range)} values)\n")
    f.write(f"  - Total number of operating conditions: {len(T3_prime_range) * len(T5_prime_range)}\n\n")
    f.write("Evaporator Areas:\n")
    f.write(f"  - Mean: {np.mean(Evaporators_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Evaporators_areas_flat):.4f} m2, which is {100 * np.std(Evaporators_areas_flat) / np.mean(Evaporators_areas_flat):.2f}% of the mean\n\n")
    f.write("GasCooler Areas:\n")
    f.write(f"  - Mean: {np.mean(GasCooler_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(GasCooler_areas_flat):.4f} m2, which is {100 * np.std(GasCooler_areas_flat) / np.mean(GasCooler_areas_flat):.2f}% of the mean\n")
    f.write("\nRecuperator Areas:\n")
    f.write(f"  - Mean: {np.mean(Recuperators_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Recuperators_areas_flat):.4f} m2, which is {100 * np.std(Recuperators_areas_flat) / np.mean(Recuperators_areas_flat):.2f}% of the mean\n")
    

############################################################
# Plot the results
############################################################

## 1. Plot the evaporator areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Evaporators_areas_flat, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel(r'$Evaporator$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()

## 2. Plot the gascooler areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(GasCooler_areas_flat, bins=12, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel(r'$GasCooler$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()

## 3. Plot the recuperator areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Recuperators_areas_flat, bins=12, edgecolor='black', alpha=0.7, color='seagreen')
ax.set_xlabel(r'$Recuperator$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()