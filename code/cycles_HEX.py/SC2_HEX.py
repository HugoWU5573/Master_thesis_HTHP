
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
beta = 0.5                      # beta = Q_evap_MT / (Q_evap_LT + Q_evap_MT) [-]

# Heat sources parameters

    # 1. LT source
external_fluid_LT = 'Water'     # External fluid in the heat source
glide_LT = 5                    # Temperature glide of the external fluid in the heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

    # 2. MT source
external_fluid_MT = 'Water'     # External fluid in the heat sink
glide_MT = 5                    # Temperature glide of the external fluid in the heat sink [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
glide_HT = 5                    # Temperature glide in the gas cooler [K]
p5_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Bounds for the optimization parameters

T_sub_min = 1                   # Minimum subcooling at the condenser outlet [K]
T_sub_max = 8                   # Maximum subcooling at the condenser outlet [K]
T_sup_1_min = 1                 # Minimum superheating at the first compressor inlet [K]
T_sup_1_max = 8                 # Maximum superheating at the first compressor inlet [K]
T_sup_3_min = 5                 # Minimum superheating at the second compressor inlet [K]
T_sup_3_max = 12                # Maximum superheating at the second compressor inlet [K]

############################################################
# Define the range of operating conditions
############################################################

T1_prime_min = 10 + 273.15      # Minimum inlet temperature of the external fluid in the heat source [K]
T1_prime_max = 20 + 273.15      # Maximum inlet temperature of the external fluid in the heat source [K]
T3_prime_min = 35 + 273.15      # Minimum inlet temperature of the external fluid in the heat sink [K]
T3_prime_max = 45 + 273.15      # Maximum inlet temperature of the external fluid in the heat sink [K]
T5_prime_min = 50 + 273.15      # Minimum inlet temperature of the external fluid in the heat sink [K]
T5_prime_max = 60 + 273.15      # Maximum inlet temperature of the external fluid in the heat sink [K]

nb_steps = 6                   # Number of steps for each operating condition

T1_prime_range = np.linspace(T1_prime_min, T1_prime_max, nb_steps)
T3_prime_range = np.linspace(T3_prime_min, T3_prime_max, nb_steps)
T5_prime_range = np.linspace(T5_prime_min, T5_prime_max, nb_steps)

Evaporator_LT_areas = np.zeros((len(T1_prime_range), len(T3_prime_range), len(T5_prime_range)))
Evaporator_MT_areas = np.zeros((len(T1_prime_range), len(T3_prime_range), len(T5_prime_range)))
Condenser_areas = np.zeros((len(T1_prime_range), len(T3_prime_range), len(T5_prime_range)))
COP_values = np.zeros((len(T1_prime_range), len(T3_prime_range), len(T5_prime_range)))

for i in range(len(T1_prime_range)):
    for j in range(len(T3_prime_range)):
        for k in range(len(T5_prime_range)):

            # Print progress
            print(f"Progress: {i*len(T3_prime_range)*len(T5_prime_range) + j*len(T5_prime_range) + k + 1}/{len(T1_prime_range) * len(T3_prime_range) * len(T5_prime_range)}", end='\r')

            T1_prime = T1_prime_range[i]     # Inlet temperature of the external fluid in the heat source [K]
            T3_prime = T3_prime_range[j]     # Inlet temperature of the external fluid in the heat sink [K]
            T5_prime = T5_prime_range[k]     # Inlet temperature of the external fluid in the heat sink [K]
            if verbose:
                print(f"\nOperating conditions : T1'={T1_prime - 273.15:.2f} °C, T3'={T3_prime - 273.15:.2f} °C, T5'={T5_prime - 273.15:.2f} °C")

            T2_prime = T1_prime - glide_LT   # Outlet temperature of the external fluid in the heat source [K]
            T4_prime = T3_prime - glide_MT  # Outlet temperature of the external fluid in the heat sink [K]
            T6_prime = T5_prime + glide_HT   # Outlet temperature of the external fluid in the heat sink [K]

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
            SC2 = Cycle("SC2")
            SC2.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
            SC2.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
            SC2.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
            SC2.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
            SC2.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
            SC2.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

            # Compressors
            SC2.Compressor_1 = Compressor_2_param(cycle=SC2, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)
            SC2.Compressor_2 = Compressor_2_param(cycle=SC2, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)

            # Ratio between the two evaporators
            SC2.beta = beta
            ratio_evaporators = beta / (1 - beta)


            ############################################################
            # Define the functions for the optimization
            ############################################################

            def iterative_process(p_gess, T_sub_current, T_sup_current_1, T_sup_current_3) :
                p1_guess = p_gess[0] ; p3_guess = p_gess[1] ; p5_guess = p_gess[2]

                # STEP 1 : Compute the states based on the guesses values

                    # Compute guessed state 1 (first compressor inlet is superheated)
                HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
                Tsat_1 = HEOS_working_fluid.T()
                SC2.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current_1, p=p1_guess)

                    # Compute guessed state 3_comp (exit of first compressor)
                T_3_comp = SC2.Compressor_1.Solve(p_ex=p3_guess, state_in=SC2.state_1, mode = 'Non-Dimensional')[1]
                SC2.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

                    # Compute guessed state 3 (second compressor inlet is superheated)
                HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
                Tsat_3 = HEOS_working_fluid.T()
                SC2.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current_3, p=p3_guess)

                    # Compute guessed state 5 (exit of second compressor)
                T_5 = SC2.Compressor_2.Solve(p_ex=p5_guess, state_in=SC2.state_3, mode = 'Non-Dimensional')[1]
                SC2.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

                    # Compute guessed state 7 (condenser outlet is subcooled)
                HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p5_guess, 0.0)
                Tsat_7 = HEOS_working_fluid.T()
                SC2.state_7 = State(HEOS_working_fluid, T=Tsat_7 - T_sub_current, p=p5_guess)
                SC2.state_9 = State(HEOS_working_fluid, T=Tsat_7 - T_sub_current, p=p5_guess)  # States 7 and 9 are the same in this configuration

                    # Compute guessed state 8
                h8 = SC2.state_7.h
                SC2.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

                    # Compute guessed state 10
                h10 = SC2.state_9.h
                SC2.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

                # STEP 2 : Compute the residual for the first evaporator

                SC2.Evaporator = HEX_Design(states_in=[SC2.state_10, SC2.state_1_prime], states_out=[SC2.state_1, SC2.state_2_prime], name="Evaporator", mode="Non-Dimensional")
                Tpinch_real = SC2.Evaporator.Compute_Pinch()
                res_evap_LT = Tpinch_real - T_pinch

                # STEP 3 : Compute the residual for the second evaporator
                    # Compute guessed state 3_evap
                def objective_h3_evap(h3_evap) :
                    left = ratio_evaporators * (SC2.state_1.h - SC2.state_10.h) / (h3_evap - SC2.state_8.h) * h3_evap + SC2.state_3_comp.h
                    right = (1 + ratio_evaporators * (SC2.state_1.h - SC2.state_10.h) / (h3_evap - SC2.state_8.h)) * SC2.state_3.h 
                    return left - right
                
                h3_evap_guess = (SC2.state_8.h + SC2.state_3.h) / 2
                h3_evap = fsolve(objective_h3_evap, h3_evap_guess)[0]
                SC2.state_3_evap = State(HEOS_working_fluid, h=h3_evap, p=p3_guess)

                
                SC2.Evaporator_MT = HEX_Design(states_in=[SC2.state_8, SC2.state_3_prime], states_out=[SC2.state_3_evap, SC2.state_4_prime], name="Evaporator_MT", mode="Non-Dimensional")
                Tpinch_real = SC2.Evaporator_MT.Compute_Pinch()
                res_evap_MT = Tpinch_real - T_pinch

                # STEP 4 : Compute the residual for the condenser

                SC2.Condenser = HEX_Design(states_in=[SC2.state_5_prime, SC2.state_5], states_out=[SC2.state_6_prime, SC2.state_7], name="Condenser", mode="Non-Dimensional")
                Tpinch_real = SC2.Condenser.Compute_Pinch()
                res_cond = Tpinch_real - T_pinch

                # STEP 5 : Assemble the residuals

                residuals = np.array([res_evap_LT, res_evap_MT, res_cond])
                return residuals


            def objective_function(optimization_vars) :

                # Unpack optimization variables
                T_sub_current = optimization_vars[0]
                T_sup_current_1 = optimization_vars[1]
                T_sup_current_3 = optimization_vars[2]

                # Initial guesses for pressures
                p1_guess = 5e5 ; p3_guess = 15e5 ; p5_guess = 30e5
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
                Delta_h_Condenser = SC2.state_5.h - SC2.state_7.h
                SC2.mdot_wf_top = Q / Delta_h_Condenser
                SC2.P_comp_top = SC2.Compressor_2.Solve(p_ex=p5_solution, state_in=SC2.state_3, mdot_wf=SC2.mdot_wf_top, mode="Dimensional")[0]

                SC2.mdot_wf_bottom = SC2.mdot_wf_top / (1 + ratio_evaporators * (SC2.state_1.h - SC2.state_10.h) / (SC2.state_3_evap.h - SC2.state_8.h))
                SC2.P_comp_bottom = SC2.Compressor_1.Solve(p_ex=p3_solution, state_in=SC2.state_1, mdot_wf=SC2.mdot_wf_bottom, mode="Dimensional")[0]
                COP = Q / (SC2.P_comp_top + SC2.P_comp_bottom)

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

            # Recompte the best cycle states (for safety)
            p1_guess = 5e5 ; p3_guess = 15e5 ; p5_guess = 30e5
            p_guess = np.array([p1_guess, p3_guess, p5_guess])
            p_best = fsolve(iterative_process, p_guess, args=(T_sub_best, T_sup_best_1, T_sup_best_3))
            p1_best = p_best[0]
            p3_best = p_best[1]
            p5_best = p_best[2]

            # Compute heat exchangers and compressor with dimensional mode
            Delta_h_Condenser = SC2.state_5.h - SC2.state_7.h
            SC2.mdot_wf_top = Q / Delta_h_Condenser
            SC2.P_comp_top = SC2.Compressor_2.Solve(p_ex=p5_best, state_in=SC2.state_3, mdot_wf=SC2.mdot_wf_top, mode="Dimensional")[0]
            SC2.mdot_wf_bottom = SC2.mdot_wf_top / (1 + ratio_evaporators * (SC2.state_1.h - SC2.state_10.h) / (SC2.state_3_evap.h - SC2.state_8.h))
            SC2.P_comp_bottom = SC2.Compressor_1.Solve(p_ex=p3_best, state_in=SC2.state_1, mdot_wf=SC2.mdot_wf_bottom, mode="Dimensional")[0]
            SC2.Evaporator_LT = HEX_Design(states_in=[SC2.state_10, SC2.state_1_prime], states_out=[SC2.state_1, SC2.state_2_prime], mdot = [SC2.mdot_wf_bottom, None], name="Evaporator_LT", mode="Dimensional")
            T_pinch_evap_LT = SC2.Evaporator_LT.Compute_Pinch()
            SC2.mdot_LT = SC2.Evaporator_LT.mdot_h
            SC2.Evaporator_MT = HEX_Design(states_in=[SC2.state_8, SC2.state_3_prime], states_out=[SC2.state_3_evap, SC2.state_4_prime],mdot = [SC2.mdot_wf_top - SC2.mdot_wf_bottom, None], name="Evaporator_MT", mode="Dimensional")
            T_pinch_evap_MT = SC2.Evaporator_MT.Compute_Pinch()
            SC2.mdot_MT = SC2.Evaporator_MT.mdot_h
            SC2.Condenser = HEX_Design(states_in=[SC2.state_5_prime, SC2.state_5], states_out=[SC2.state_6_prime, SC2.state_7], mdot=[None, SC2.mdot_wf_top], name="Condenser", mode="Dimensional")
            T_pinch_cond = SC2.Condenser.Compute_Pinch()
            SC2.mdot_HT = SC2.Condenser.mdot_c

            # Compute cycle performance
            SC2.COP = SC2.Condenser.Q / (SC2.P_comp_top + SC2.P_comp_bottom)
            COP_values[i, j, k] = SC2.COP

            # Raise error if pinch points are not satisfied
            if not (np.isclose(T_pinch_evap_LT, T_pinch, atol=1e-4) and np.isclose(T_pinch_evap_MT, T_pinch, atol=1e-4) and np.isclose(T_pinch_cond, T_pinch, atol=1e-4)):
                raise ValueError("Pinch point constraints not satisfied in the best cycle found.")

            ############################################################
            # Compute heat exchanger areas
            ############################################################

            SC2.Evaporator_LT.Compute_Area()
            SC2.Evaporator_MT.Compute_Area()
            SC2.Condenser.Compute_Area()

            Evaporator_LT_areas[i, j] = SC2.Evaporator_LT.A
            Evaporator_MT_areas[i, j] = SC2.Evaporator_MT.A
            Condenser_areas[i, j] = SC2.Condenser.A


############################################################
# Compute the statistics of the results
############################################################

Evaporators_LT_areas_flat = Evaporator_LT_areas.flatten()
Evaporators_MT_areas_flat = Evaporator_MT_areas.flatten()
Condenser_areas_flat = Condenser_areas.flatten()

# Compute the mean values
mean_evaporator_LT_area = np.mean(Evaporators_LT_areas_flat)
mean_evaporator_MT_area = np.mean(Evaporators_MT_areas_flat)
mean_condenser_area = np.mean(Condenser_areas_flat)

# Compute the standard deviations
std_evaporator_LT_area = np.std(Evaporators_LT_areas_flat)
std_evaporator_MT_area = np.std(Evaporators_MT_areas_flat)
std_condenser_area = np.std(Condenser_areas_flat)


############################################################
# Print the results
############################################################

if verbose:
    print("\nCOP Values:")
    print(COP_values)
    print("\nEvaporator LT Areas (m²):")
    print(Evaporator_LT_areas)
    print("\nEvaporator MT Areas (m²):")
    print(Evaporator_MT_areas)
    print("\nCondenser Areas (m²):")
    print(Condenser_areas)

print("\n\nStatistics of Evaporator LT Areas:")
print(f"  - Mean: {np.mean(Evaporators_LT_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Evaporators_LT_areas_flat):.4f} m², which is {100 * np.std(Evaporators_LT_areas_flat) / np.mean(Evaporators_LT_areas_flat):.2f}% of the mean")

print("\n\nStatistics of Evaporator MT Areas:")
print(f"  - Mean: {np.mean(Evaporators_MT_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Evaporators_MT_areas_flat):.4f} m², which is {100 * np.std(Evaporators_MT_areas_flat) / np.mean(Evaporators_MT_areas_flat):.2f}% of the mean")

print("\n\nStatistics of Condenser Areas:")
print(f"  - Mean: {np.mean(Condenser_areas_flat):.4f} m²")
print(f"  - Standard Deviation: {np.std(Condenser_areas_flat):.4f} m², which is {100 * np.std(Condenser_areas_flat) / np.mean(Condenser_areas_flat):.2f}% of the mean")

# Save prints to a text file
output_file = Path(__file__).parent.parent / "Figures" / SC2.name / f"{SC2.name}_statistics.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    f.write(f"Temperature ranges:\n")
    f.write(f"  - T1' range: {T1_prime_min - 273.15:.1f} C to {T1_prime_max - 273.15:.1f} C ({len(T1_prime_range)} values)\n")
    f.write(f"  - T3' range: {T3_prime_min - 273.15:.1f} C to {T3_prime_max - 273.15:.1f} C ({len(T3_prime_range)} values)\n")
    f.write(f"  - T5' range: {T5_prime_min - 273.15:.1f} C to {T5_prime_max - 273.15:.1f} C ({len(T5_prime_range)} values)\n")
    f.write(f"  - Total number of operating conditions: {len(T1_prime_range) * len(T3_prime_range) * len(T5_prime_range)}\n\n")
    f.write("Evaporator LT Areas:\n")
    f.write(f"  - Mean: {np.mean(Evaporators_LT_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Evaporators_LT_areas_flat):.4f} m2, which is {100 * np.std(Evaporators_LT_areas_flat) / np.mean(Evaporators_LT_areas_flat):.2f}% of the mean\n\n")
    f.write("Evaporator MT Areas:\n")
    f.write(f"  - Mean: {np.mean(Evaporators_MT_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Evaporators_MT_areas_flat):.4f} m2, which is {100 * np.std(Evaporators_MT_areas_flat) / np.mean(Evaporators_MT_areas_flat):.2f}% of the mean\n\n")
    f.write("Condenser Areas:\n")
    f.write(f"  - Mean: {np.mean(Condenser_areas_flat):.4f} m2\n")
    f.write(f"  - Standard Deviation: {np.std(Condenser_areas_flat):.4f} m2, which is {100 * np.std(Condenser_areas_flat) / np.mean(Condenser_areas_flat):.2f}% of the mean\n")
    

############################################################
# Plot the results
############################################################

## 1. Plot the evaporator LT areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Evaporators_LT_areas_flat, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel(r'$Evaporator$ $LT$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()

## 2. Plot the evaporator MT areas distribution as a histogram
fig, ax = plt.subplots()
ax.hist(Evaporators_MT_areas_flat, bins=12, edgecolor='black', alpha=0.7, color='seagreen')
ax.set_xlabel(r'$Evaporator$ $MT$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()

## 3. Plot the condenser areas distribution as a histogram

fig, ax = plt.subplots()
ax.hist(Condenser_areas_flat, bins=12, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel(r'$Condenser$ $HEX$ $Area$  $[m^2]$')
ax.set_ylabel(r'$Frequency$')
plt.tight_layout()
plt.show()