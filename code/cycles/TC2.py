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
from scipy.optimize import fsolve, least_squares

rapid_optimization = True  # Set to True for rapid optimization with less points

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
Q = 30e3                        # Power output at the GasCooler [W]
ratio_evaporators = 1           # Ratio of power between the two evaporators

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
T5_prime = 60 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_HT = 55                   # Temperature glide in the gas cooler [K]
T6_prime = T5_prime + glide_HT  # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Optimization parameters

if rapid_optimization :
    nb_points = 8
else :
    nb_points = 15

T_7 = np.linspace(336.15, 340, nb_points)    # GasCooler outlet temperature [K]
T_sup_1 = np.linspace(1, 8, nb_points)       # Superheating at the compressor inlet [K]
T_sup_3 = np.linspace(5, 12, nb_points)      # Superheating at the second evaporator outlet [K]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids
HEOS_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
HEOS_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState("HEOS", working_fluid)

# Cycle with its fixed states and mass flow rates
TC2 = Cycle("TC2")
TC2.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
TC2.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
TC2.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC2.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
TC2.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC2.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

# Compressors
TC2.Compressor_1 = Compressor_2_param(cycle=TC2, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)
TC2.Compressor_2 = Compressor_2_param(cycle=TC2, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
############################################################

def iterative_process(p_gess, T_7_current, T_sup_current_1, T_sup_current_3) :
    p1_guess = p_gess[0] ; p3_guess = p_gess[1] ; p5_guess = p_gess[2]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 1 (first compressor inlet is superheated)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
    Tsat_1 = HEOS_working_fluid.T()
    TC2.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup_current_1, p=p1_guess)

        # Compute guessed state 3_comp (exit of first compressor)
    T_3_comp = TC2.Compressor_1.Solve(p_ex=p3_guess, state_in=TC2.state_1, mode = 'Non-Dimensional')[1]
    TC2.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

        # Compute guessed state 3 (second compressor inlet is superheated)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    TC2.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup_current_3, p=p3_guess)

        # Compute guessed state 5 (exit of second compressor)
    T_5 = TC2.Compressor_2.Solve(p_ex=p5_guess, state_in=TC2.state_3, mode = 'Non-Dimensional')[1]
    TC2.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

        # Compute guessed state 7 (GasCooler outlet is subcooled)
    TC2.state_7 = State(HEOS_working_fluid, T=T_7_current, p=p5_guess)
    TC2.state_9 = State(HEOS_working_fluid, T=T_7_current, p=p5_guess)  # States 7 and 9 are the same in this configuration

        # Compute guessed state 8
    h8 = TC2.state_7.h
    TC2.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

        # Compute guessed state 10
    h10 = TC2.state_9.h
    TC2.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

    # STEP 2 : Compute the residual for the first evaporator

    TC2.Evaporator_LT = HEX_Design(states_in=[TC2.state_10, TC2.state_1_prime], states_out=[TC2.state_1, TC2.state_2_prime], name="Evaporator_LT", mode="Non-Dimensional")
    Tpinch_real = TC2.Evaporator_LT.Compute_Pinch()
    res_evap_LT = Tpinch_real - T_pinch

    # STEP 3 : Compute the residual for the second evaporator
        # Compute guessed state 3_evap
    def objective_h3_evap(h3_evap) :
        left = ratio_evaporators * (TC2.state_1.h - TC2.state_10.h) / (h3_evap - TC2.state_8.h) * h3_evap + TC2.state_3_comp.h
        right = (1 + ratio_evaporators * (TC2.state_1.h - TC2.state_10.h) / (h3_evap - TC2.state_8.h)) * TC2.state_3.h 
        return left - right
    
    h3_evap_guess = (TC2.state_8.h + TC2.state_3.h) / 2
    h3_evap = fsolve(objective_h3_evap, h3_evap_guess)[0]
    TC2.state_3_evap = State(HEOS_working_fluid, h=h3_evap, p=p3_guess)

    
    TC2.Evaporator_MT = HEX_Design(states_in=[TC2.state_8, TC2.state_3_prime], states_out=[TC2.state_3_evap, TC2.state_4_prime], name="Evaporator_MT", mode="Non-Dimensional")
    Tpinch_real = TC2.Evaporator_MT.Compute_Pinch()
    res_evap_MT = Tpinch_real - T_pinch

    # STEP 4 : Compute the residual for the GasCooler

    TC2.GasCooler = HEX_Design(states_in=[TC2.state_5_prime, TC2.state_5], states_out=[TC2.state_6_prime, TC2.state_7], name="Gas Cooler", mode="Non-Dimensional")
    Tpinch_real = TC2.GasCooler.Compute_Pinch()
    res_cond = Tpinch_real - T_pinch

    # STEP 5 : Assemble the residuals

    residuals = np.array([res_evap_LT, res_evap_MT, res_cond])

    return residuals


# Initial guesses
p1_guess = 3e5 ; p3_guess = 10e5 ; p5_guess = 45e5
p_guess = np.array([p1_guess, p3_guess, p5_guess])

# Compute the solution for each combination of (T_sub, T_sup_1, T_sup_3)
p_solution = np.zeros((len(T_7), len(T_sup_1), len(T_sup_3), 3))
COP_matrix = np.zeros((len(T_7), len(T_sup_1), len(T_sup_3)))

for i in range(len(T_7)) :
    print(f"Solving for T_7 = {T_7[i]:.2f} K ({i+1}/{len(T_7)})")
    for j in range(len(T_sup_1)) :
        for k in range(len(T_sup_3)) :

            T_sub_current = T_7[i]
            T_sup_current_1 = T_sup_1[j]
            T_sup_current_3 = T_sup_3[k]

            # Find the pressures that satisfy the pinch constraints
            try :
                p_solution[i,j,k, :] = fsolve(iterative_process, p_guess, args=(T_sub_current, T_sup_current_1, T_sup_current_3))
                p3_solution = p_solution[i,j,k,1]
                p5_solution = p_solution[i,j,k,2]
            except :
                COP_matrix[i,j,k] = 0
                continue

            # Compute the COP for the current cycle
            Delta_h_GasCooler = TC2.state_5.h - TC2.state_7.h
            TC2.mdot_wf_top = Q / Delta_h_GasCooler
            TC2.P_comp_top = TC2.Compressor_2.Solve(p_ex=p5_solution, state_in=TC2.state_3, mdot_wf=TC2.mdot_wf_top, mode="Dimensional")[0]

            TC2.mdot_wf_bottom = TC2.mdot_wf_top / (1 + ratio_evaporators * (TC2.state_1.h - TC2.state_10.h) / (TC2.state_3_evap.h - TC2.state_8.h))
            TC2.P_comp_bottom = TC2.Compressor_1.Solve(p_ex=p3_solution, state_in=TC2.state_1, mdot_wf=TC2.mdot_wf_bottom, mode="Dimensional")[0]
            COP = Q / (TC2.P_comp_top + TC2.P_comp_bottom)
            if TC2.GasCooler.Tpinch - T_pinch < -1e-4 : COP_matrix[i,j,k] = np.nan
            else : COP_matrix[i,j,k] = COP

if np.all(np.isnan(COP_matrix)):
    raise ValueError("COP_matrix contains only NaNs; cannot determine best cycle.")
flat_idx = np.nanargmax(COP_matrix)
best_index = np.unravel_index(flat_idx, COP_matrix.shape)
T_7_best = T_7[best_index[0]]
T_sup_best_1 = T_sup_1[best_index[1]]
T_sup_best_3 = T_sup_3[best_index[2]]
p1_best = p_solution[best_index][0]
p3_best = p_solution[best_index][1]
p5_best = p_solution[best_index][2]

print("\nBest cycle found with parameters :")
print(f"  - Best value for T7 : {T_7_best:.2f} K")
print(f"  - Superheating at compressor inlet 1 : {T_sup_best_1:.2f} K")
print(f"  - Superheating at compressor inlet 3 : {T_sup_best_3:.2f} K")

print(f"  - High pressure (compressor 2 outlet) : {p5_best/1e5:.2f} bar")
print(f"  - Medium pressure (compressor 1 outlet) : {p3_best/1e5:.2f} bar")
print(f"  - Low pressure (compressor 1 inlet) : {p1_best/1e5:.2f} bar")

# Recompute the cycle with the best parameters
iterative_process(np.array([p1_best, p3_best, p5_best]), T_7_best, T_sup_best_1, T_sup_best_3)

# Compute heat exchangers and compressor with dimensional mode
Delta_h_GasCooler = TC2.state_5.h - TC2.state_7.h
TC2.mdot_wf_top = Q / Delta_h_GasCooler
TC2.P_comp_top = TC2.Compressor_2.Solve(p_ex=p5_best, state_in=TC2.state_3, mdot_wf=TC2.mdot_wf_top, mode="Dimensional")[0]
TC2.mdot_wf_bottom = TC2.mdot_wf_top / (1 + ratio_evaporators * (TC2.state_1.h - TC2.state_10.h) / (TC2.state_3_evap.h - TC2.state_8.h))
TC2.P_comp_bottom = TC2.Compressor_1.Solve(p_ex=p3_best, state_in=TC2.state_1, mdot_wf=TC2.mdot_wf_bottom, mode="Dimensional")[0]
TC2.Evaporator_LT = HEX_Design(states_in=[TC2.state_10, TC2.state_1_prime], states_out=[TC2.state_1, TC2.state_2_prime], mdot = [TC2.mdot_wf_bottom, None], name="Evaporator_LT", mode="Dimensional")
TC2.Evaporator_LT.Compute_Pinch()
TC2.mdot_LT = TC2.Evaporator_LT.mdot_h
TC2.Evaporator_MT = HEX_Design(states_in=[TC2.state_8, TC2.state_3_prime], states_out=[TC2.state_3_evap, TC2.state_4_prime],mdot = [TC2.mdot_wf_top - TC2.mdot_wf_bottom, None], name="Evaporator_MT", mode="Dimensional")
TC2.Evaporator_MT.Compute_Pinch()
TC2.mdot_MT = TC2.Evaporator_MT.mdot_h
TC2.GasCooler = HEX_Design(states_in=[TC2.state_5_prime, TC2.state_5], states_out=[TC2.state_6_prime, TC2.state_7], mdot=[None, TC2.mdot_wf_top], name="Gas Cooler", mode="Dimensional")
TC2.GasCooler.Compute_Pinch()
TC2.mdot_HT = TC2.GasCooler.mdot_c

# Compute cycle performance
TC2.COP = TC2.GasCooler.Q / (TC2.P_comp_top + TC2.P_comp_bottom)
TC2.beta = TC2.Evaporator_MT.Q / (TC2.Evaporator_LT.Q + TC2.Evaporator_MT.Q)
print(f"  - Best cycle COP : {TC2.COP:.2f}")
print(f"  - Compressor power : {(TC2.P_comp_top + TC2.P_comp_bottom)/1e3:.2f} kW")

############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
TC2.transforms = [Transform('isobaric_mixing', '3_comp', '3_evap', None),
                  Transform('comp', '1', '3_comp', TC2.Compressor_1),
                  Transform('comp', '3', '5', TC2.Compressor_2),
                  Transform('hex', '5', '7', TC2.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                  Transform('adex', '7', '8', None),
                  Transform('adex', '9', '10', None),
                  Transform('hex', '8', '3_evap',TC2.Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                  Transform('hex', '10', '1',TC2.Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime')]

# Plot T-s diagram with saturation curve
TC2.Ts_diagram(n=100, plot=True)
TC2.ph_diagram(n=100, plot=True)

if full_details and not rapid_optimization:  # Plot detailed charts only if not in rapid optimization mode

    # Plot energy and exergy charts
    TC2.energy_chart(plot=True)
    TC2.exergy_chart(T0=293.15, p0 = 1e5, plot=True)
    
    # Plot heat exchangers diagrams
    TC2.Evaporator_LT._plot(save=True, name_cycle=TC2.name, plot=True)
    TC2.Evaporator_MT._plot(save=True, name_cycle=TC2.name, plot=True)
    TC2.GasCooler._plot(save=True, name_cycle=TC2.name, plot=True)


############################################################
# Print the results
############################################################

print(TC2)

if full_details and not rapid_optimization:  # Print detailed results only if not in rapid optimization mode
    TC2.Evaporator_LT.Compute_Area()
    TC2.Evaporator_MT.Compute_Area()
    TC2.GasCooler.Compute_Area()
    print(TC2.Evaporator_LT)
    print(TC2.Evaporator_MT)
    print(TC2.GasCooler)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / TC2.name / f"{TC2.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(TC2) + '\n')
        f.write('\n' + str(TC2.Evaporator_LT) + '\n')
        f.write('\n' + str(TC2.Evaporator_MT) + '\n')
        f.write('\n' + str(TC2.GasCooler) + '\n')