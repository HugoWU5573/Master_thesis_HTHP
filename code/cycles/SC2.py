
""" TO BE MODIFIED + OPTIMIZED WITH NEW HEX MODEL """

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
Q = 30e3                        # Power output at the condenser [W]
ratio_evaporators = 0.5         # Ratio of power between the two evaporators

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
T5_prime = 60 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_HT = 5                    # Temperature glide in the gas cooler [K]
T6_prime = T5_prime + glide_HT  # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Optimization parameters

if rapid_optimization :
    nb_points = 8
else :
    nb_points = 71

T_sub = np.linspace(1, 8, nb_points)      # Subcooling at the condenser outlet [K]
T_sup_1 = np.linspace(1, 8, nb_points)      # Superheating at the compressor inlet [K]
T_sup_3 = np.linspace(1, 8, nb_points)  # Superheating at the second evaporator outlet [K]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids
HEOS_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
HEOS_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState("HEOS", working_fluid)

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


############################################################
# Solve the cycle to determine the unknown states
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


# Initial guesses
p1_guess = 5e5 ; p3_guess = 15e5 ; p5_guess = 30e5
p_guess = np.array([p1_guess, p3_guess, p5_guess])

# Compute the solution for each combination of (T_sub, T_sup_1, T_sup_3)
p_solution = np.zeros((len(T_sub), len(T_sup_1), len(T_sup_3), 3))
COP_matrix = np.zeros((len(T_sub), len(T_sup_1), len(T_sup_3)))

for i in range(len(T_sub)) :
    print(f"Solving for T_sub = {T_sub[i]:.2f} K ({i+1}/{len(T_sub)})")
    for j in range(len(T_sup_1)) :
        for k in range(len(T_sup_3)) :

            T_sub_current = T_sub[i]
            T_sup_current_1 = T_sup_1[j]
            T_sup_current_3 = T_sup_3[k]

            # Find the pressures that satisfy the pinch constraints
            p_solution[i,j,k, :] = least_squares(iterative_process, p_guess, bounds=([1e5, 10e5, 20e5], [10e5, 20e5, 40e5]), args=(T_sub_current, T_sup_current_1, T_sup_current_3), xtol=1e-6).x
            p3_solution = p_solution[i,j,k,1]
            p5_solution = p_solution[i,j,k,2]
            # Compute the COP for the current cycle
            Delta_h_Condenser = SC2.state_5.h - SC2.state_7.h
            SC2.mdot_wf_top = Q / Delta_h_Condenser
            SC2.P_comp_top = SC2.Compressor_2.Solve(p_ex=p5_solution, state_in=SC2.state_3, mdot_wf=SC2.mdot_wf_top, mode="Dimensional")[0]

            SC2.mdot_wf_bottom = SC2.mdot_wf_top / (1 + ratio_evaporators * (SC2.state_1.h - SC2.state_10.h) / (SC2.state_3_evap.h - SC2.state_8.h))
            SC2.P_comp_bottom = SC2.Compressor_1.Solve(p_ex=p3_solution, state_in=SC2.state_1, mdot_wf=SC2.mdot_wf_bottom, mode="Dimensional")[0]
            COP = Q / (SC2.P_comp_top + SC2.P_comp_bottom)
            COP_matrix[i,j,k] = COP

best_index = np.unravel_index(np.argmax(COP_matrix, axis=None), COP_matrix.shape)
T_sub_best = T_sub[best_index[0]]
T_sup_best_1 = T_sup_1[best_index[1]]
T_sup_best_3 = T_sup_3[best_index[2]]
p1_best = p_solution[best_index][0]
p3_best = p_solution[best_index][1]
p5_best = p_solution[best_index][2]

print("\nBest cycle found with parameters :")
print(f"  - Subcooling at condenser outlet : {T_sub_best:.2f} K")
print(f"  - Superheating at compressor inlet 1 : {T_sup_best_1:.2f} K")
print(f"  - Superheating at compressor inlet 3 : {T_sup_best_3:.2f} K")

# Recompute the cycle with the best parameters
iterative_process(np.array([p1_best, p3_best, p5_best]), T_sub_best, T_sup_best_1, T_sup_best_3)

# Compute heat exchangers and compressor with dimensional mode
Delta_h_Condenser = SC2.state_5.h - SC2.state_7.h
SC2.mdot_wf_top = Q / Delta_h_Condenser
SC2.P_comp_top = SC2.Compressor_2.Solve(p_ex=p5_best, state_in=SC2.state_3, mdot_wf=SC2.mdot_wf_top, mode="Dimensional")[0]
SC2.mdot_wf_bottom = SC2.mdot_wf_top / (1 + ratio_evaporators * (SC2.state_1.h - SC2.state_10.h) / (SC2.state_3_evap.h - SC2.state_8.h))
SC2.P_comp_bottom = SC2.Compressor_1.Solve(p_ex=p3_best, state_in=SC2.state_1, mdot_wf=SC2.mdot_wf_bottom, mode="Dimensional")[0]
SC2.Evaporator_LT = HEX_Design(states_in=[SC2.state_10, SC2.state_1_prime], states_out=[SC2.state_1, SC2.state_2_prime], mdot = [SC2.mdot_wf_bottom, None], name="Evaporator_LT", mode="Dimensional")
SC2.Evaporator_LT.Compute_Pinch()
SC2.mdot_LT = SC2.Evaporator_LT.mdot_h
SC2.Evaporator_MT = HEX_Design(states_in=[SC2.state_8, SC2.state_3_prime], states_out=[SC2.state_3_evap, SC2.state_4_prime],mdot = [SC2.mdot_wf_top - SC2.mdot_wf_bottom, None], name="Evaporator_MT", mode="Dimensional")
SC2.Evaporator_MT.Compute_Pinch()
SC2.mdot_MT = SC2.Evaporator_MT.mdot_h
SC2.Condenser = HEX_Design(states_in=[SC2.state_5_prime, SC2.state_5], states_out=[SC2.state_6_prime, SC2.state_7], mdot=[None, SC2.mdot_wf_top], name="Condenser", mode="Dimensional")
SC2.Condenser.Compute_Pinch()
SC2.mdot_HT = SC2.Condenser.mdot_c

# Compute cycle performance
SC2.COP = SC2.Condenser.Q / (SC2.P_comp_top + SC2.P_comp_bottom)
print(f"  - Best cycle COP : {SC2.COP:.2f}")
print(f"  - Compressor power : {(SC2.P_comp_top + SC2.P_comp_bottom)/1e3:.2f} kW")

############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
SC2.transforms = [Transform('isobaric_mixing', '3_comp', '3_evap', None),
                  Transform('comp', '1', '3_comp', SC2.Compressor_1),
                  Transform('comp', '3', '5', SC2.Compressor_2),
                  Transform('hex', '5', '7', SC2.Condenser, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                  Transform('adex', '7', '8', None),
                  Transform('adex', '9', '10', None),
                  Transform('hex', '8', '3_evap',SC2.Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                  Transform('hex', '10', '1',SC2.Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime')]

# Plot T-s diagram with saturation curve
SC2.Ts_diagram(n=100, plot=True)
SC2.ph_diagram(n=100, plot=True)

if full_details :

    # Plot energy and exergy charts
    SC2.energy_chart(plot=True)
    SC2.exergy_chart(T0=293.15, p0 = 1e5, plot=True)
    
    # Plot heat exchangers diagrams
    SC2.Evaporator_LT._plot(save=True, name_cycle=SC2.name, plot=True)
    SC2.Evaporator_MT._plot(save=True, name_cycle=SC2.name, plot=True)
    SC2.Condenser._plot(save=True, name_cycle=SC2.name, plot=True)


############################################################
# Print the results
############################################################

print(SC2)

if full_details :
    SC2.Evaporator_LT.Compute_Area()
    SC2.Evaporator_MT.Compute_Area()
    SC2.Condenser.Compute_Area()
    print(SC2.Evaporator_LT)
    print(SC2.Evaporator_MT)
    print(SC2.Condenser)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / SC2.name / f"{SC2.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(SC2) + '\n')
        f.write('\n' + str(SC2.Evaporator_LT) + '\n')
        f.write('\n' + str(SC2.Evaporator_MT) + '\n')
        f.write('\n' + str(SC2.Condenser) + '\n')