
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
import matplotlib
import matplotlib.pyplot as plt

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

# Heat source parameters
external_fluid_LT = 'Water'     # External fluid in the heat source
T1_prime = 15 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
glide_LT = 5                    # Temperature glide of the external fluid in the heat source [K]
T2_prime = T1_prime - glide_LT  # Outlet temperature of the external fluid in the heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_MT = 'Water'     # External fluid in the heat sink
T4_prime = 40 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
glide_MT = 5                    # Temperature glide of the external fluid in the heat sink [K]
T3_prime = T4_prime + glide_MT  # Outlet temperature of the external fluid in the heat sink [K]
p4_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]

# Optimization parameters

if rapid_optimization :
    nb_points = 8
else :
    nb_points = 71

T_sub = np.linspace(1, 8, nb_points)      # Subcooling at the condenser outlet [K]
T_sup = np.linspace(1, 8, nb_points)      # Superheating at the compressor inlet [K]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids
HEOS_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
HEOS_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
HEOS_working_fluid = CoolProp.AbstractState("HEOS", working_fluid)

# Cycle with its fixed states and mass flow rates
SC1 = Cycle("SC1")
SC1.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
SC1.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
SC1.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p4_prime)
SC1.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p4_prime)

# Compressor
SC1.Compressor = Compressor_2_param(cycle=SC1, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
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


# Initial guesses
p1_guess = 5e5 ; p3_guess = 20e5
p_guess = np.array([p1_guess, p3_guess])

# Compute the solution for each combination of (T_sub, T_sup)
p_solution = np.zeros((len(T_sub), len(T_sup), 2))
COP_matrix = np.zeros((len(T_sub), len(T_sup)))

for i in range(len(T_sub)) :
    print(f"Solving for T_sub = {T_sub[i]:.2f} K ({i+1}/{len(T_sub)})")
    for j in range(len(T_sup)) :

        T_sub_current = T_sub[i]
        T_sup_current = T_sup[j]

        # Find the pressures that satisfy the pinch constraints
        p_solution[i,j, :] = fsolve(iterative_process, p_guess, args=(T_sub_current, T_sup_current))
        # p_solution[i,j, :] = least_squares(iterative_process, p_guess, bounds=([1e5, 10e5], [10e5, 40e5]), args=(T_sub_current, T_sup_current), xtol=1e-6).x
        p3_solution = p_solution[i,j,1]

        # Compute the COP for the current cycle
        Delta_h_Condenser = SC1.state_3.h - SC1.state_9.h
        SC1.mdot_wf_bottom = Q / Delta_h_Condenser
        SC1.P_comp_bottom = SC1.Compressor.Solve(p_ex=p3_solution, state_in=SC1.state_1, mdot_wf=SC1.mdot_wf_bottom, mode="Dimensional")[0]
        COP = Q / SC1.P_comp_bottom
        COP_matrix[i,j] = COP

# Determine the best cycle
best_index = np.unravel_index(np.argmax(COP_matrix, axis=None), COP_matrix.shape)
T_sub_best = T_sub[best_index[0]]
T_sup_best = T_sup[best_index[1]]
p1_best = p_solution[best_index][0]
p3_best = p_solution[best_index][1]

print("\nBest cycle found with parameters :")
print(f"  - Subcooling at condenser outlet : {T_sub_best:.2f} K")
print(f"  - Superheating at compressor inlet : {T_sup_best:.2f} K")

# Recompute the best cycle states
iterative_process(np.array([p1_best, p3_best]), T_sub_best, T_sup_best)

# Compute heat exchangers and compressor with dimensional mode
Delta_h_Condenser = SC1.state_3.h - SC1.state_9.h
SC1.mdot_wf_bottom = Q / Delta_h_Condenser
SC1.P_comp_bottom = SC1.Compressor.Solve(p_ex=p3_best, state_in=SC1.state_1, mdot_wf=SC1.mdot_wf_bottom, mode="Dimensional")[0]
SC1.Evaporator = HEX_Design(states_in=[SC1.state_10, SC1.state_1_prime], states_out=[SC1.state_1, SC1.state_2_prime], mdot=[SC1.mdot_wf_bottom, None], name="Evaporator", mode="Dimensional")
SC1.Evaporator.Compute_Pinch()
SC1.mdot_LT = SC1.Evaporator.mdot_h
SC1.Condenser = HEX_Design(states_in=[SC1.state_4_prime, SC1.state_3], states_out=[SC1.state_3_prime, SC1.state_9], mdot=[None, SC1.mdot_wf_bottom], name="Condenser", mode="Dimensional")
SC1.Condenser.Compute_Pinch()
SC1.mdot_MT = SC1.Condenser.mdot_c

# Compute cycle performance
SC1.COP = SC1.Condenser.Q / SC1.P_comp_bottom
print(f"  - Best cycle COP : {SC1.COP:.2f}")
print(f"  - Compressor power : {SC1.P_comp_bottom/1e3:.2f} kW")


############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
SC1.transforms = [Transform('comp', '1', '3', SC1.Compressor), 
                  Transform('hex', '10', '1',SC1.Evaporator, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                  Transform('adex', '9', '10', None), 
                  Transform('hex', '3', '9', SC1.Condenser, label_in_secondary='4_prime', label_out_secondary='3_prime')]

# Plot T-s and p-h diagrams with saturation curve
SC1.Ts_diagram(n=100, plot=True)
SC1.ph_diagram(n=100, plot=True)

if full_details and not rapid_optimization:   # Full details only available for non-rapid (i.e. full) optimization

    # Plot energy and exergy charts
    SC1.energy_chart(plot=True)
    SC1.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    SC1.Evaporator._plot(save=True, name_cycle=SC1.name, plot=True)
    SC1.Condenser._plot(save=True, name_cycle=SC1.name, plot=True)

    # Plot the results of the optimization as a color map
    X, Y = np.meshgrid(T_sub, T_sup)
    Z = COP_matrix.T   

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=50, cmap=matplotlib.cm.jet)
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical')
    cbar.set_label(r'$COP$ [-]', rotation=0, labelpad=50, fontsize=12, loc='top')

    cbar_ticks = np.array([np.min(Z), (np.max(Z)+np.min(Z))/2, np.max(Z)])
    cbar.set_ticks(cbar_ticks, labels=[f"{tick:.1f}" for tick in cbar_ticks])

    ax.set_xlabel(r'$T_{sub}$ [K]', labelpad=10, fontsize=12)
    ax.set_ylabel(r'$T_{sup}$ [K]', rotation=0, labelpad=30, fontsize=12)

    # Mark best point
    ax.scatter(T_sub_best, T_sup_best, color='black', marker='x', s=100, linewidths=2, label=f'Best COP = {SC1.COP:.2f}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("code/Figures/" + SC1.name + "/" + SC1.name + "_optimization.png", dpi=600)
    plt.show()


############################################################
# Print the results
############################################################

print(SC1)

if full_details and not rapid_optimization:   # Full details only available for non-rapid (i.e. full) optimization
    SC1.Evaporator.Compute_Area()
    SC1.Condenser.Compute_Area()
    print(SC1.Evaporator)
    print(SC1.Condenser)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / SC1.name / f"{SC1.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(SC1) + '\n')
        f.write('\n' + str(SC1.Evaporator) + '\n')
        f.write('\n' + str(SC1.Condenser) + '\n')