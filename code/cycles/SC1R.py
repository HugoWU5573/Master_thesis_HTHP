
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
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt

rapid_optimization = True  # Set to True for rapid optimization with less points

############################################################
# Parameters
############################################################

# Technological parameters
T_pinch = 3                       # Minimum temperature difference in heat exchangers [K]
eta_v = 0.8                       # Volumetric efficiency
eta_is_max = 0.7                  # Maximum isentropic efficiency
eta_elme = 0.95                   # Electrical-mechanical efficiency
recuperator_effectiveness = 0.8   # Effectiveness of the recuperator

# Cycle parameters
working_fluid = 'R290'          # Working fluid
P_comp = 10e3                   # Compressor power [W]

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
    nb_points = 9
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
SC1R = Cycle("SC1R")
SC1R.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
SC1R.state_2_prime = State(HEOS_external_fluid_LT, T=T2_prime, p=p1_prime)
SC1R.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p4_prime)
SC1R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p4_prime)

# Compressor
SC1R.P_comp_bottom = P_comp
SC1R.Compressor = Compressor_2_param(cycle=SC1R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
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
    SC1R.mdot_wf_bottom, T_3 = SC1R.Compressor.Solve(P_el=P_comp, p_ex=p3_guess, state_in=SC1R.state_2)
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

        p_solution[i,j, :] = fsolve(iterative_process, p_guess, args=(T_sub_current, T_sup_current))
        Q_cond = SC1R.mdot_wf_bottom * (SC1R.state_3.h - SC1R.state_8.h)
        COP = Q_cond / P_comp
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

# Compute heat exchangers with dimensional mode
SC1R.Evaporator = HEX_Design(states_in=[SC1R.state_10, SC1R.state_1_prime], states_out=[SC1R.state_1, SC1R.state_2_prime], mdot=[SC1R.mdot_wf_bottom, None], name="Evaporator", mode="Dimensional")
SC1R.Evaporator.Compute_Pinch()
SC1R.mdot_LT = SC1R.Evaporator.mdot_h
SC1R.Condenser = HEX_Design(states_in=[SC1R.state_4_prime, SC1R.state_3], states_out=[SC1R.state_3_prime, SC1R.state_8], mdot=[None, SC1R.mdot_wf_bottom], name="Condenser", mode="Dimensional")
SC1R.Condenser.Compute_Pinch()
SC1R.mdot_MT = SC1R.Condenser.mdot_c
SC1R.Recuperator = HEX_Design(states_in=[SC1R.state_1, SC1R.state_8], states_out=[SC1R.state_2, SC1R.state_9], mdot=[SC1R.mdot_wf_bottom, SC1R.mdot_wf_bottom], name = 'Recuperator', mode="Dimensional", type="Recuperator",  epsilon=recuperator_effectiveness)
SC1R.Recuperator.Solve_Recuperator()

# Compute cycle performance
SC1R.COP = SC1R.Condenser.Q / P_comp


############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
SC1R.transforms = [Transform('comp', '2', '3', SC1R.Compressor), 
                  Transform('hex', '10', '1',SC1R.Evaporator, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                  Transform('adex', '9', '10', None), 
                  Transform('hex', '3', '8', SC1R.Condenser, label_in_secondary='4_prime', label_out_secondary='3_prime'),
                  Transform('hex', '1', '2', SC1R.Recuperator, label_in_secondary='8', label_out_secondary='9')]

# Plot T-s diagram with saturation curve
SC1R.Ts_diagram(n=100, plot=True)

if full_details and not rapid_optimization:  # Full details only available for non-rapid (i.e. full) optimization

    # Plot energy and exergy charts
    SC1R.energy_chart(plot=True)
    SC1R.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    SC1R.Evaporator._plot(save=True, name_cycle=SC1R.name, plot=True)
    SC1R.Condenser._plot(save=True, name_cycle=SC1R.name, plot=True)
    SC1R.Recuperator._plot(save=True, name_cycle=SC1R.name, plot=True)

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
    ax.scatter(T_sub_best, T_sup_best, color='black', marker='x', s=100, linewidths=2, label=f'Best COP = {SC1R.COP:.2f}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("code/Figures/" + SC1R.name + "/" + SC1R.name + "_optimization.png", dpi=600)
    plt.show()


############################################################
# Print the results
############################################################

print(SC1R)

if full_details and not rapid_optimization:  # Full details only available for non-rapid (i.e. full) optimization
    SC1R.Evaporator.Compute_Area()
    SC1R.Condenser.Compute_Area()
    SC1R.Recuperator.Compute_Area()
    print(SC1R.Evaporator)
    print(SC1R.Condenser)
    print(SC1R.Recuperator)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / SC1R.name / f"{SC1R.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(SC1R) + '\n')
        f.write('\n' + str(SC1R.Evaporator) + '\n')
        f.write('\n' + str(SC1R.Condenser) + '\n')
        f.write('\n' + str(SC1R.Recuperator) + '\n')
