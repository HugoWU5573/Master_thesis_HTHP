
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
T_pinch = 3                     # Minimum temperature difference in heat exchangers [K]
eta_v = 0.8                     # Volumetric efficiency
eta_is_max = 0.7                # Maximum isentropic efficiency
eta_elme = 0.95                 # Electrical-mechanical efficiency

# Cycle parameters
working_fluid = 'R290'          # Working fluid
P_comp = 10e3                   # Compressor power [W]

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

# Optimization parameters

if rapid_optimization :
    nb_points_1 = 8
    nb_points_2 = 11
else :
    nb_points_1 = 71
    nb_points_2 = 51

T_sup = np.linspace(1, 8, nb_points_1)      # Superheating at the compressor inlet [K]
T_7 = np.linspace(335, 340, nb_points_2)    # Subcooling at the condenser outlet [K]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids
HEOS_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState("HEOS", working_fluid)

# Cycle with its fixed states and mass flow rates
TC1 = Cycle("TC1")
TC1.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC1.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p3_prime)
TC1.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC1.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p5_prime)

# Compressor
TC1.P_comp_top = P_comp
TC1.Compressor = Compressor_2_param(cycle=TC1, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
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
    TC1.mdot_wf_top, T_5 = TC1.Compressor.Solve(P_el=P_comp, p_ex=p5_guess, state_in=TC1.state_3)
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


# Initial guesses
p3_guess = 10e5 ; p5_guess = 45e5
p_guess = np.array([p3_guess, p5_guess])

# Compute the solution for each combination of (T_sup, T_7)
p_solution = np.zeros((len(T_sup), len(T_7), 2))
COP_matrix = np.zeros((len(T_sup), len(T_7)))

for i in range(len(T_sup)) :
    print(f"Solving for T_sup = {T_sup[i]:.2f} K ({i+1}/{len(T_sup)})")
    for j in range(len(T_7)) :

        T_sup_current = T_sup[i]
        T_7_current = T_7[j]

        try :
            p_solution[i,j, :] = fsolve(iterative_process, p_guess, args=(T_sup_current, T_7_current))
        except :
            p_solution[i,j, :] = np.array([np.nan, np.nan])
            COP_matrix[i,j] = np.nan
            continue

        # Detect unphysical solutions and set COP to NaN
        if TC1.GasCooler.Tpinch - T_pinch < -1e-4 :
            COP = np.nan
        else :
            Q_gas_cooler = TC1.mdot_wf_top * (TC1.state_5.h - TC1.state_7.h)
            COP = Q_gas_cooler / P_comp
        
        COP_matrix[i,j] = COP

# Determine the best cycle

# Select best index ignoring NaNs
if np.all(np.isnan(COP_matrix)):
    raise ValueError("COP_matrix contains only NaNs; cannot determine best cycle.")
flat_idx = np.nanargmax(COP_matrix)
best_index = np.unravel_index(flat_idx, COP_matrix.shape)
T_sup_best = T_sup[best_index[0]]
T_7_best = T_7[best_index[1]]
p3_best = p_solution[best_index][0]
p5_best = p_solution[best_index][1]

print("\nBest cycle found with parameters :")
print(f"  - Superheating at compressor inlet : {T_sup_best:.2f} K")
print(f"  - Outlet temperature of gas cooler : {T_7_best:.0f} K")

# Recompute the best cycle states
iterative_process(np.array([p3_best, p5_best]), T_sup_best, T_7_best)

# Compute heat exchangers with dimensional mode
TC1.Evaporator = HEX_Design(states_in=[TC1.state_8, TC1.state_3_prime], states_out=[TC1.state_3, TC1.state_4_prime], mdot=[TC1.mdot_wf_top, None], name="Evaporator", mode="Dimensional")
TC1.Evaporator.Compute_Pinch()
TC1.mdot_MT = TC1.Evaporator.mdot_h
TC1.GasCooler = HEX_Design(states_in=[TC1.state_5_prime, TC1.state_5], states_out=[TC1.state_6_prime, TC1.state_7], mdot=[None, TC1.mdot_wf_top], name="Gas Cooler", mode="Dimensional")
TC1.GasCooler.Compute_Pinch()
TC1.mdot_HT = TC1.GasCooler.mdot_c

# Compute cycle performance
TC1.COP = TC1.GasCooler.Q / P_comp

# Limit the highest pressure of the cycle to 50 bars
if TC1.state_5.p > 5e6 :
    raise ValueError("The highest pressure of the cycle exceeds 50 bars. Please adjust the input parameters.")

############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
TC1.transforms = [Transform('comp', '3', '5', TC1.Compressor), 
                  Transform('hex', '5', '7',TC1.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'), 
                  Transform('adex', '7', '8', None), 
                  Transform('hex', '8', '3', TC1.Evaporator, label_in_secondary='3_prime', label_out_secondary='4_prime')]

# Plot T-s diagram with saturation curve
TC1.Ts_diagram(n=100, plot=True)
# Plot p-h diagram with saturation curve
TC1.ph_diagram(n=100, plot=True)

if full_details and not rapid_optimization:   # Full details only available for non-rapid (i.e. full) optimization

    # Plot energy and exergy charts
    TC1.energy_chart(plot=True)
    TC1.exergy_chart(T0 = 293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    TC1.Evaporator._plot(save=True, name_cycle=TC1.name, plot=True)
    TC1.GasCooler._plot(save=True, name_cycle=TC1.name, plot=True)

    # Plot the results of the optimization as a color map
    X, Y = np.meshgrid(T_sup, T_7)
    Z = COP_matrix.T   

    fig, ax = plt.subplots(figsize=(8, 6))
    # Mask NaNs so they plot as the "bad" color (white)
    masked_Z = np.ma.masked_invalid(Z)

    # Create a colormap and set NaNs to white
    try:
        cmap = matplotlib.colormaps.get_cmap('jet').copy()
    except Exception:
        cmap = matplotlib.colormaps.get_cmap('jet')
    cmap.set_bad('white')

    # Determine vmin/vmax from non-NaN values (handle all-NaN or vmin==vmax cases)
    if np.all(masked_Z.mask):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6

    contour = ax.contourf(X, Y, masked_Z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical')
    cbar.set_label(r'$COP$ [-]', rotation=0, labelpad=50, fontsize=12, loc='top')

    cbar_ticks = np.array([vmin, (vmin + vmax) / 2.0, vmax])
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])

    ax.set_xlabel(r'$T_{sup}$ [K]', labelpad=10, fontsize=12)
    ax.set_ylabel(r'$T_{7}$ [K]', rotation=0, labelpad=30, fontsize=12)

    # Mark best point
    ax.scatter(T_sup_best, T_7_best, color='black', marker='x', s=100, linewidths=2, label=f'Best COP = {TC1.COP:.2f}', clip_on=False)

    # Mark the physical limit on T7 due to pinch point constraint
    T_7_min = T5_prime + T_pinch
    ax.axhline(y=T_7_min, color='black', linestyle='--', linewidth=2, label=r'$T_{7,min}$ (physical limit due to pinch)')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("code/Figures/" + TC1.name + "/" + TC1.name + "_optimization.png", dpi=600)
    plt.show()


############################################################
# Print the results
############################################################

print(TC1)

if full_details and not rapid_optimization:   # Full details only available for non-rapid (i.e. full) optimization
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