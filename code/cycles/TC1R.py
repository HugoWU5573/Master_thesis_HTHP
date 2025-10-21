
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


############################################################
# Parameters
############################################################

# Technological parameters
T_pinch = 3                     # Minimum temperature difference in heat exchangers [K]
glide = 55                      # Temperature glide in the gas cooler [K]
T_sup = 3                       # Superheating at the compressor inlet [K]
eta_v = 0.8                     # Volumetric efficiency
eta_is_max = 0.7                # Maximum isentropic efficiency
eta_elme = 0.95                 # Electrical-mechanical efficiency

# Cycle parameters
working_fluid = 'R290'          # Working fluid
P_comp = 7.5e3                  # Compressor power [W]

# Heat source parameters
external_fluid_MT = 'Water'     # External fluid in the heat source
mdot_MT = 0.5                   # Mass flow rate of external fluid the heat source [kg/s]
T3_prime = 40 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
mdot_HT = 0.1                   # Mass flow rate of external fluid the heat sink [kg/s]
T5_prime = 60 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
T6_prime = T5_prime + glide     # Outlet temperature of the external fluid in the heat sink [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]
p6_prime = p5_prime             # Outlet pressure of the external fluid in the heat sink [Pa]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids
HEOS_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState("HEOS", working_fluid)

# Cycle with its fixed states and mass flow rates
TC1R = Cycle("TC1R")
TC1R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC1R.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC1R.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p6_prime)
TC1R.mdot_MT = mdot_MT
TC1R.mdot_HT = mdot_HT

# Compressor
TC1R.P_comp = P_comp
TC1R.Compressor = Compressor_2_param(cycle=TC1R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
############################################################

def iterative_process(p_gess) :
    p3_guess = p_gess[0] ; p5_guess = p_gess[1]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 3
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    TC1R.state_3 = State(HEOS_working_fluid, T=Tsat_3, Q = 1)

        # Compute guessed state 4
    TC1R.state_4 = State(HEOS_working_fluid, T=TC1R.state_3.T + T_sup, p=p3_guess)

        # Compute guessed state 5
    TC1R.mdot_wf, T_5 = TC1R.Compressor.Solve(P_el=P_comp, p_ex=p5_guess, state_in=TC1R.state_4)
    TC1R.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

    # STEP 2 : Compute the residual for the gas cooler

    TC1R.GasCooler = HEX_Design(states_in=[TC1R.state_5_prime, TC1R.state_5], states_out=[TC1R.state_6_prime, None], mdot=[TC1R.mdot_HT, TC1R.mdot_wf], name="Gas Cooler")
    Tpinch_real = TC1R.GasCooler.Compute_Pinch()
    TC1R.state_6 = TC1R.GasCooler.state_out_h
    res_gas_cooler = Tpinch_real - T_pinch

    # STEP 3 : Compute states 7 and 8

    TC1R.Recuperator = HEX_Design(states_in=[TC1R.state_3, TC1R.state_6], states_out=[TC1R.state_4, None], mdot=[TC1R.mdot_wf, TC1R.mdot_wf], name="Recuperator")
    TC1R.Recuperator.Compute_Pinch()
    TC1R.state_7 = TC1R.Recuperator.state_out_h

    h8 = TC1R.state_7.h
    TC1R.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    # STEP 4 : Compute the residual for the evaporator

    TC1R.Evaporator = HEX_Design(states_in=[TC1R.state_8, TC1R.state_3_prime], states_out=[TC1R.state_3, None], mdot=[TC1R.mdot_wf, TC1R.mdot_MT], name="Evaporator")
    Tpinch_real = TC1R.Evaporator.Compute_Pinch()
    TC1R.state_4_prime = TC1R.Evaporator.state_out_h
    res_evap = Tpinch_real - T_pinch

    # STEP 5 : Assemble the residuals

    residuals = np.array([res_evap, res_gas_cooler])
    return residuals


# Initial guesses
p3_guess = 10e5 ; p5_guess = 50e5
p_guess = np.array([p3_guess, p5_guess])

# Compute the solution
fsolve(iterative_process, p_guess)

# Limit the highest pressure of the cycle to 50 bars
if TC1R.state_5.p > 5e6 :
    raise ValueError("The highest pressure of the cycle exceeds 50 bars. Please adjust the input parameters.")

TC1R.COP = TC1R.GasCooler.Q / P_comp


############################################################
# Print the results
############################################################

full_details = False

print(TC1R)

if full_details:
    print(TC1R.Evaporator)
    print(TC1R.GasCooler)
    print(TC1R.Recuperator)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / TC1R.name / f"{TC1R.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(TC1R) + '\n')
        f.write('\n' + str(TC1R.Evaporator) + '\n')
        f.write('\n' + str(TC1R.GasCooler) + '\n')
        f.write('\n' + str(TC1R.Recuperator) + '\n')


############################################################
# Plot the results
############################################################

# Define the transforms 
TC1R.transforms = [Transform('comp', '4', '5', TC1R.Compressor), Transform('cond', '5', '6',TC1R.GasCooler, label_in_secondary='5_prime', label_out_secondary='6_prime'), 
                  Transform('adex', '7', '8', None), Transform('evap', '8', '3', TC1R.Evaporator, label_in_secondary='3_prime', label_out_secondary='4_prime')]

# Plot T-s diagram with saturation curve
TC1R.Ts_diagram(n=100, plot=True)


if full_details :

    # Plot energy and exergy charts
    TC1R.energy_chart(plot=True)
    TC1R.exergy_chart(T0 = 293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    TC1R.Evaporator._plot(save=True, name_cycle=TC1R.name, plot=True)
    TC1R.GasCooler._plot(save=True, name_cycle=TC1R.name, plot=True)
    TC1R.Recuperator._plot(save=True, name_cycle=TC1R.name, plot=True)


"""

WHAT REMAINS TO BE DONE :
    - Complete the Transformations part

"""