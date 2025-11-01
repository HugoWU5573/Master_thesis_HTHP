
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
T_sup = 3                       # Superheating at the compressor inlet [K]
T_sub = 6                       # Subcooling at the condenser outlet [K]
eta_v = 0.8                     # Volumetric efficiency
eta_is_max = 0.7                # Maximum isentropic efficiency
eta_elme = 0.95                 # Electrical-mechanical efficiency

# Cycle parameters
working_fluid = 'R290'          # Working fluid
P_comp = 2.5e3                  # Compressor power [W]

# Heat source parameters
external_fluid_LT = 'Water'     # External fluid in the heat source
mdot_LT = 0.4                   # Mass flow rate of external fluid the heat source [kg/s]
T1_prime = 10 + 273.15          # Inlet temperature of the external fluid in the heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in the heat source [Pa]

# Heat sink parameters
external_fluid_MT = 'Water'     # External fluid in the heat sink
mdot_MT = 0.5                   # Mass flow rate of external fluid the heat sink [kg/s]
T4_prime = 35 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
p4_prime = 1e5                  # Inlet pressure of the external fluid in the heat sink [Pa]


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
SC1.state_4_prime = State(HEOS_external_fluid_MT, T=T4_prime, p=p4_prime)
SC1.mdot_LT = mdot_LT
SC1.mdot_MT = mdot_MT

# Compressor
SC1.P_comp_bottom = P_comp
SC1.Compressor = Compressor_2_param(cycle=SC1, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
############################################################

def iterative_process(p_gess) :
    p1_guess = p_gess[0] ; p3_guess = p_gess[1]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 1
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
    Tsat_1 = HEOS_working_fluid.T()
    SC1.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup, p=p1_guess)

        # Compute guessed state 3
    SC1.mdot_wf_bottom, T_3 = SC1.Compressor.Solve(P_el=P_comp, p_ex=p3_guess, state_in=SC1.state_1)
    SC1.state_3 = State(HEOS_working_fluid, T=T_3, p=p3_guess)

        # Compute guessed state 9
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_9 = HEOS_working_fluid.T()
    SC1.state_9 = State(HEOS_working_fluid, T=Tsat_9 - T_sub, p=p3_guess)

        # Compute guessed state 10
    h10 = SC1.state_9.h
    SC1.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

    # STEP 2 : Compute the residual for the evaporator

    SC1.Evaporator = HEX_Design(states_in=[SC1.state_10, SC1.state_1_prime], states_out=[SC1.state_1, None], mdot=[SC1.mdot_wf_bottom, SC1.mdot_LT], name="Evaporator")
    Tpinch_real = SC1.Evaporator.Compute_Pinch()
    SC1.state_2_prime = SC1.Evaporator.state_out_h
    res_evap = Tpinch_real - T_pinch

    # STEP 3 : Compute the residual for the condenser

    SC1.Condenser = HEX_Design(states_in=[SC1.state_4_prime, SC1.state_3], states_out=[None, SC1.state_9], mdot=[SC1.mdot_MT, SC1.mdot_wf_bottom], name="Condenser")
    Tpinch_real = SC1.Condenser.Compute_Pinch()
    SC1.state_3_prime = SC1.Condenser.state_out_c
    res_cond = Tpinch_real - T_pinch

    # STEP 4 : Assemble the residuals

    residuals = np.array([res_evap, res_cond])
    return residuals


# Initial guesses
p1_guess = 5e5 ; p3_guess = 20e5
p_guess = np.array([p1_guess, p3_guess])

# Compute the solution
fsolve(iterative_process, p_guess)
SC1.COP = SC1.Condenser.Q / P_comp


############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
SC1.transforms = [Transform('comp', '1', '3', SC1.Compressor), 
                  Transform('hex', '10', '1',SC1.Evaporator, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                  Transform('adex', '9', '10', None), 
                  Transform('hex', '3', '9', SC1.Condenser, label_in_secondary='4_prime', label_out_secondary='3_prime')]

# Plot T-s diagram with saturation curve
SC1.Ts_diagram(n=100, plot=True)

if full_details :

    # Plot energy and exergy charts
    SC1.energy_chart(plot=True)
    SC1.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    SC1.Evaporator._plot(save=True, name_cycle=SC1.name, plot=True)
    SC1.Condenser._plot(save=True, name_cycle=SC1.name, plot=True)


############################################################
# Print the results
############################################################

print(SC1)

if full_details :
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
