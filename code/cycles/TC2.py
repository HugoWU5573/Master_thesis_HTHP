
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
glide = 55                      # Temperature glide in the gas cooler [K]
eta_v = 0.8                     # Volumetric efficiency
eta_is_max = 0.7                # Maximum isentropic efficiency
eta_elme = 0.95                 # Electrical-mechanical efficiency

# Cycle parameters
working_fluid = 'R290'          # Working fluid
P_comp_1 = 2.5e3                # Electrical power of the first compressor [W]
P_comp_2 = 7.5e3                # Electrical power of the second compressor [W]

# Heat sources parameters

    # 1. LT source
external_fluid_LT = 'Water'     # External fluid in the LT source
""" MODIFIED START """
mdot_LT = 0.5                   # Mass flow rate of external fluid in the LT heat source [kg/s]
""" MODIFIED END """
T1_prime = 10 + 273.15          # Inlet temperature of the external fluid in the LT heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in LT the heat source [Pa]

    # 2. MT source
external_fluid_MT = 'Water'     # External fluid in the MT source
""" MODIFIED START """
mdot_MT = 0.4                   # Mass flow rate of external fluid in the MT heat source [kg/s]
""" MODIFIED END """
T3_prime = 40 + 273.15          # Inlet temperature of the external fluid in the MT heat source [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in MT the heat source [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
""" MODIFIED START """
mdot_HT = 0.13                  # Mass flow rate of external fluid in the heat sink [kg/s]
""" MODIFIED END """
T5_prime = 60 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]
T6_prime = T5_prime + glide     # Outlet temperature of the external fluid in the heat sink [K]
p6_prime = p5_prime             # Outlet pressure of the external fluid in the heat sink [Pa]


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
TC2.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
TC2.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
TC2.state_6_prime = State(HEOS_external_fluid_HT, T=T6_prime, p=p6_prime)
TC2.mdot_LT = mdot_LT
TC2.mdot_MT = mdot_MT
TC2.mdot_HT = mdot_HT

# Compressors
TC2.P_comp_bottom = P_comp_1
TC2.P_comp_top = P_comp_2
TC2.Compressor_1 = Compressor_2_param(cycle=TC2, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)
TC2.Compressor_2 = Compressor_2_param(cycle=TC2, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
############################################################

def iterative_process(p_gess) :
    p1_guess = p_gess[0] ; p3_guess = p_gess[1] ; p5_guess = p_gess[2]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 1 (first compressor inlet is superheated)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
    Tsat_1 = HEOS_working_fluid.T()
    TC2.state_1 = State(HEOS_working_fluid, T=Tsat_1 + T_sup, p=p1_guess)

        # Compute guessed state 3 (second compressor inlet is superheated)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    TC2.state_3 = State(HEOS_working_fluid, T=Tsat_3 + T_sup, p=p3_guess)

        # Compute guessed state 3_comp (exit of first compressor)
    TC2.mdot_wf_bottom, T_3_comp = TC2.Compressor_1.Solve(P_el=P_comp_1, p_ex=p3_guess, state_in=TC2.state_1)
    TC2.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

        # Compute guessed state 5 (exit of second compressor)
    TC2.mdot_wf_top, T_5 = TC2.Compressor_2.Solve(P_el=P_comp_2, p_ex=p5_guess, state_in=TC2.state_3)
    TC2.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

    # STEP 2 : Compute the residual for the gas cooler
    TC2.GasCooler = HEX_Design(states_in=[TC2.state_5_prime, TC2.state_5], states_out=[TC2.state_6_prime, None], mdot=[TC2.mdot_HT, TC2.mdot_wf_top], name="Gas Cooler")
    Tpinch_real = TC2.GasCooler.Compute_Pinch()
    TC2.state_7 = TC2.GasCooler.state_out_h
    res_gas_cooler = Tpinch_real - T_pinch

    # STEP 3 : Compute the residual for the first evaporator
        
        # Compute guessed state 8
    h8 = TC2.state_7.h
    TC2.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    """ MODIFIED START """
        # Compute guessed state 9 (same as state 7)
    TC2.state_9 = State(HEOS_working_fluid, h=TC2.state_7.h, p=TC2.state_7.p)
    """ MODIFIED END """

        # Compute guessed state 10
    h10 = TC2.state_9.h
    TC2.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

        # Compute the residual for the first evap
    TC2.Evaporator_LT = HEX_Design(states_in=[TC2.state_10, TC2.state_1_prime], states_out=[TC2.state_1, None], mdot=[TC2.mdot_wf_bottom, TC2.mdot_LT], name="Evaporator_LT")
    Tpinch_real = TC2.Evaporator_LT.Compute_Pinch()
    TC2.state_2_prime = TC2.Evaporator_LT.state_out_h
    res_evap_LT = Tpinch_real - T_pinch

    # STEP 4 : Compute the residual for the second evaporator
    mdot_evap_MT = TC2.mdot_wf_top - TC2.mdot_wf_bottom  # Mass flow rate through the second evaporator

        # We first make a power balance to find state 3_evap (exit of the second evaporator) based on states 3 and 3_comp
    h_3_evap = (TC2.mdot_wf_top * TC2.state_3.h - TC2.mdot_wf_bottom * TC2.state_3_comp.h) / mdot_evap_MT
    TC2.state_3_evap = State(HEOS_working_fluid, h=h_3_evap, p=p3_guess)

        # We can now compute the residual for the second evaporator
    TC2.Evaporator_MT = HEX_Design(states_in=[TC2.state_8, TC2.state_3_prime], states_out=[TC2.state_3_evap, None], mdot=[mdot_evap_MT, TC2.mdot_MT], name="Evaporator_MT")
    Tpinch_real = TC2.Evaporator_MT.Compute_Pinch()
    TC2.state_4_prime = TC2.Evaporator_MT.state_out_h
    res_evap_MT = Tpinch_real - T_pinch
  
    # STEP 5 : Assemble the residuals
    residuals = np.array([res_evap_LT, res_evap_MT, res_gas_cooler])
    return residuals


# Initial guesses
""" MODIFIED START """
p1_guess = 5e5 ; p3_guess = 10e5 ; p5_guess = 50e5
""" MODIFIED END """
p_guess = np.array([p1_guess, p3_guess, p5_guess])

# Compute the solution
fsolve(iterative_process, p_guess)
TC2.COP = TC2.GasCooler.Q / (TC2.P_comp_top + TC2.P_comp_bottom)


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

if full_details :

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

if full_details :
    """ MODIFIED START """
    TC2.Evaporator_LT.Compute_Area()
    TC2.Evaporator_MT.Compute_Area()
    TC2.GasCooler.Compute_Area()
    """ MODIFIED END """
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