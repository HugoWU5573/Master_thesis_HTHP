
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
from scipy.optimize import fsolve


############################################################
# Parameters
############################################################

# Technological parameters
T_pinch = 3                     # Minimum temperature difference in heat exchangers [K]
T_sup = 3                       # Superheating at the compressor inlet [K]
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
mdot_LT = 0.4                   # Mass flow rate of external fluid in the LT heat source [kg/s]
T1_prime = 10 + 273.15          # Inlet temperature of the external fluid in the LT heat source [K]
p1_prime = 1e5                  # Inlet pressure of the external fluid in LT the heat source [Pa]

    # 2. MT source
external_fluid_MT = 'Water'     # External fluid in the MT source
mdot_MT = 1.2                   # Mass flow rate of external fluid in the MT heat source [kg/s]
T3_prime = 40 + 273.15          # Inlet temperature of the external fluid in the MT heat source [K]
p3_prime = 1e5                  # Inlet pressure of the external fluid in MT the heat source [Pa]

# Heat sink parameters
external_fluid_HT = 'Water'     # External fluid in the heat sink
mdot_HT = 1                     # Mass flow rate of external fluid in the heat sink [kg/s]
T5_prime = 60 + 273.15          # Inlet temperature of the external fluid in the heat sink [K]
p5_prime = 2e5                  # Inlet pressure of the external fluid in the heat sink [Pa]


############################################################
# Instantiate objects
############################################################

# CoolProp low-level interface for all the fluids
HEOS_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
HEOS_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
HEOS_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
HEOS_working_fluid = CoolProp.AbstractState("HEOS", working_fluid)

# Cycle with its fixed states and mass flow rates
SC2R = Cycle("SC2R")
SC2R.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p1_prime)
SC2R.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p3_prime)
SC2R.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p5_prime)
SC2R.mdot_LT = mdot_LT
SC2R.mdot_MT = mdot_MT
SC2R.mdot_HT = mdot_HT

# Compressors
SC2R.P_comp_bottom = P_comp_1
SC2R.P_comp_top = P_comp_2
SC2R.Compressor_1 = Compressor_2_param(cycle=SC2R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)
SC2R.Compressor_2 = Compressor_2_param(cycle=SC2R, eta_v=eta_v, eta_is_max=eta_is_max, fluid=working_fluid, eta_elme=eta_elme)


############################################################
# Solve the cycle to determine the unknown states
############################################################

def iterative_process(p_gess) :
    p1_guess = p_gess[0] ; p3_guess = p_gess[1] ; p5_guess = p_gess[2]

    # STEP 1 : Compute the states based on the guesses values

        # Compute guessed state 1 (saturated vapor)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p1_guess, 0.0)
    Tsat_1 = HEOS_working_fluid.T()
    SC2R.state_1 = State(HEOS_working_fluid, T=Tsat_1, Q=1)

        # Compute guessed state 2 (superheated vapor at the first compressor inlet)
    SC2R.state_2 = State(HEOS_working_fluid, T=SC2R.state_1.T + T_sup, p=p1_guess)

        # Compute guessed state 3 (saturated vapor)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_3 = HEOS_working_fluid.T()
    SC2R.state_3 = State(HEOS_working_fluid, T=Tsat_3, Q=1)

        # Compute guessed state 4 (superheated vapor at the second compressor inlet)
    SC2R.state_4 = State(HEOS_working_fluid, T=SC2R.state_3.T + T_sup, p=p3_guess)

        # Compute guessed state 3_comp (exit of first compressor)
    SC2R.mdot_wf_bottom, T_3_comp = SC2R.Compressor_1.Solve(P_el=P_comp_1, p_ex=p3_guess, state_in=SC2R.state_2)
    SC2R.state_3_comp = State(HEOS_working_fluid, T=T_3_comp, p=p3_guess)

        # Compute guessed state 5 (exit of second compressor)
    SC2R.mdot_wf_top, T_5 = SC2R.Compressor_2.Solve(P_el=P_comp_2, p_ex=p5_guess, state_in=SC2R.state_4)
    SC2R.state_5 = State(HEOS_working_fluid, T=T_5, p=p5_guess)

        # Compute guessed state 6 (condenser outlet is saturated liquid)
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p5_guess, 0.0)
    Tsat_6 = HEOS_working_fluid.T()
    SC2R.state_6 = State(HEOS_working_fluid, T=Tsat_6, Q=0)

    # STEP 2 : Compute states 7 and 8 through the second recuperator model

    SC2R.Recuperator_2 = HEX_Design(states_in=[SC2R.state_3, SC2R.state_6], states_out=[SC2R.state_4, None], mdot=[SC2R.mdot_wf_top, SC2R.mdot_wf_top], name="Recuperator_2")
    SC2R.Recuperator_2.Compute_Pinch()
    SC2R.state_7 = SC2R.Recuperator_2.state_out_h

    h8 = SC2R.state_7.h
    SC2R.state_8 = State(HEOS_working_fluid, h=h8, p=p3_guess)

    # STEP 3 : Compute states 9 and 10 through the first recuperator model

    SC2R.Recuperator_1 = HEX_Design(states_in=[SC2R.state_1, SC2R.state_7], states_out=[SC2R.state_2, None], mdot=[SC2R.mdot_wf_bottom, SC2R.mdot_wf_bottom], name="Recuperator_1")
    SC2R.Recuperator_1.Compute_Pinch()
    SC2R.state_9 = SC2R.Recuperator_1.state_out_h

    h10 = SC2R.state_9.h
    SC2R.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)

    # STEP 4 : Compute the residual for the first evaporator

    SC2R.Evaporator_LT = HEX_Design(states_in=[SC2R.state_10, SC2R.state_1_prime], states_out=[SC2R.state_1, None], mdot=[SC2R.mdot_wf_bottom, SC2R.mdot_LT], name="Evaporator_LT")
    Tpinch_real = SC2R.Evaporator_LT.Compute_Pinch()
    SC2R.state_2_prime = SC2R.Evaporator_LT.state_out_h
    res_evap_LT = Tpinch_real - T_pinch

    # STEP 5 : Compute the residual for the second evaporator
    mdot_evap_MT = SC2R.mdot_wf_top - SC2R.mdot_wf_bottom  # Mass flow rate through the second evaporator

        # We first make a power balance to find state 3_evap (exit of the second evaporator) based on states 3 and 3_comp
    h_3_evap = (SC2R.mdot_wf_top * SC2R.state_3.h - SC2R.mdot_wf_bottom * SC2R.state_3_comp.h) / mdot_evap_MT
    SC2R.state_3_evap = State(HEOS_working_fluid, h=h_3_evap, p=p3_guess)

        # We can now compute the residual for the second evaporator
    SC2R.Evaporator_MT = HEX_Design(states_in=[SC2R.state_8, SC2R.state_3_prime], states_out=[SC2R.state_3_evap, None], mdot=[mdot_evap_MT, SC2R.mdot_MT], name="Evaporator_MT")
    Tpinch_real = SC2R.Evaporator_MT.Compute_Pinch()
    SC2R.state_4_prime = SC2R.Evaporator_MT.state_out_h
    res_evap_MT = Tpinch_real - T_pinch

    # STEP 6 : Compute the residual for the condenser

    SC2R.Condenser = HEX_Design(states_in=[SC2R.state_5_prime, SC2R.state_5], states_out=[None, SC2R.state_6], mdot=[SC2R.mdot_HT, SC2R.mdot_wf_top], name="Condenser")
    Tpinch_real = SC2R.Condenser.Compute_Pinch()
    SC2R.state_6_prime = SC2R.Condenser.state_out_c
    res_cond = Tpinch_real - T_pinch

    # STEP 7 : Assemble the residuals

    residuals = np.array([res_evap_LT, res_evap_MT, res_cond])
    return residuals


# Initial guesses
p1_guess = 5e5 ; p3_guess = 10e5 ; p5_guess = 30e5
p_guess = np.array([p1_guess, p3_guess, p5_guess])

# Compute the solution
fsolve(iterative_process, p_guess)
SC2R.COP = SC2R.Condenser.Q / (SC2R.P_comp_top + SC2R.P_comp_bottom)


############################################################
# Plot the results
############################################################

full_details = False

# Define the transforms 
SC2R.transforms = [Transform('isobaric_mixing', '3_comp', '3_evap', None),
                  Transform('comp', '2', '3_comp', SC2R.Compressor_1),
                  Transform('comp', '4', '5', SC2R.Compressor_2),
                  Transform('hex', '5', '6', SC2R.Condenser, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                  Transform('adex', '7', '8', None),
                  Transform('adex', '9', '10', None),
                  Transform('hex', '8', '3_evap',SC2R.Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                  Transform('hex', '10', '1',SC2R.Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'),
                  Transform('hex', '3', '4',SC2R.Recuperator_2, label_in_secondary='6', label_out_secondary='7'),
                  Transform('hex', '1', '2',SC2R.Recuperator_1, label_in_secondary='7', label_out_secondary='9')]

# Plot T-s diagram with saturation curve
SC2R.Ts_diagram(n=100, plot=True)

if full_details :

    # Plot energy and exergy charts
    SC2R.energy_chart(plot=True)
    SC2R.exergy_chart(T0=293.15, p0 = 1e5, plot=True)

    # Plot heat exchangers diagrams
    SC2R.Evaporator_LT._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Evaporator_MT._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Condenser._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Recuperator_1._plot(save=True, name_cycle=SC2R.name, plot=True)
    SC2R.Recuperator_2._plot(save=True, name_cycle=SC2R.name, plot=True)


############################################################
# Print the results
############################################################

print(SC2R)

if full_details :
    SC2R.Evaporator_LT.Compute_Area()
    SC2R.Evaporator_MT.Compute_Area()
    SC2R.Condenser.Compute_Area()
    SC2R.Recuperator_1.Compute_Area()
    SC2R.Recuperator_2.Compute_Area()
    print(SC2R.Evaporator_LT)
    print(SC2R.Evaporator_MT)
    print(SC2R.Condenser)
    print(SC2R.Recuperator_1)
    print(SC2R.Recuperator_2)

    # Save prints to a text file
    output_file = Path(__file__).parent.parent / "Figures" / SC2R.name / f"{SC2R.name}_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(SC2R) + '\n')
        f.write('\n' + str(SC2R.Evaporator_LT) + '\n')
        f.write('\n' + str(SC2R.Evaporator_MT) + '\n')
        f.write('\n' + str(SC2R.Condenser) + '\n')
        f.write('\n' + str(SC2R.Recuperator_1) + '\n')
        f.write('\n' + str(SC2R.Recuperator_2) + '\n')


""" WHAT REMAINS TO BE DONE :

    /!\ /!\ SAME AS TC1R CYCLE /!\ /!\
     
    - There is still a problem with the exergy plot (visual problem)
    
    - Verify the part of the Ts diagram where the recuperators are (with 'hex' transforms,
      only one side of the hex is represented, maybe add a new 'recup' transform)

"""