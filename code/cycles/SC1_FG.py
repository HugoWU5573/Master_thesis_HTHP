
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
from CoolProp.CoolProp import PropsSI
import CoolProp
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve


############################################################
# Parameters
############################################################

# Technological parameters
T_pinch = 3                     # Minimum temperature difference in heat exchangers [K]
T_sup = 3                       # Superheating at the compressor inlet [K]
T_sub = 3                       # Subcooling at the condenser outlet [K]
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
    SC1.mdot_wf, T_3 = SC1.Compressor.Solve(P_el=P_comp, p_ex=p3_guess, state_in=SC1.state_1)
    SC1.state_3 = State(HEOS_working_fluid, T=T_3, p=p3_guess)

        # Compute guessed state 9
    HEOS_working_fluid.update(CoolProp.PQ_INPUTS, p3_guess, 0.0)
    Tsat_9 = HEOS_working_fluid.T()
    SC1.state_9 = State(HEOS_working_fluid, T=Tsat_9 - T_sub, p=p3_guess)

        # Compute guessed state 10
    h10 = SC1.state_9.h
    SC1.state_10 = State(HEOS_working_fluid, h=h10, p=p1_guess)


    # STEP 2 : Compute the residual for the evaporator

    SC1.Evaporator = HEX_Design(states_in=[SC1.state_10, SC1.state_1_prime], states_out=[SC1.state_1, None], mdot=[SC1.mdot_wf, SC1.mdot_LT], name="Evaporator")
    Tpinch_real = SC1.Evaporator.Compute_Pinch()
    SC1.state_2_prime = SC1.Evaporator.state_out_h
    res_evap = Tpinch_real - T_pinch

    # STEP 3 : Compute the residual for the condenser

    SC1.Condenser = HEX_Design(states_in=[SC1.state_4_prime, SC1.state_3], states_out=[None, SC1.state_9], mdot=[SC1.mdot_MT, SC1.mdot_wf], name="Condenser")
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

'''
print(SC1)
print(SC1.Evaporator)
print(SC1.Condenser)
SC1.Evaporator._plot()
SC1.Condenser._plot()
'''
# Define the transforms 
SC1.transforms = [Transform('comp', '1', '3', SC1.Compressor), Transform('hex', '10', '1',SC1.Evaporator), 
                  Transform('adex', '9', '10', None), Transform('hex', '3', '9', SC1.Condenser)]


# Plot T-s diagram with saturation curve
'''
# Generate saturation curve for working fluid
T_min = PropsSI('Tmin', working_fluid) + 1
T_crit = PropsSI('Tcrit', working_fluid) - 1
T_sat = np.linspace(T_min, T_crit, 500)
s_liq = [PropsSI('S', 'T', T, 'Q', 0, working_fluid) for T in T_sat]
s_vap = [PropsSI('S', 'T', T, 'Q', 1, working_fluid) for T in T_sat]

plt.figure(figsize=(8,6))
plt.plot(s_liq, T_sat, 'b-', label='Saturated Liquid')
plt.plot(s_vap, T_sat, 'r-', label='Saturated Vapor')

# Extract states for plotting
states = [
    SC1.state_1,
    SC1.state_3,
    SC1.state_9,
    SC1.state_10,
]
state_labels = ['1', '3', '9', '10']

T_points = [s.T for s in states]
s_points = [s.s for s in states]

plt.plot(s_points, T_points, 'ko')

# Add labels to each state point
for i, (s_val, T_val, label) in enumerate(zip(s_points, T_points, state_labels)):
    plt.text(s_val, T_val, f' {label}', fontsize=10, verticalalignment='bottom', horizontalalignment='left')

plt.xlabel('Entropy [J/kg-K]')
plt.ylabel('Temperature [K]')
plt.title('Two-Stage Cycle in T-s Diagram')
plt.grid(True)
plt.tight_layout()
plt.show()
'''

SC1.Ts_diagram(n=100)