
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
from components.compressor import Compressor_LP, Compressor_HP
from components.valve import Valve, Valve_other
from components.HEX import HEX_Operational
from components.cycle import Cycle
import CoolProp
import numpy as np
from scipy.optimize import fsolve, root, minimize, least_squares
import time
import itertools
from multiprocessing import Pool, cpu_count, Lock


# PARAMETERS
    # 1. Component parameters
compressor_LP = Compressor_LP()
compressor_HP = Compressor_HP()
valve_LP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
valve_HP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
evaporator_LP = None
evaporator_HP = None
condenser = None

    # 2. Cycle parameters
fluid = 'R290'                                  # Working fluid
heos_working_fluid = CoolProp.AbstractState("HEOS", fluid)
cycle = Cycle("CAD")
    
############################################################
# Inputs 
############################################################
# Frequencies of the compressors 
N_LP = 50
N_HP = 50

# Power of the compressors
P_LP = 4e3 #np.arange(2.5, 3.6, 0.5) * 1e3
P_HP = 6e3 #np.arange(4.5, 7.5, 0.5) * 1e3


# Valve opening
z_LP = 25 #np.arange(35,46, 1)
z_HP = 20


# Manual Valve opening
HP_evaporator = False

############################################################
# Parameters
############################################################
# Heat sources parameters

    # 1. LT source
T1_prime = 15 + 273.75                          # Inlet temperature of the external fluid in the heat source [K] 
p1_prime = 3e5                                  # Inlet pressure of the external fluid in the heat source [Pa]
mdot_external_fluid_LT = 1.2                      # Mass flow rate of the external fluid in the heat source [kg/s]
external_fluid_LT = 'Water'
heos_external_fluid_LT = CoolProp.AbstractState("HEOS", external_fluid_LT)
cycle.state_1_prime = State(heos = heos_external_fluid_LT, T = T1_prime, p = p1_prime) 

    # 2. MT source
T3_prime = 70 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
p3_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
mdot_external_fluid_MT = 10                      # Mass flow rate of the external fluid in the heat sink [kg/s]
external_fluid_MT = 'Water'
heos_external_fluid_MT = CoolProp.AbstractState("HEOS", external_fluid_MT)
cycle.state_3_prime = State(heos = heos_external_fluid_MT, T = T3_prime, p = p3_prime)


# Heat sink parameters
T5_prime = 50 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
p5_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
mdot_external_fluid_HT = 0.5                    # Mass flow rate of the external fluid in the heat sink [kg/s]
external_fluid_HT = 'Water'
heos_external_fluid_HT = CoolProp.AbstractState("HEOS", external_fluid_HT)
cycle.state_5_prime = State(heos = heos_external_fluid_HT, T = T5_prime, p = p5_prime)



def objective(args) :
    p_1, p_3, h_1, dp_LP = args
    p_1 = np.exp(p_1)
    p_3 = np.exp(p_3)
    h_1 = np.exp(h_1)
    dp_LP = np.exp(dp_LP)
    
    #print(args)

    try : 
        cycle.state_1 = State(heos = heos_working_fluid, h = h_1, p = p_1)
        #print("State 1:", cycle.state_1)

        # LP Compressor
        h_3, P_LP_calc, cycle.mdot_wf_bottom = compressor_LP.Solve(cycle.state_1, p_3, N_LP)
        cycle.state_3 = State(heos = heos_working_fluid, h = h_3, p = p_3)
        #print("State 3 :", cycle.state_3)
        print("LP Compressor Power:", P_LP_calc)

        cycle.state_5 = State(heos = heos_working_fluid, h = cycle.state_3.h, p = cycle.state_3.p)
        cycle.mdot_wf_top = cycle.mdot_wf_bottom

        # Condenser
        condenser = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], mdot = [mdot_external_fluid_HT, cycle.mdot_wf_top], name = 'Condenser', N = 86, model = "ACP70X")
        cycle.state_6_prime, cycle.state_6 = condenser.Solve()[0]
        #print("State 6:", cycle.state_6)
        
        # LP Valve
        p_10 = p_1 + dp_LP
        h_10, z_LP_calc = valve_LP.Solve(cycle.state_6, p_10, cycle.mdot_wf_bottom)
        h_10 = cycle.state_6.h
        cycle.state_10 = State(heos = heos_working_fluid, h = h_10, p = p_10)
        #print("State 10:", cycle.state_10)

        # LP Evaporator
        evaporator_LP = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], mdot = [cycle.mdot_wf_bottom, mdot_external_fluid_LT], name = 'Evaporator_LP', N = 57, model = "ACP70X")
        state_1_calc, cycle.state_2_prime = evaporator_LP.Solve()[0]

        
        residual = [(state_1_calc.h - h_1) / h_1, (state_1_calc.p - p_1) / p_1, (z_LP_calc - z_LP) / z_LP, (P_LP_calc - P_LP) / P_LP]
        print(residual)
        if np.max(np.abs(residual)) < 5e-3:
            residual = [0, 0, 0, 0]
    
        return residual
        
    except Exception as e:
        print(f"Error in calculation: {e}")
        return [1e6, 1e6, 1e6, 1e6]  # Return large residuals to indicate failure


print("Starting simulation and optimization...")
start = time.time()
initial_guess = [np.log(5e5), np.log(20e5), np.log(630e3), np.log(0.5e5)]
initial_guess = np.log(np.array([663414.6753798822, 2346474.6741513005, 577373.2382013982, 4658.436906614825]))
for method in ['hybr'] :
    p_1, p_3, h_1, dp_LP = root(objective, initial_guess, options={'col_deriv': True}, method = method).x
    residual = objective([p_1, p_3, h_1, dp_LP])
    with open(f"code/Figures/CAD/results.txt", "a") as f:
        f.write(f"{P_LP} \t {P_HP} \t {N_LP} \t {N_HP} \t {z_LP} \t {z_HP} \t {np.exp(p_1)} \t {np.exp(p_3)} \t {np.exp(h_1)} \t {np.exp(dp_LP)} \t {np.sqrt(np.sum(np.array(residual)**2))} \n")
    break
end = time.time()
print(f"Simulation and optimization completed in {end - start:.2f} seconds.")


evaporator_LP = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], mdot = [cycle.mdot_wf_bottom, mdot_external_fluid_LT], name = 'Evaporator_LP', N = 57, model = "ACP70X")
Q_II = evaporator_LP.Solve()[1]
condenser = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], mdot = [mdot_external_fluid_HT, cycle.mdot_wf_top], name = 'Condenser', N = 86, model = "ACP70X")
Q_I = condenser.Solve()[1]
print("COP:", Q_I / (Q_I - Q_II))
cycle.transforms = [Transform('comp', '1', '3', compressor_LP), 
                    Transform('hex', '10', '1', evaporator_LP, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                    Transform('adex', '6', '10', valve_LP), 
                    Transform('hex', '5', '6', condenser)]
print(cycle.mdot_wf_bottom)
cycle.Ts_diagram(save = False)
cycle.ph_diagram(save = False)



    



