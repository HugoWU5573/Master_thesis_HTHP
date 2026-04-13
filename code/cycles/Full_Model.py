
############################################################
# Import libraries and modules
############################################################
import sys
from pathlib import Path

code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from components.transform import Transform
from components.state import State
from components.compressor import Compressor_other
from components.valve import Valve
from components.HEX import HEX_Operational
from components.cycle import Cycle
import CoolProp
import numpy as np
from scipy.optimize import root, fsolve, minimize, brentq
from time import time

def run_Full_Model(compressors_frequencies, valves_openings, recuperator=False):

    ############################################################
    # Model input parameters
    ############################################################

    ## Cycle parameters
    working_fluid = 'R290'      # Working fluid
    Q = 25e3                    # Heat load at the condenser/gas cooler [W]

    ## Heat sources parameters

        # 1. LT source
    external_fluid_LT = 'Water'     # External fluid in the LT heat source
    T1_prime = 308.15               # Inlet temperature of the external fluid in the LT heat source [K]
    glide_LT = 5                    # Temperature glide of the external fluid in the LT heat source [K]
    p_prime_LT = 3e5                # Inlet pressure of the external fluid in the LT heat source [Pa]

    # Heat sink parameters
    external_fluid_HT = 'Water'     # External fluid in the heat sink
    T5_prime = 333.15               # Inlet temperature of the external fluid in the heat sink [K]
    glide_HT = 5                    # Temperature glide of the external fluid in the heat sink [K]
    p_prime_HT = 5e5                # Inlet pressure of the external fluid in the heat sink [Pa]

    ## Components parameters

        # 1. Compressors
    N_LP = compressors_frequencies[0]     # Frequency of the low-pressure compressor [Hz]
    N_HP = compressors_frequencies[1]     # Frequency of the high-pressure compressor [Hz]

        # 2. EEVs
    z_LP = valves_openings[0]             # Opening of the low-pressure valve [%]
    z_HP = valves_openings[1]             # Opening of the high-pressure valve [%]

        # 3. BPHEs
    Nb_plates_Evap_LT = 57
    Nb_plates_Evap_MT = 30
    Nb_plates_Cond_Gas_Cooler = 85 
    Nb_plates_Recup_LT = 31
    Nb_plates_Recup_MT = 28


    ############################################################
    # Instantiate objects
    ############################################################

    # CoolProp low-level interface for all the fluids

    HEOS_type = "HEOS"  # Choose from "HEOS", "TTSE&HEOS"

    HEOS_external_fluid_LT = CoolProp.AbstractState(HEOS_type, external_fluid_LT)
    HEOS_external_fluid_HT = CoolProp.AbstractState(HEOS_type, external_fluid_HT)
    HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

    # Cycle with its fixed states
    cycle = Cycle("Full_Model")
    cycle.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p_prime_LT)
    cycle.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p_prime_HT)

    # Compressors
    Compressor_LP = Compressor_other()
    Compressor_HP = Compressor_other()

    # EEVs
    Valve_LP = Valve([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
    Valve_HP = Valve([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])

    ############################################################
    # Iterative process to solve the cycle
    ############################################################

    def iterative_process(args, results=False, tol=5e-3):

        try :
            
            p5 = np.exp(args)[0]

            # Initial guesses
            T1_first_guess = T1_prime
            HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T1_first_guess-10)
            p1_first_guess = HEOS_working_fluid.p()
            HEOS_working_fluid.update(CoolProp.PT_INPUTS, p1_first_guess, T1_first_guess)
            h1_first_guess = HEOS_working_fluid.hmass()

            # Mass flow rate in the condenser/gas cooler
            HEOS_external_fluid_HT.update(CoolProp.PT_INPUTS, p_prime_HT, T5_prime)
            cp = HEOS_external_fluid_HT.cpmass()
            m_dot_HT_first_guess = Q/(cp*glide_HT)

            # Mass flow rate in the Evaporator LT
            COP_guess = 4
            Q_evap_LT_guess = (COP_guess-1)/COP_guess*Q
            HEOS_external_fluid_LT.update(CoolProp.PT_INPUTS, p_prime_LT, T1_prime)
            cp = HEOS_external_fluid_LT.cpmass()
            m_dot_LT_first_guess = Q_evap_LT_guess/(cp*glide_LT)

            h1_old = 0 ; p1_old = 0
            h1_new = h1_first_guess ; p1_new = p1_first_guess

            mdot_LT_old = 0 ; mdot_HT_old = 0
            mdot_LT_new = m_dot_LT_first_guess ; mdot_HT_new = m_dot_HT_first_guess
            
            Q_calculated = 0
            iteration_counter = 1
            mdot_working_fluid = None

            while ((np.abs((h1_old - h1_new)/h1_new) > tol) or (np.abs((p1_old - p1_new)/p1_new) > tol) 
                    or (np.abs((mdot_LT_old - mdot_LT_new)/mdot_LT_new) > tol) or 
                    (np.abs((mdot_HT_old - mdot_HT_new)/mdot_HT_new) > tol)):

                print("Iteration ", iteration_counter, " with p5 = ", p5/1e5, " bar")

                # Create state 1
                cycle.state_1 = State(HEOS_working_fluid, p=p1_new, h=h1_new)

                if (not recuperator) :
                    # State 2 is the same as state 1
                    cycle.state_2 = State(HEOS_working_fluid, p=cycle.state_1.p, h=cycle.state_1.h)
                elif (iteration_counter == 1) :
                    # Initial guess for state 2 (superheated state at the same pressure as state 1)
                    cycle.state_2 = State(HEOS_working_fluid, p=cycle.state_1.p, T=cycle.state_1.T+10)
                else :
                    # Obtain states 2 and 9 with the recuperator
                    Reccuperator_LT = HEX_Operational(states_in=[cycle.state_1, cycle.state_6], 
                                                      mdot=[mdot_working_fluid, mdot_working_fluid],
                                                      name='Recuperator_LT', N=Nb_plates_Recup_LT, 
                                                      model="ACK18")
                    cycle.state_2, cycle.state_9 = Reccuperator_LT.Solve()[0]

                # Obtain state 5 and the mass flow rate for the working fluid with the Compressor
                T5, P_LP, mdot_working_fluid = Compressor_LP.Solve(cycle.state_2, p5, N_LP)
                cycle.state_5 = State(HEOS_working_fluid, T=T5, p=p5)

                # Obtain state 6 with the Condenser/Gas Cooler
                Condenser_Gas_Cooler = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], 
                                                    mdot = [mdot_HT_new, mdot_working_fluid], 
                                                    name = 'Condenser/GasCooler', N = Nb_plates_Cond_Gas_Cooler, 
                                                    model = "ACP70X")
                
                cycle.state_6_prime, cycle.state_6 = Condenser_Gas_Cooler.Solve()[0]
                Q_calculated = Condenser_Gas_Cooler.Q

                if (not recuperator) :
                    # State 9 is the same as state 6
                    cycle.state_9 = State(HEOS_working_fluid, p=cycle.state_6.p, h=cycle.state_6.h)
                else :
                    # Obtain states 2 and 9 with the recuperator
                    Reccuperator_LT = HEX_Operational(states_in=[cycle.state_1, cycle.state_6], 
                                                      mdot=[mdot_working_fluid, mdot_working_fluid],
                                                      name='Recuperator_LT', N=Nb_plates_Recup_LT, 
                                                      model="ACK18")
                    cycle.state_2, cycle.state_9 = Reccuperator_LT.Solve()[0]

                # Obtain state 10 with the EEV
                p_10, h_10 = Valve_LP.Solve(cycle.state_9, mdot_working_fluid, z_LP)
                cycle.state_10 = State(HEOS_working_fluid, p=p_10, h=h_10)

                # Obtain state 1 with the Evaporator
                Evaporator_LT = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], 
                                                mdot = [mdot_working_fluid, mdot_LT_new], 
                                                name = 'Evaporator_LT', N = Nb_plates_Evap_LT, 
                                                model = "ACP70X")
                cycle.state_1, cycle.state_2_prime = Evaporator_LT.Solve()[0]

                h1_old = h1_new ; p1_old = p1_new
                mdot_LT_old = mdot_LT_new ; mdot_HT_old = mdot_HT_new

                # Obtain the new mass flow rates in the LT and HT circuits

                T_mean_LT = (cycle.state_1_prime.T + cycle.state_2_prime.T)/2
                p_mean_LT = (cycle.state_1_prime.p + cycle.state_2_prime.p)/2
                HEOS_external_fluid_LT.update(CoolProp.PT_INPUTS, p_mean_LT, T_mean_LT)
                cp_LT = HEOS_external_fluid_LT.cpmass()
                mdot_LT_new = Evaporator_LT.Q/(cp_LT*glide_LT)

                T_mean_HT = (cycle.state_5_prime.T + cycle.state_6_prime.T)/2
                p_mean_HT = (cycle.state_5_prime.p + cycle.state_6_prime.p)/2
                HEOS_external_fluid_HT.update(CoolProp.PT_INPUTS, p_mean_HT, T_mean_HT)
                cp_HT = HEOS_external_fluid_HT.cpmass()
                mdot_HT_new = Condenser_Gas_Cooler.Q/(cp_HT*glide_HT)

                # Under-relaxation to ensure convergence
                if (not recuperator) :
                    under_relaxation_factor = 0.25
                else :
                    under_relaxation_factor = 0.1
                h1_new = under_relaxation_factor*cycle.state_1.h + (1-under_relaxation_factor)*h1_old
                p1_new = under_relaxation_factor*cycle.state_1.p + (1-under_relaxation_factor)*p1_old

                print(cycle.state_1)
                print(h1_new/1e3, p1_new/1e5)

                iteration_counter += 1

            # Final results after convergence

            if results:

                cycle.transforms = [Transform('comp', '2', '5', Compressor_LP), 
                                    Transform('hex', '5', '6', Condenser_Gas_Cooler, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                                    Transform('hex', '10', '1', Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                                    Transform('adex', '9', '10', Valve_LP)]
                
                if recuperator :
                    cycle.transforms.append(Transform('hex', '1', '2', Reccuperator_LT))

                cycle.Ts_diagram(save = False)
                cycle.ph_diagram(save = False)

                cycle.COP = Q_calculated/P_LP
                cycle.mdot_wf_bottom = mdot_working_fluid
                cycle.P_comp_bottom = P_LP

                print(cycle)
                
                print(Evaporator_LT)
                print(Condenser_Gas_Cooler)
                if recuperator :
                    print(Reccuperator_LT)
                Evaporator_LT._plot()
                Condenser_Gas_Cooler._plot()
                if recuperator :
                    Reccuperator_LT._plot()

            # Compute the resisudals

                # 1. Heat load at the condenser/gas cooler
            residual_1 = (Q_calculated - Q)/Q

            print(f"\t Iteration with p5 = {p5/1e5:.2f} bar : residual = {residual_1:.2f}, Q_calculated = {Q_calculated/1e3:.2f} kW")

            return residual_1
        
        except Exception as e:

            print(f"\t Iteration with p5 = {p5/1e5:.2f} bar : An error occurred during the iterative process : {e}")
        
            return -1   # Return large residuals to indicate failure in the iteration
    

    ############################################################
    # Solve the cycle
    ############################################################

    # Initial guesses for the iterative process

        # State 5
    T5_first_guess = T5_prime + 10
    HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T5_first_guess)
    p5_first_guess = HEOS_working_fluid.p()

    initial_guess = np.log(p5_first_guess)

    # Proceed with the iterative process with the root function

    sol = root(iterative_process,initial_guess, method = "lm", options={"ftol": 1e-4, "xtol": 1e-4, "gtol": 1e-4})
    #sol = minimize(iterative_process, initial_guess, bounds=[(np.log(20e5), np.log(30e5))])

    if sol.success:
        p5 = np.exp(sol.x[0])
        print(f"Solution found : p5 = {p5/1e5:.2f} bar")
    else :
        print("No solution found")

    # Run the iterative process one last time with the solution to get the final results
    iterative_process(sol.x, results=True, tol=5e-4)

    


if __name__ == "__main__":

    compressors_frequencies = [50, 50]     # Frequencies of the compressors [Hz]
    valves_openings = [31, 50]             # Openings of the valves [-]

    
    start = time()

    run_Full_Model(compressors_frequencies, valves_openings, recuperator=True)

    end = time()

    min = (end - start)//60
    sec = np.round((end - start) % 60, 0)

    print(f"Execution time : {min:.0f} min {sec:.0f} sec")

    
    
    
    """
    valves_openings_list = np.linspace(1, 100, 100)
    
    for i in range(len(valves_openings_list)):
        print(f"Iteration {i+1}/{len(valves_openings_list)} with valve opening = {valves_openings_list[i]:.2f} %")
        valves_openings[0] = valves_openings_list[i]
        run_Full_Model(compressors_frequencies, valves_openings, recuperator=True)

    """


    """

    Avec récupérateur, z = 31
    Sans récupérateur, z = 37

    
    """