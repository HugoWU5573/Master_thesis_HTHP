
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
from components.compressor import Compressor_LP, Compressor_HP
from components.valve import Valve_other
from components.HEX import HEX_Operational
from components.cycle import Cycle
import CoolProp
import numpy as np
from scipy.optimize import root, fsolve, minimize, brentq, differential_evolution
from time import time

def run_Full_Model(name="", dual_compressor=False, active_compressor="LP", transcritical=False, recuperators=[False, False], save_results=False):

    ############################################################
    # Model input parameters
    ############################################################

    ## Cycle parameters
    working_fluid = 'R290'      # Working fluid
    Q = 25e3                    # Heat load at the condenser/gas cooler [W]
    alpha = 0.5                 # Heat fraction coefficient (Q_MT / (Q_MT + Q_LT)) [-]

    ## Heat sources parameters

        # 1. LT source
    external_fluid_LT = 'Water'     # External fluid in the LT heat source
    T1_prime = 273.15 + 15          # Inlet temperature of the external fluid in the LT heat source [K]
    glide_LT = 5                    # Temperature glide of the external fluid in the LT heat source [K]
    p_prime_LT = 3e5                # Inlet pressure of the external fluid in the LT heat source [Pa]

        # 2. MT source
    external_fluid_MT = 'Water'     # External fluid in the MT heat source
    T3_prime = 40 + 273.15          # Inlet temperature of the external fluid in the MT heat source [K]
    glide_MT = 5                    # Temperature glide of the external fluid in the MT heat source [K]
    p_prime_MT = 3e5                # Inlet pressure of the external fluid in the MT heat source [Pa]

    # Heat sink parameters
    external_fluid_HT = 'Water'     # External fluid in the heat sink
    if not transcritical :
        T5_prime = 60 + 273.15      # Inlet temperature of the external fluid in the heat sink [K]
        glide_HT = 5                # Temperature glide of the external fluid in the heat sink [K]
    else :
        T5_prime = 363.15           # Inlet temperature of the external fluid in the heat sink [K]
        glide_HT = 25               # Temperature glide of the external fluid in the heat sink [K]
    p_prime_HT = 5e5                # Inlet pressure of the external fluid in the heat sink [Pa]

    ## Components parameters

        # BPHEs
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
    HEOS_external_fluid_MT = CoolProp.AbstractState(HEOS_type, external_fluid_MT)
    HEOS_external_fluid_HT = CoolProp.AbstractState(HEOS_type, external_fluid_HT)
    HEOS_working_fluid = CoolProp.AbstractState(HEOS_type, working_fluid)

    # Cycle with its fixed states
    cycle_name = "Full_Model_" + name
    cycle = Cycle(cycle_name)
    cycle.state_1_prime = State(HEOS_external_fluid_LT, T=T1_prime, p=p_prime_LT)
    cycle.state_3_prime = State(HEOS_external_fluid_MT, T=T3_prime, p=p_prime_MT)
    cycle.state_5_prime = State(HEOS_external_fluid_HT, T=T5_prime, p=p_prime_HT)

    # Compressors
    Comp_LP = Compressor_LP()
    Comp_HP = Compressor_HP()

    if not dual_compressor and active_compressor == "HP" :
        Comp = Comp_HP
    elif not dual_compressor and active_compressor == "LP" :
        Comp = Comp_LP

    # Valves
    Valve_LP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
    Valve_HP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])

    ############################################################
    # Iterative process to solve the cycle
    ############################################################

    def iterative_process_single_compressor(args, results=False, tol=5e-3):

        try :
            
            p5 = np.exp(args[0])
            N = args[1]

            # Initial guesses

                # State 1
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

                # Pressure drop in the evaporator LT
            delta_p_evap_LT_guess = 10

            h1_old = 0 ; p1_old = 0
            h1_new = h1_first_guess ; p1_new = p1_first_guess

            mdot_LT_old = 0 ; mdot_HT_old = 0
            mdot_LT_new = m_dot_LT_first_guess ; mdot_HT_new = m_dot_HT_first_guess

            delta_p_evap_LT_old = 0
            delta_p_evap_LT_new = delta_p_evap_LT_guess
            
            Q_calculated = 0
            iteration_counter = 1
            mdot_working_fluid = None

            while ((np.abs((h1_old - h1_new)/h1_new) > tol) or (np.abs((p1_old - p1_new)/p1_new) > tol) 
                    or (np.abs((mdot_LT_old - mdot_LT_new)/mdot_LT_new) > tol) or 
                    (np.abs((mdot_HT_old - mdot_HT_new)/mdot_HT_new) > tol) or 
                    (np.abs((delta_p_evap_LT_old - delta_p_evap_LT_new)/delta_p_evap_LT_new) > tol)):

                print(f"Iteration {iteration_counter} with p5 = {p5/1e5:.2f} bar, N = {N:.2f} Hz")

                # Create state 1
                cycle.state_1 = State(HEOS_working_fluid, p=p1_new, h=h1_new)

                if (not recuperators[0]) :
                    # State 2 is the same as state 1
                    cycle.state_2 = State(HEOS_working_fluid, p=cycle.state_1.p, h=cycle.state_1.h)
                elif (iteration_counter == 1) :
                    # Initial guess for state 2 (superheated state at the same pressure as state 1)
                    cycle.state_2 = State(HEOS_working_fluid, p=(cycle.state_1.p), T=cycle.state_1.T+10)
                else :
                    # Obtain states 2 and 9 with the recuperator
                    Recuperator_LT = HEX_Operational(states_in=[cycle.state_1, cycle.state_6], 
                                                      mdot=[mdot_working_fluid, mdot_working_fluid],
                                                      name='Recuperator_LT', N=Nb_plates_Recup_LT, 
                                                      model="ACK18")
                    cycle.state_2, cycle.state_9 = Recuperator_LT.Solve()[0]

                # Obtain state 5 and the mass flow rate for the working fluid with the Compressor
                h5, P, mdot_working_fluid = Comp.Solve(cycle.state_2, p5, N)
                cycle.state_5 = State(HEOS_working_fluid, h=h5, p=p5)

                # Obtain state 6 with the Condenser/Gas Cooler
                Condenser_Gas_Cooler = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], 
                                                    mdot = [mdot_HT_new, mdot_working_fluid], 
                                                    name = 'Condenser_GasCooler', N = Nb_plates_Cond_Gas_Cooler, 
                                                    model = "ACP70X")
                
                cycle.state_6_prime, cycle.state_6 = Condenser_Gas_Cooler.Solve()[0]
                Q_calculated = Condenser_Gas_Cooler.Q

                if (not recuperators[0]) :
                    # State 9 is the same as state 6
                    cycle.state_9 = State(HEOS_working_fluid, p=cycle.state_6.p, h=cycle.state_6.h)
                else :
                    # Obtain states 2 and 9 with the recuperator
                    Recuperator_LT = HEX_Operational(states_in=[cycle.state_1, cycle.state_6], 
                                                      mdot=[mdot_working_fluid, mdot_working_fluid],
                                                      name='Recuperator_LT', N=Nb_plates_Recup_LT, 
                                                      model="ACK18")
                    cycle.state_2, cycle.state_9 = Recuperator_LT.Solve()[0]

                # Obtain state 10 with the pressure drop guess in the evaporator LT
                p10 = cycle.state_1.p + delta_p_evap_LT_new
                h10, z = Valve_LP.Solve(cycle.state_9, p10, mdot_working_fluid)
                cycle.state_10 = State(HEOS_working_fluid, p=p10, h=h10)

                # Obtain state 1 with the Evaporator
                Evaporator_LT = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], 
                                                mdot = [mdot_working_fluid, mdot_LT_new], 
                                                name = 'Evaporator_LT', N = Nb_plates_Evap_LT, 
                                                model = "ACP70X")
                cycle.state_1, cycle.state_2_prime = Evaporator_LT.Solve()[0]

                h1_old = h1_new ; p1_old = p1_new
                mdot_LT_old = mdot_LT_new ; mdot_HT_old = mdot_HT_new
                delta_p_evap_LT_old = delta_p_evap_LT_new

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
                under_relaxation_factor = 0.5

                h1_new = under_relaxation_factor*cycle.state_1.h + (1-under_relaxation_factor)*h1_old
                p1_new = under_relaxation_factor*cycle.state_1.p + (1-under_relaxation_factor)*p1_old

                # Obtain the new pressure drop in the evaporator LT
                delta_p_evap_LT_new = cycle.state_10.p - cycle.state_1.p

                iteration_counter += 1

            # Final results after convergence
            cycle.transforms = [Transform('hex', '5', '6', Condenser_Gas_Cooler, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                                Transform('hex', '10', '1', Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                                Transform('adex', '9', '10', None), 
                                Transform('comp', '2', '5', Comp)]

            if recuperators[0] :
                cycle.transforms.append(Transform('hex', '1', '2', Recuperator_LT))
            if active_compressor == "HP" :
                cycle.mdot_wf_top = mdot_working_fluid
                cycle.mdot_wf_bottom = 0
                cycle.P_comp_top = P
                cycle.P_comp_bottom = 0
            else :
                cycle.mdot_wf_top = 0
                cycle.mdot_wf_bottom = mdot_working_fluid
                cycle.P_comp_top = 0
                cycle.P_comp_bottom = P

            cycle.COP = Q_calculated/(cycle.P_comp_top + cycle.P_comp_bottom)

            if results:

                cycle.Ts_diagram(save = save_results)
                cycle.ph_diagram(save = save_results)

                print(cycle)
                
                print(Evaporator_LT)
                print(Condenser_Gas_Cooler)
                if recuperators[0] :
                    print(Recuperator_LT)
                Evaporator_LT._plot(save = save_results, name_cycle=cycle.name)
                Condenser_Gas_Cooler._plot(save = save_results, name_cycle=cycle.name)
                if recuperators[0] :
                    Recuperator_LT._plot(save = save_results, name_cycle=cycle.name)

                # Save prints to a text file
                if save_results:
                    output_file = Path(__file__).parent.parent / "Figures" / cycle.name / f"{cycle.name}_results.txt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(str(cycle) + '\n')
                        f.write('\n' + str(Evaporator_LT) + '\n')
                        f.write('\n' + str(Condenser_Gas_Cooler) + '\n')
                        if recuperators[0] :
                            f.write('\n' + str(Recuperator_LT) + '\n')


            # We want to satisfy the heat load constraint
            residual_Q = (Q_calculated - Q)/Q
            COP = cycle.COP

            residual = -COP + 1e3*np.abs(residual_Q)

            print(f"\t End : COP = {cycle.COP:.2f}, z = {z:.2f} %, N = {N:.2f} Hz, Q_calculated = {Q_calculated/1e3:.2f} kW", f"-> Residual = {residual:.4f}")

            return residual
        
        except Exception as e:

            print(f"\t End : An error occurred during the iterative process : {e}")
        
            return 1e6   # Return a positive value to indicate that the solution is not valid for this p5
        

    def iterative_process_dual_compressor(args, results=False, tol=1e-2):

        try :

            p3 = np.exp(args[0])
            p5 = np.exp(args[1])
            N_LP = args[2]
            N_HP = args[3]

            # Check that p5 is higher than p3
            if p5 <= p3 :
                raise ValueError(f"p5 must be higher than p3. Got p3 = {p3/1e5:.2f} bar and p5 = {p5/1e5:.2f} bar.")

            # Initial guesses

                # State 1
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
            Q_evap_LT_guess = (COP_guess-1)/COP_guess*Q*(1-alpha)
            HEOS_external_fluid_LT.update(CoolProp.PT_INPUTS, p_prime_LT, T1_prime)
            cp = HEOS_external_fluid_LT.cpmass()
            m_dot_LT_first_guess = Q_evap_LT_guess/(cp*glide_LT)

                # Mass flow rate in the Evaporator MT
            Q_evap_MT_guess = (COP_guess-1)/COP_guess*Q*alpha
            HEOS_external_fluid_MT.update(CoolProp.PT_INPUTS, p_prime_MT, T3_prime)
            cp = HEOS_external_fluid_MT.cpmass()
            m_dot_MT_first_guess = Q_evap_MT_guess/(cp*glide_MT)

                # Pressure drop in the evaporator LT
            delta_p_evap_LT_guess = 10

                # Pressure drop in the evaporator MT
            delta_p_evap_MT_guess = 10

            h1_old = 0 ; p1_old = 0
            h1_new = h1_first_guess ; p1_new = p1_first_guess

            mdot_LT_old = 0 ; mdot_MT_old = 0 ; mdot_HT_old = 0
            mdot_LT_new = m_dot_LT_first_guess ; mdot_MT_new = m_dot_MT_first_guess ; mdot_HT_new = m_dot_HT_first_guess

            delta_p_evap_LT_old = 0
            delta_p_evap_LT_new = delta_p_evap_LT_guess

            delta_p_evap_MT_old = 0
            delta_p_evap_MT_new = delta_p_evap_MT_guess
            
            Q_calculated = 0
            iteration_counter = 1
            mdot_wf_bottom = 1
            mdot_wf_top = 1

            while ((np.abs((h1_old - h1_new)/h1_new) > tol) or (np.abs((p1_old - p1_new)/p1_new) > tol) 
                    or (np.abs((mdot_LT_old - mdot_LT_new)/mdot_LT_new) > tol) or 
                    (np.abs((mdot_HT_old - mdot_HT_new)/mdot_HT_new) > tol) or 
                    (np.abs((mdot_MT_old - mdot_MT_new)/mdot_MT_new) > tol) or
                    (np.abs((delta_p_evap_LT_old - delta_p_evap_LT_new)/delta_p_evap_LT_new) > tol) or 
                    (np.abs((delta_p_evap_MT_old - delta_p_evap_MT_new)/delta_p_evap_MT_new) > tol)):
                
                if iteration_counter == 1 :
                    print(f"The input parameters are p3 = {p3/1e5:.2f} bar, p5 = {p5/1e5:.2f} bar, N_LP = {N_LP:.2f} Hz, N_HP = {N_HP:.2f} Hz.")

                # Stop if there are too many iterations to avoid infinite loops
                if iteration_counter > 50 :
                    raise ValueError("Too many iterations. The solution may not be converging.")

                # STATE 1
                cycle.state_1 = State(HEOS_working_fluid, p=p1_new, h=h1_new)

                # STATE 2
                if (not recuperators[0]) :
                    # State 2 is the same as state 1
                    cycle.state_2 = State(HEOS_working_fluid, p=cycle.state_1.p, h=cycle.state_1.h)
                elif (iteration_counter == 1) :
                    # Initial guess for state 2 (superheated state at the same pressure as state 1)
                    cycle.state_2 = State(HEOS_working_fluid, p=(cycle.state_1.p), T=cycle.state_1.T+10)
                else :
                    # Obtain states 2 and 9 with the recuperator LT
                    Recuperator_LT = HEX_Operational(states_in=[cycle.state_1, cycle.state_7], 
                                                      mdot=[mdot_wf_bottom, mdot_wf_bottom],
                                                      name='Recuperator_LT', N=Nb_plates_Recup_LT, 
                                                      model="ACK18")
                    cycle.state_2, cycle.state_9 = Recuperator_LT.Solve()[0]

                # STATE 3 & Mass flow rate in the bottom part of the cycle
                if (iteration_counter == 1) :
                    h3, P_LP, mdot_wf_bottom = Comp_LP.Solve(cycle.state_2, p3, N_LP)
                    cycle.state_3 = State(HEOS_working_fluid, h=h3, p=p3)
                else :
                    h3_comp, P_LP, mdot_wf_bottom = Comp_LP.Solve(cycle.state_2, p3, N_LP)
                    mdot_wf_evap_MT = mdot_wf_top - mdot_wf_bottom
                    cycle.state_3_comp = State(HEOS_working_fluid, h=h3_comp, p=p3)
                    h3 = (h3_comp*mdot_wf_bottom + cycle.state_3_evap.h * mdot_wf_evap_MT)/mdot_wf_top
                    cycle.state_3 = State(HEOS_working_fluid, h=h3, p=p3)

                # STATE 4
                if (not recuperators[1]) :
                    # State 4 is the same as state 3
                    cycle.state_4 = State(HEOS_working_fluid, p=cycle.state_3.p, h=cycle.state_3.h)
                elif (iteration_counter == 1) :
                    # Initial guess for state 4 (superheated state at the same pressure as state 3)
                    cycle.state_4 = State(HEOS_working_fluid, p=(cycle.state_3.p), T=cycle.state_3.T+10)
                else :
                    # Obtain states 4 and 7 with the recuperator MT
                    Recuperator_MT = HEX_Operational(states_in=[cycle.state_3, cycle.state_6], 
                                                      mdot=[mdot_wf_top, mdot_wf_top],
                                                      name='Recuperator_MT', N=Nb_plates_Recup_MT, 
                                                      model="ACK18")
                    cycle.state_4, cycle.state_7 = Recuperator_MT.Solve()[0]
                
                # STATE 5 & Mass flow rate in the top part of the cycle
                h5, P_HP, mdot_wf_top = Comp_HP.Solve(cycle.state_4, p5, N_HP)
                cycle.state_5 = State(HEOS_working_fluid, h=h5, p=p5)

                # STATE 6
                Condenser_Gas_Cooler = HEX_Operational(states_in=[cycle.state_5_prime, cycle.state_5], 
                                                    mdot = [mdot_HT_new, mdot_wf_top], 
                                                    name = 'Condenser_GasCooler', N = Nb_plates_Cond_Gas_Cooler, 
                                                    model = "ACP70X")
                
                cycle.state_6_prime, cycle.state_6 = Condenser_Gas_Cooler.Solve()[0]
                Q_calculated = Condenser_Gas_Cooler.Q

                # STATE 7
                if (not recuperators[1]) :
                    # State 7 is the same as state 6
                    cycle.state_7 = State(HEOS_working_fluid, p=cycle.state_6.p, h=cycle.state_6.h)
                else :
                    # Obtain states 4 and 7 with the recuperator MT
                    Recuperator_MT = HEX_Operational(states_in=[cycle.state_3, cycle.state_6], 
                                                      mdot=[mdot_wf_top, mdot_wf_top],
                                                      name='Recuperator_MT', N=Nb_plates_Recup_MT, 
                                                      model="ACK18")
                    cycle.state_4, cycle.state_7 = Recuperator_MT.Solve()[0]

                # STATE 8
                p8 = cycle.state_3.p + delta_p_evap_MT_new
                h8, z_MT = Valve_HP.Solve(cycle.state_7, p8, (mdot_wf_top - mdot_wf_bottom))
                cycle.state_8 = State(HEOS_working_fluid, p=p8, h=h8)

                # STATE 3_evap
                Evaporator_MT = HEX_Operational(states_in=[cycle.state_8, cycle.state_3_prime],
                                                mdot = [mdot_wf_top - mdot_wf_bottom, mdot_MT_new], 
                                                name = 'Evaporator_MT', N = Nb_plates_Evap_MT, 
                                                model = "ACP70X")
                cycle.state_3_evap, cycle.state_4_prime = Evaporator_MT.Solve()[0]
                
                # STATE 9
                if (not recuperators[0]) :
                    # State 9 is the same as state 7
                    cycle.state_9 = State(HEOS_working_fluid, p=cycle.state_7.p, h=cycle.state_7.h)
                else :
                    # Obtain states 2 and 9 with the recuperator
                    Recuperator_LT = HEX_Operational(states_in=[cycle.state_1, cycle.state_7], 
                                                      mdot=[mdot_wf_bottom, mdot_wf_bottom],
                                                      name='Recuperator_LT', N=Nb_plates_Recup_LT, 
                                                      model="ACK18")
                    cycle.state_2, cycle.state_9 = Recuperator_LT.Solve()[0]

                # STATE 10
                p10 = cycle.state_1.p + delta_p_evap_LT_new
                h10, z_LP = Valve_LP.Solve(cycle.state_9, p10, mdot_wf_bottom)
                cycle.state_10 = State(HEOS_working_fluid, p=p10, h=h10)

                # STATE 1
                Evaporator_LT = HEX_Operational(states_in=[cycle.state_10, cycle.state_1_prime], 
                                                mdot = [mdot_wf_bottom, mdot_LT_new], 
                                                name = 'Evaporator_LT', N = Nb_plates_Evap_LT, 
                                                model = "ACP70X")
                cycle.state_1, cycle.state_2_prime = Evaporator_LT.Solve()[0]

                h1_old = h1_new ; p1_old = p1_new
                mdot_LT_old = mdot_LT_new
                mdot_MT_old = mdot_MT_new
                mdot_HT_old = mdot_HT_new
                delta_p_evap_LT_old = delta_p_evap_LT_new
                delta_p_evap_MT_old = delta_p_evap_MT_new

                # Under-relaxation to ensure convergence
                under_relaxation_factor = 0.5

                h1_new = under_relaxation_factor*cycle.state_1.h + (1-under_relaxation_factor)*h1_old
                p1_new = under_relaxation_factor*cycle.state_1.p + (1-under_relaxation_factor)*p1_old

                # Obtain the new mass flow rates in the LT, MT and HT circuits

                T_mean_LT = (cycle.state_1_prime.T + cycle.state_2_prime.T)/2
                p_mean_LT = (cycle.state_1_prime.p + cycle.state_2_prime.p)/2
                HEOS_external_fluid_LT.update(CoolProp.PT_INPUTS, p_mean_LT, T_mean_LT)
                cp_LT = HEOS_external_fluid_LT.cpmass()
                mdot_LT_new = Evaporator_LT.Q/(cp_LT*glide_LT)

                T_mean_MT = (cycle.state_3_prime.T + cycle.state_4_prime.T)/2
                p_mean_MT = (cycle.state_3_prime.p + cycle.state_4_prime.p)/2
                HEOS_external_fluid_MT.update(CoolProp.PT_INPUTS, p_mean_MT, T_mean_MT)
                cp_MT = HEOS_external_fluid_MT.cpmass()
                mdot_MT_new = Evaporator_MT.Q/(cp_MT*glide_MT)

                T_mean_HT = (cycle.state_5_prime.T + cycle.state_6_prime.T)/2
                p_mean_HT = (cycle.state_5_prime.p + cycle.state_6_prime.p)/2
                HEOS_external_fluid_HT.update(CoolProp.PT_INPUTS, p_mean_HT, T_mean_HT)
                cp_HT = HEOS_external_fluid_HT.cpmass()
                mdot_HT_new = Condenser_Gas_Cooler.Q/(cp_HT*glide_HT)

                # Obtain the new pressure drop in the evaporator LT
                delta_p_evap_LT_new = cycle.state_10.p - cycle.state_1.p
                delta_p_evap_MT_new = cycle.state_8.p - cycle.state_3_evap.p

                # Print the results of the current iteration
                residual_h1 = (h1_old - h1_new)/h1_new
                residual_p1 = (p1_old - p1_new)/p1_new
                residual_mdot_LT = (mdot_LT_old - mdot_LT_new)/mdot_LT_new
                residual_mdot_MT = (mdot_MT_old - mdot_MT_new)/mdot_MT_new
                residual_mdot_HT = (mdot_HT_old - mdot_HT_new)/mdot_HT_new
                residual_delta_p_evap_LT = (delta_p_evap_LT_old - delta_p_evap_LT_new)/delta_p_evap_LT_new
                residual_delta_p_evap_MT = (delta_p_evap_MT_old - delta_p_evap_MT_new)/delta_p_evap_MT_new

                # Residuals table
                if iteration_counter == 1:
                    print("-" * 118)
                    print(
                        f"{'It.':>4} | {'res_h1':>12} | {'res_p1':>12} | {'res_mdot_LT':>12} | "
                        f"{'res_mdot_MT':>12} | {'res_mdot_HT':>12} | {'res_dp_LT':>12} | {'res_dp_MT':>12}"
                    )
                    print("-" * 118)

                print(
                    f"{iteration_counter:4d} | "
                    f"{residual_h1:12.2%} | {residual_p1:12.2%} | {residual_mdot_LT:12.2%} | "
                    f"{residual_mdot_MT:12.2%} | {residual_mdot_HT:12.2%} | "
                    f"{residual_delta_p_evap_LT:12.2%} | {residual_delta_p_evap_MT:12.2%}"
                )

                iteration_counter += 1

            # Final results after convergence
            cycle.transforms = [Transform('hex', '5', '6', Condenser_Gas_Cooler, label_in_secondary='5_prime', label_out_secondary='6_prime'),
                                Transform('hex', '10', '1', Evaporator_LT, label_in_secondary='1_prime', label_out_secondary='2_prime'), 
                                Transform('hex', '8', '3_evap', Evaporator_MT, label_in_secondary='3_prime', label_out_secondary='4_prime'),
                                Transform('adex', '9', '10', None), 
                                Transform('adex', '7', '8', None),
                                Transform('isobaric_mixing', '3_comp', '3_evap', None),
                                Transform('comp', '2', '3_comp', Comp_LP),
                                Transform('comp', '4', '5', Comp_HP)]

            if recuperators[0] :
                cycle.transforms.append(Transform('hex', '1', '2', Recuperator_LT))
            if recuperators[1] :
                cycle.transforms.append(Transform('hex', '3', '4', Recuperator_MT))
            
            cycle.mdot_wf_top = mdot_wf_top
            cycle.mdot_wf_bottom = mdot_wf_bottom
            cycle.P_comp_top = P_HP
            cycle.P_comp_bottom = P_LP
            cycle.alpha = (Evaporator_MT.Q)/(Evaporator_LT.Q + Evaporator_MT.Q)

            cycle.COP = Q_calculated/(cycle.P_comp_top + cycle.P_comp_bottom)

            if results:

                cycle.Ts_diagram(save = save_results)
                cycle.ph_diagram(save = save_results)

                print(cycle)
                
                print(Evaporator_LT)
                print(Evaporator_MT)
                print(Condenser_Gas_Cooler)
                if recuperators[0] :
                    print(Recuperator_LT)
                if recuperators[1] :
                    print(Recuperator_MT)
                Evaporator_LT._plot(save = save_results, name_cycle=cycle.name)
                Evaporator_MT._plot(save = save_results, name_cycle=cycle.name)
                Condenser_Gas_Cooler._plot(save = save_results, name_cycle=cycle.name)
                if recuperators[0] :
                    Recuperator_LT._plot(save = save_results, name_cycle=cycle.name)
                if recuperators[1] :
                    Recuperator_MT._plot(save = save_results, name_cycle=cycle.name)

                # Save prints to a text file
                if save_results:
                    output_file = Path(__file__).parent.parent / "Figures" / cycle.name / f"{cycle.name}_results.txt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(str(cycle) + '\n')
                        f.write('\n' + str(Evaporator_LT) + '\n')
                        f.write('\n' + str(Evaporator_MT) + '\n')
                        f.write('\n' + str(Condenser_Gas_Cooler) + '\n')
                        if recuperators[0] :
                            f.write('\n' + str(Recuperator_LT) + '\n')
                        if recuperators[1] :
                            f.write('\n' + str(Recuperator_MT) + '\n')


            # We want to maximize the COP and to respect the following constraints : Q_calculated = Q and mdot_wf_top = mdot_wf_bottom
            residual_Q = (Q_calculated - Q)/Q
            alpha_calculated = Evaporator_MT.Q/(Evaporator_LT.Q + Evaporator_MT.Q)
            residual_alpha = alpha_calculated - alpha
            COP = cycle.COP

            residual = -COP + 1e3*abs(residual_Q) + 1e3*abs(residual_alpha)

            print(f"End : COP = {cycle.COP:.2f}, z_LP = {z_LP:.2f} %, z_HP = {z_MT:.2f} %, N_LP = {N_LP:.2f} Hz, N_HP = {N_HP:.2f} Hz, Q_calculated = {Q_calculated/1e3:.2f} kW, alpha_calculated = {alpha_calculated:.4f}", f"-> Residual = {residual:.4f}" + "\n")

            return residual
        
        except Exception as e:

            print(f"End : An error occurred during the iterative process : {e}" + "\n")
        
            return 1e6   # Return a positive value to indicate that the solution is not valid for this p5
    
    

    ############################################################
    # Solve the cycle
    ############################################################

    # Initial guesses for the iterative process

    T1_first_guess = T1_prime
    HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T1_first_guess-10)
    p1_first_guess = HEOS_working_fluid.p()

        # State 5
    if not transcritical :
        T5_first_guess = T5_prime + 10
        HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T5_first_guess)
        p5_first_guess = HEOS_working_fluid.p()
        p5_max = 40e5
        p5_min = p1_first_guess
    else :
        p5_first_guess = HEOS_working_fluid.p_critical() + 5e5
        p5_max = 55e5
        p5_min = p1_first_guess

    if not dual_compressor :

        initial_guess = np.zeros(2)

        initial_guess[0] = np.log(p5_first_guess)
        initial_guess[1] = 50     # Initial guess for the compressor frequency [Hz]

        bounds = [(np.log(p5_min), np.log(p5_max)), (25, 75)]     # Bounds for p5 and compressor frequency

        sol = minimize(iterative_process_single_compressor, initial_guess, bounds=bounds, options={'ftol': 1e-1})

        if sol.success:
            p5 = np.exp(sol.x[0])
            N = sol.x[1]
            print(f"Solution found : p5 = {p5/1e5:.2f} bar, N = {N:.2f} Hz")
        else :
            print("No solution found")

        # Run the iterative process one last time with the solution to get the final results
        iterative_process_single_compressor(sol.x, results=True, tol=5e-4)

    else :

        T3_first_guess = T3_prime
        HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T3_first_guess - 10)
        p3_first_guess = HEOS_working_fluid.p()

        # Pressure bounds
        p3_min = p1_first_guess
        p3_max = p5_max
        p5_min = p3_first_guess

        log_p3_min, log_p3_max = np.log(p3_min), np.log(p3_max)
        log_p5_min, log_p5_max = np.log(p5_min), np.log(p5_max)
        N_min, N_max = 25.0, 75.0

        bounds_4d = [
            (log_p3_min, log_p3_max),   # log(p3)
            (log_p5_min, log_p5_max),   # log(p5)
            (N_min, N_max),             # N_LP  [Hz]
            (N_min, N_max),             # N_HP  [Hz]
        ]

        # PHASE 1 : Global search

        print("\n" + "=" * 60)
        print("PHASE 1 - Global search")
        print("=" * 60)

        sol_global = differential_evolution(
            iterative_process_dual_compressor,
            bounds=bounds_4d,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.9,
            tol=1e-2,
            atol=0,
            seed=42,
            polish=False,
            workers=1,
            init='sobol',
            disp=True,
        )

        print(f"\nGlobal search finished.  Best residual = {sol_global.fun:.4f}")
        x_global = sol_global.x
        p3_g, p5_g = np.exp(x_global[0]), np.exp(x_global[1])
        print(f"  p3 = {p3_g/1e5:.2f} bar,  p5 = {p5_g/1e5:.2f} bar,"
              f"  N_LP = {x_global[2]:.1f} Hz,  N_HP = {x_global[3]:.1f} Hz")

        # PHASE 2 : Local refinement

        print("\n" + "=" * 60)
        print("PHASE 2 - Local refinement")
        print("=" * 60)

        sol_local = minimize(
            iterative_process_dual_compressor,
            x0=x_global,
            method='L-BFGS-B',
            bounds=bounds_4d,
            options={
                'ftol': 1e-4,
                'gtol': 1e-5,
                'maxiter': 200,
                'disp': True,
            },
        )

        if sol_local.fun <= sol_global.fun:
            sol_best = sol_local
        else:
            print("Local step did not improve; keeping global result.")
            sol_best = sol_global

        x_best = sol_best.x
        p3_best = np.exp(x_best[0])
        p5_best = np.exp(x_best[1])
        N_LP_best, N_HP_best = x_best[2], x_best[3]

        print("\n" + "=" * 60)
        print("OPTIMISATION COMPLETE")
        print("=" * 60)
        if sol_best.success:
            print(f"Solution found:")
        else:
            print(f"Best found (may not be fully converged):")
        print(f"  p3    = {p3_best/1e5:.3f} bar")
        print(f"  p5    = {p5_best/1e5:.3f} bar")
        print(f"  N_LP  = {N_LP_best:.2f} Hz")
        print(f"  N_HP  = {N_HP_best:.2f} Hz")
        print(f"  Residual = {sol_best.fun:.4f}")

        # PHASE 3 : Final evaluation with best parameters

        iterative_process_dual_compressor(x_best, results=True, tol=5e-4)
        


        """ V1 
        initial_guess = np.zeros(4)

        T3_first_guess = T3_prime
        HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T3_first_guess-10)
        p3_first_guess = HEOS_working_fluid.p()

        initial_guess[0] = np.log(p3_first_guess)
        initial_guess[1] = np.log(p5_first_guess)
        initial_guess[2] = 40     # Initial guess for the LP compressor frequency [Hz]
        initial_guess[3] = 60     # Initial guess for the HP compressor frequency [Hz]

        p5_min = p3_first_guess
        p3_max = p5_max
        p3_min = p1_first_guess

        bounds = [(np.log(p3_min), np.log(p3_max)), 
                  (np.log(p5_min), np.log(p5_max)), (25, 75), (25, 75)]     # Bounds for p3, p5 and compressor frequencies
        
        sol = minimize(iterative_process_dual_compressor, initial_guess, bounds=bounds, options={'ftol': 1e-2, 'xtol': 1e-2})

        if sol.success:
            p3 = np.exp(sol.x[0])
            p5 = np.exp(sol.x[1])
            N_LP = sol.x[2]
            N_HP = sol.x[3]
            print(f"Solution found : p3 = {p3/1e5:.2f} bar, p5 = {p5/1e5:.2f} bar, N_LP = {N_LP:.2f} Hz, N_HP = {N_HP:.2f} Hz")
        else :
            print("No solution found")
        
        # Run the iterative process one last time with the solution to get the final results
        iterative_process_dual_compressor(sol.x, results=True, tol=5e-4)
        """
    


if __name__ == "__main__":

    run_SC1 = False
    run_SC1R = False
    run_TC1 = False
    run_TC1R = False
    run_SC2 = True
    run_SC2R = False

    save_results = False

    start = time()

    # SC1 cycle

    if run_SC1 :
        run_Full_Model(name="SC1", dual_compressor=False, active_compressor="LP", transcritical=False, recuperators=[False, None], save_results=save_results)

    # SC1R cycle

    if run_SC1R :
        run_Full_Model(name="SC1R", dual_compressor=False, active_compressor="LP", transcritical=False, recuperators=[True, None], save_results=save_results)

    # TC1 cycle

    if run_TC1 :
        run_Full_Model(name="TC1", dual_compressor=False, active_compressor="HP", transcritical=True, recuperators=[False, None], save_results=save_results)

    # TC1R cycle

    if run_TC1R :
        run_Full_Model(name="TC1R", dual_compressor=False, active_compressor="HP", transcritical=True, recuperators=[True, None], save_results=save_results)

    # SC2 cycle

    if run_SC2 :
        run_Full_Model(name="SC2", dual_compressor=True, transcritical=False, recuperators=[False, False], save_results=save_results)

    # SC2R cycle

    if run_SC2R :
        run_Full_Model(name="SC2R", dual_compressor=True, transcritical=False, recuperators=[True, False], save_results=save_results)


    end = time()

    min = (end - start)//60
    sec = np.round((end - start) % 60, 0)

    print(f"Execution time : {min:.0f} min {sec:.0f} sec")


    """
    A FAIRE :

        - Il y a un problème avec le p1 (il est imposé en pratique dans mon code actuel)
        - Lorsque nous avons les deux compresseurs, le récupérateur MT ne fonctionne pas très bien (T3>T6)
    
    """