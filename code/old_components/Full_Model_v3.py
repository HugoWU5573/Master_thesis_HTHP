
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

def run_Full_Model(name="", transcritical=False, recuperators=[False, False], save_results=False):

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

    # Valves
    Valve_LP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])
    Valve_HP = Valve_other([-2.29201900e-10,  6.60049743e-08,  2.27974980e-08])

    ############################################################
    # Iterative process to solve the cycle
    ############################################################

    def iterative_process(args, results=False, tol=5e-2):

        try :

            p1 = args[0]*50e5
            p3 = args[1]*50e5
            p5 = args[2]*50e5
            N_LP = args[3]*75
            N_HP = args[4]*75

            # Check that the pressures are in the correct order
            if (p3 <= p1) or (p5 <= p3) :
                raise ValueError(f"Invalid pressures : p1 = {p1/1e5:.2f} bar, p3 = {p3/1e5:.2f} bar, p5 = {p5/1e5:.2f} bar. They must satisfy p1 < p3 < p5.")

            # Initial guesses

                # State 1
            T1_first_guess = T1_prime
            HEOS_working_fluid.update(CoolProp.PT_INPUTS, p1, T1_first_guess)
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
            delta_p_evap_LT_guess = 2000

                # Pressure drop in the evaporator MT
            delta_p_evap_MT_guess = 2000

            h1_old = 0
            h1_new = h1_first_guess

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

            while ((np.abs((h1_old - h1_new)/h1_new) > tol) or 
                   (np.abs((mdot_LT_old - mdot_LT_new)/mdot_LT_new) > tol) or 
                   (np.abs((mdot_HT_old - mdot_HT_new)/mdot_HT_new) > tol) or 
                   (np.abs((mdot_MT_old - mdot_MT_new)/mdot_MT_new) > tol) or
                   (np.abs((delta_p_evap_LT_old - delta_p_evap_LT_new)/delta_p_evap_LT_new) > tol) or 
                   (np.abs((delta_p_evap_MT_old - delta_p_evap_MT_new)/delta_p_evap_MT_new) > tol)):
                
                if iteration_counter == 1 :
                    print(f"The input parameters are p1 = {p1/1e5:.2f} bar, p3 = {p3/1e5:.2f} bar, p5 = {p5/1e5:.2f} bar, N_LP = {N_LP:.2f} Hz, N_HP = {N_HP:.2f} Hz.")

                # Stop if there are too many iterations to avoid infinite loops
                if iteration_counter > 50 :
                    raise ValueError("Too many iterations. The solution may not be converging.")

                # STATE 1
                cycle.state_1 = State(HEOS_working_fluid, p=p1, h=h1_new)

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

                h1_old = h1_new
                mdot_LT_old = mdot_LT_new
                mdot_MT_old = mdot_MT_new
                mdot_HT_old = mdot_HT_new
                delta_p_evap_LT_old = delta_p_evap_LT_new
                delta_p_evap_MT_old = delta_p_evap_MT_new

                # Obtain the new value of h1
                h1_new = cycle.state_1.h

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
                residual_mdot_LT = (mdot_LT_old - mdot_LT_new)/mdot_LT_new
                residual_mdot_MT = (mdot_MT_old - mdot_MT_new)/mdot_MT_new
                residual_mdot_HT = (mdot_HT_old - mdot_HT_new)/mdot_HT_new
                residual_delta_p_evap_LT = (delta_p_evap_LT_old - delta_p_evap_LT_new)/delta_p_evap_LT_new
                residual_delta_p_evap_MT = (delta_p_evap_MT_old - delta_p_evap_MT_new)/delta_p_evap_MT_new

                # Residuals table
                if iteration_counter == 1:
                    print("-" * 100)
                    print(
                        f"{'It.':>4} | {'res_h1':>12} | {'res_mdot_LT':>12} | "
                        f"{'res_mdot_MT':>12} | {'res_mdot_HT':>12} | {'res_dp_LT':>12} | {'res_dp_MT':>12}"
                    )
                    print("-" * 100)

                print(
                    f"{iteration_counter:4d} | "
                    f"{residual_h1:12.2%} | {residual_mdot_LT:12.2%} | "
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

    T3_first_guess = T3_prime
    HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T3_first_guess-10)
    p3_first_guess = HEOS_working_fluid.p()

    if not transcritical :
        T5_first_guess = T5_prime + 10
        HEOS_working_fluid.update(CoolProp.QT_INPUTS, 1, T5_first_guess)
        p5_first_guess = HEOS_working_fluid.p()
        p5_max = 40e5
    else :
        p5_first_guess = HEOS_working_fluid.p_critical() + 5e5
        p5_max = 55e5

    initial_guess = np.zeros(5)

    p5_min = p3_first_guess
    p3_max = p5_first_guess
    p3_min = p1_first_guess
    p1_max = p3_first_guess
    p1_min = 1e5

    initial_guess[0] = p1_first_guess/50e5
    initial_guess[1] = p3_first_guess/50e5
    initial_guess[2] = p5_first_guess/50e5
    initial_guess[3] = 38/75     # Initial guess for the LP compressor frequency [Hz]
    initial_guess[4] = 59/75     # Initial guess for the HP compressor frequency [Hz]

    bounds = [(p1_min/50e5, p1_max/50e5),
              (p3_min/50e5, p3_max/50e5), 
              (p5_min/50e5, p5_max/50e5), 
              (25/75, 75/75), 
              (25/75, 75/75)]     # Bounds for p1, p3, p5 and compressor frequencies
    
    sol = minimize(iterative_process, initial_guess, bounds=bounds, options={'ftol': 1e-2, 'xtol': 1e-2})

    if sol.success:
        p1 = sol.x[0]*50e5
        p3 = sol.x[1]*50e5
        p5 = sol.x[2]*50e5
        N_LP = sol.x[3]*75
        N_HP = sol.x[4]*75
        print(f"Solution found : p1 = {p1/1e5:.2f} bar, p3 = {p3/1e5:.2f} bar, p5 = {p5/1e5:.2f} bar, N_LP = {N_LP:.2f} Hz, N_HP = {N_HP:.2f} Hz")
    else :
        print("No solution found")
    
    # Run the iterative process one last time with the solution to get the final results
    iterative_process(sol.x, results=True, tol=5e-4)
    


if __name__ == "__main__":

    run_SC2 = True
    run_SC2R = False

    save_results = False

    start = time()

    # SC2 cycle
    if run_SC2 :
        run_Full_Model(name="SC2", transcritical=False, recuperators=[False, False], save_results=save_results)

    # SC2R cycle
    if run_SC2R :
        run_Full_Model(name="SC2R", transcritical=False, recuperators=[True, False], save_results=save_results)


    end = time()

    min = (end - start)//60
    sec = np.round((end - start) % 60, 0)

    print(f"Execution time : {min:.0f} min {sec:.0f} sec")


    """
    A FAIRE :
        - Lorsque nous avons les deux compresseurs, le récupérateur MT ne fonctionne pas très bien (T3>T6)
    
    """