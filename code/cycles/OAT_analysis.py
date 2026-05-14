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
from matplotlib import pyplot as plt
from cycles.CAD_8 import CAD

plot_figures = False
sensitivity_analysis = True
file_sensitivity_analysis = "code/Figures/CAD/sensitivity_analysis.txt"

############################################################
# Functions
############################################################

def sensitivity_index(x_plus, x_minus, y_plus, y_minus) :
    y_mean = 0.5 * (y_plus + y_minus)
    x_mean = 0.5 * (x_plus + x_minus)
    dy = y_plus - y_minus
    dx = x_plus - x_minus
    SI = (dy / y_mean) / (dx / x_mean)
    return SI


############################################################
# Parameters
############################################################
# Heat sources parameters

    # 1. LT source
T1_prime = 15 + 273.75                          # Inlet temperature of the external fluid in the heat source [K] 
p1_prime = 3e5                                  # Inlet pressure of the external fluid in the heat source [Pa]
glide_external_fluid_LT = 5                      # Mass flow rate of the external fluid in the heat source [kg/s] 
external_fluid_LT_param = {'T1_prime': T1_prime, 'p1_prime': p1_prime, 'glide_external_fluid_LT': glide_external_fluid_LT}


    # 2. MT source
T3_prime = 40 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
p3_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
glide_external_fluid_MT = 5                      # Mass flow rate of the external fluid in the heat sink [kg/s]
external_fluid_MT_param = {'T3_prime': T3_prime, 'p3_prime': p3_prime, 'glide_external_fluid_MT': glide_external_fluid_MT}


# Heat sink parameters
T5_prime = 60 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
p5_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
glide_external_fluid_HT = 5                    # Mass flow rate of the external fluid in the heat sink [kg/s]
external_fluid_HT_param = {'T5_prime': T5_prime, 'p5_prime': p5_prime, 'glide_external_fluid_HT': glide_external_fluid_HT}

recuperator_LP = False


file_name = "code/Figures/CAD/results.txt"
data = np.loadtxt(file_name, skiprows=1)  # Skip the header row
reference_cycle = data[0]  # Assuming the first row corresponds to the reference cycle
Q_cond_cycles = data[1:3]
N_LP_cycles = data[3:5]
N_HP_cycles = data[5:7]
z_LP_cycles = data[7:9]
z_HP_cycles = data[9:11]

def OAT_analysis(cycles, variable_name, save_analysis = False, plot_figure = False, save_figure = False) : 
    # Reference cycle analysis
    compressor_inputs_ref = {'N_LP': reference_cycle[1], 'N_HP': reference_cycle[2]}
    valve_inputs_ref = {'z_LP': reference_cycle[3], 'z_HP': reference_cycle[4]}
    initial_guess_ref = np.log(reference_cycle[5:15])
    cycle_ref = CAD(reference_cycle[0], compressor_inputs_ref, valve_inputs_ref, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess_ref, save_results = False, plot_figures = False, solve = True)
    T_points_ref, s_points_ref, states_ref = cycle_ref.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram

    # Variable cycle analysis
    T_points = []
    s_points = []
    states = []
    h = []
    p = []
    mass_flow_rates = []
    Q = []
    P = []
    COP = []

    for i in range(2): 
        compressor_inputs = {'N_LP': cycles[i][1], 'N_HP': cycles[i][2]}
        valve_inputs = {'z_LP': cycles[i][3], 'z_HP': cycles[i][4]}
        initial_guess = np.log(cycles[i][5:15])
        cycle = CAD(cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                    initial_guess, save_results = False, plot_figures = False, solve = True)
        
        T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
        T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

        h.append([cycle.state_1.h, cycle.state_3.h, cycle.state_5.h, cycle.state_7.h, cycle.state_8.h, cycle.state_10.h, cycle.state_3_evap.h,\
                cycle.state_3_comp.h, cycle.state_2_prime.h, cycle.state_4_prime.h, cycle.state_6_prime.h])
        p.append([cycle.state_1.p, cycle.state_3.p, cycle.state_5.p, cycle.state_7.p, cycle.state_8.p, cycle.state_10.p, cycle.state_3_evap.p,\
                cycle.state_3_comp.p, cycle.state_2_prime.p, cycle.state_4_prime.p, cycle.state_6_prime.p])
        mass_flow_rates.append([cycle.mdot_wf_top, cycle.mdot_wf_bottom, cycle.mdot_LT, cycle.mdot_MT, cycle.mdot_HT])
        Q.append([cycle.Q_LT, cycle.Q_MT, cycle.Q_HT])
        P.append([cycle.P_comp_top, cycle.P_comp_bottom])
        COP.append(cycle.COP)

    if save_analysis :
        h = np.array(h) ; p = np.array(p) ; mass_flow_rates = np.array(mass_flow_rates) ; Q = np.array(Q) ; P = np.array(P) ; COP = np.array(COP)
        SI_h = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], h[1], h[0])
        SI_p = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], p[1], p[0])
        SI_mass_flow_rates = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], mass_flow_rates[1], mass_flow_rates[0])
        SI_Q = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], Q[1], Q[0])
        SI_P = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], P[1], P[0])
        SI_COP = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], COP[1], COP[0])

        with open(file_sensitivity_analysis, "a") as f:
            '''
            f.write("Variable \t h_1 \t h_3 \t h_5 \t h_7 \t h_8 \t h_10 \t h_3_evap \t h_3_comp \t h_2_prime \t h_4_prime \t h_6_prime" \
                    "\t p_1 \t p_3 \t p_5 \t p_7 \t p_8 \t p_10 \t p_3_evap \t p_3_comp \t p_2_prime \t p_4_prime \t p_6_prime \t mdot_wf_top"\
                    "\t mdot_wf_bottom \t mdot_LT \t mdot_MT \t mdot_HT \t Q_LT \t Q_MT \t Q_HT \t P_comp_top \t P_comp_bottom \t COP\n")
            '''
            str_h = "\t".join([f"{SI:.2f}" for SI in SI_h])
            str_p = "\t".join([f"{SI:.2f}" for SI in SI_p])
            str_mass_flow_rates = "\t".join([f"{SI:.2f}" for SI in SI_mass_flow_rates])
            str_Q = "\t".join([f"{SI:.2f}" for SI in SI_Q])
            str_P = "\t".join([f"{SI:.2f}" for SI in SI_P])
            str_COP = "\t".join([f"{SI_COP:.2f}"])
            f.write(variable_name + f" \t {str_h} \t {str_p} \t {str_mass_flow_rates} \t {str_Q} \t {str_P} \t {str_COP}\n")
    
    if plot_figure : 
        color = ['blue', 'red', 'green']

        heos = CoolProp.AbstractState("HEOS", list(states[0].values())[0].fluid)
        T_crit = heos.T_critical()
        p_crit = heos.p_critical()

        label_low = "" ; label_up = ""; label_ref = ""
        if variable_name == "Q_cond" :
            label_low = fr'$Q_{{cond}} = {cycles[0][0]/1e3:.2f}$ kW'
            label_up  = fr'$Q_{{cond}} = {cycles[1][0]/1e3:.2f}$ kW'
            label_ref = fr'$Q_{{cond}} = {reference_cycle[0]/1e3:.2f}$ kW (Reference)'
        elif variable_name == "N_LP" :
            label_low = fr'$N_{{LP}} = {cycles[0][1]}$ Hz'
            label_up = fr'$N_{{LP}} = {cycles[1][1]}$ Hz'
            label_ref = fr'$N_{{LP}} = {reference_cycle[1]}$ Hz (Reference)'
        elif variable_name == "N_HP" :
            label_low = fr'$N_{{HP}} = {cycles[0][2]}$ Hz'
            label_up = fr'$N_{{HP}} = {cycles[1][2]}$ Hz'
            label_ref = fr'$N_{{HP}} = {reference_cycle[2]}$ Hz (Reference)'
        elif variable_name == "z_LP" :
            label_low = fr'$z_{{LP}} = {cycles[0][3]}$ %'
            label_up = fr'$z_{{LP}} = {cycles[1][3]}$ %'
            label_ref = fr'$z_{{LP}} = {reference_cycle[3]}$ % (Reference)'
        elif variable_name == "z_HP" :
            label_low = fr'$z_{{HP}} = {cycles[0][4]}$ %'
            label_up = fr'$z_{{HP}} = {cycles[1][4]}$ %'
            label_ref = fr'$z_{{HP}} = {reference_cycle[4]}$ % (Reference)'

        plt.figure(figsize=(8, 6))
        for i in range(2):
            if i == 0 : 
                label_legend = label_low
            else :
                label_legend = label_up

            for k, (label, state) in enumerate(states[i].items()):
                plt.scatter(state.s/1e3, state.T - 273.15, color = color[i], clip_on = False)

            for j in range(len(T_points[i])):
                if j == 0 :
                    plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', label= label_legend, color = color[i], clip_on = False)
                else :
                    plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', color = color[i], clip_on = False)
        
        for j in range(len(T_points_ref)):
            if j == 0 :
                plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', label= label_ref, color = color[2], clip_on = False)
            else :
                plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', color = color[2], clip_on = False)

        for k, (label, state) in enumerate(states_ref.items()):
            plt.scatter(state.s/1e3, state.T - 273.15, color = color[2], clip_on = False)

        s_max_state = np.max(s_points)
        s_min_state = np.min(s_points)
        T_max_state = np.max(T_points)
        T_min_state = np.min(T_points)

        heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
        s_min_liq = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
        #s_max_vap = heos.smass()

        plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
        plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

        T_sat = np.linspace(T_min_state, T_crit - 1, 100)
        s_liq = np.zeros(100)
        s_vap = np.zeros(100)
        for i, T in enumerate(T_sat):
            heos.update(CoolProp.QT_INPUTS, 0, T)
            s_liq[i] = heos.smass()
            heos.update(CoolProp.QT_INPUTS, 1,T)
            s_vap[i] = heos.smass()

        plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = False)
        plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = False)
        heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
        s_crit = heos.smass()
        plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point


        plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.set_title('Temperature [°C]', loc='left', fontsize=12)

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move bottom and left spines away
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))

        # Improve readability
        plt.tick_params(axis='x', rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
        if save_figure :
            plt.savefig("code/Figures/CAD/Ts_diagram_" + variable_name + "_variation.png", dpi = 300, bbox_inches = 'tight')
        plt.show()
    
    return 


OAT_analysis(N_LP_cycles, "N_LP", plot_figure = True, save_figure = True)
OAT_analysis(N_HP_cycles, "N_HP", plot_figure = True, save_figure = True)
OAT_analysis(z_LP_cycles, "z_LP", plot_figure = True, save_figure = True)
OAT_analysis(z_HP_cycles, "z_HP", plot_figure = True, save_figure = True)



'''

# Reference cycle analysis
T_points_ref = []
s_points_ref = []
states_ref = []
compressor_inputs_ref = {'N_LP': reference_cycle[1], 'N_HP': reference_cycle[2]}
valve_inputs_ref = {'z_LP': reference_cycle[3], 'z_HP': reference_cycle[4]}
initial_guess_ref = np.log(reference_cycle[5:15])
cycle_ref = CAD(reference_cycle[0], compressor_inputs_ref, valve_inputs_ref, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess_ref, save_results = False, plot_figures = False, solve = True)
T_points_ref, s_points_ref, states_ref = cycle_ref.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram

# Q_cond analysis
T_points = []
s_points = []
states = []
h = []
p = []
mass_flow_rates = []
Q = []
P = []
COP = []

for i in range(2): 
    compressor_inputs = {'N_LP': Q_cond_cycles[i][1], 'N_HP': Q_cond_cycles[i][2]}
    valve_inputs = {'z_LP': Q_cond_cycles[i][3], 'z_HP': Q_cond_cycles[i][4]}
    initial_guess = np.log(Q_cond_cycles[i][5:15])
    cycle = CAD(Q_cond_cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                initial_guess, save_results = False, plot_figures = False, solve = True)
    
    T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
    T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

    h.append([cycle.state_1.h, cycle.state_3.h, cycle.state_5.h, cycle.state_7.h, cycle.state_8.h, cycle.state_10.h, cycle.state_3_evap.h,\
              cycle.state_3_comp.h, cycle.state_2_prime.h, cycle.state_4_prime.h, cycle.state_6_prime.h])
    p.append([cycle.state_1.p, cycle.state_3.p, cycle.state_5.p, cycle.state_7.p, cycle.state_8.p, cycle.state_10.p, cycle.state_3_evap.p,\
              cycle.state_3_comp.p, cycle.state_2_prime.p, cycle.state_4_prime.p, cycle.state_6_prime.p])
    mass_flow_rates.append([cycle.mdot_wf_top, cycle.mdot_wf_bottom, cycle.mdot_LT, cycle.mdot_MT, cycle.mdot_HT])
    Q.append([cycle.Q_LT, cycle.Q_MT, cycle.Q_HT])
    P.append([cycle.P_comp_top, cycle.P_comp_bottom])
    COP.append(cycle.COP)


if sensitivity_analysis :
    h = np.array(h) ; p = np.array(p) ; mass_flow_rates = np.array(mass_flow_rates) ; Q = np.array(Q) ; P = np.array(P) ; COP = np.array(COP)
    SI_h = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], h[1], h[0])
    SI_p = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], p[1], p[0])
    SI_mass_flow_rates = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], mass_flow_rates[1], mass_flow_rates[0])
    SI_Q = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], Q[1], Q[0])
    SI_P = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], P[1], P[0])
    SI_COP = sensitivity_index(Q_cond_cycles[1][0], Q_cond_cycles[0][0], COP[1], COP[0])
    with open(file_sensitivity_analysis, "a") as f:
        f.write("Variable \t h_1 \t h_3 \t h_5 \t h_7 \t h_8 \t h_10 \t h_3_evap \t h_3_comp \t h_2_prime \t h_4_prime \t h_6_prime" \
                "\t p_1 \t p_3 \t p_5 \t p_7 \t p_8 \t p_10 \t p_3_evap \t p_3_comp \t p_2_prime \t p_4_prime \t p_6_prime \t mdot_wf_top"\
                "\t mdot_wf_bottom \t mdot_LT \t mdot_MT \t mdot_HT \t Q_LT \t Q_MT \t Q_HT \t P_comp_top \t P_comp_bottom \t COP\n")
        str_h = "\t".join([f"{SI:.2f}" for SI in SI_h])
        str_p = "\t".join([f"{SI:.2f}" for SI in SI_p])
        str_mass_flow_rates = "\t".join([f"{SI:.2f}" for SI in SI_mass_flow_rates])
        str_Q = "\t".join([f"{SI:.2f}" for SI in SI_Q])
        str_P = "\t".join([f"{SI:.2f}" for SI in SI_P])
        str_COP = "\t".join([f"{SI_COP:.2f}"])
        f.write(f"Q_cond \t {str_h} \t {str_p} \t {str_mass_flow_rates} \t {str_Q} \t {str_P} \t {str_COP}\n")

if plot_figures :
    color = ['blue', 'red', 'green']

    heos = CoolProp.AbstractState("HEOS", list(states[0].values())[0].fluid)
    T_crit = heos.T_critical()
    p_crit = heos.p_critical()

    plt.figure(figsize=(8, 6))
    for i in range(2):
        for k, (label, state) in enumerate(states[i].items()):
            plt.scatter(state.s/1e3, state.T - 273.15, color = color[i], clip_on = False)

        for j in range(len(T_points[i])):
            if j == 0 :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', label= f'Q_cond = {Q_cond_cycles[i][0]/1e3} kW', color = color[i], clip_on = False)
            else :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', color = color[i], clip_on = False)

    for j in range(len(T_points_ref)):
        if j == 0 :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', label= f'Q_cond = {reference_cycle[0]/1e3} kW (Reference)', color = color[2], clip_on = False)
        else :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', color = color[2], clip_on = False)

    for k, (label, state) in enumerate(states_ref.items()):
        plt.scatter(state.s/1e3, state.T - 273.15, color = color[2], clip_on = False)

    s_max_state = np.max(s_points)
    s_min_state = np.min(s_points)
    T_max_state = np.max(T_points)
    T_min_state = np.min(T_points)

    heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
    s_min_liq = heos.smass()
    heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
    #s_max_vap = heos.smass()

    plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
    plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

    T_sat = np.linspace(T_min_state, T_crit - 1, 100)
    s_liq = np.zeros(100)
    s_vap = np.zeros(100)
    for i, T in enumerate(T_sat):
        heos.update(CoolProp.QT_INPUTS, 0, T)
        s_liq[i] = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1,T)
        s_vap[i] = heos.smass()

    plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = False)
    plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = False)
    heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
    s_crit = heos.smass()
    plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point


    plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Temperature [°C]', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Improve readability
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
    plt.savefig("code/Figures/CAD/Ts_diagram_Q_cond_variation.png", dpi = 300, bbox_inches = 'tight')
    plt.show()


# N_LP analysis
T_points = []
s_points = []
states = []
for i in range(2): 
    compressor_inputs = {'N_LP': N_LP_cycles[i][1], 'N_HP': N_LP_cycles[i][2]}
    valve_inputs = {'z_LP': N_LP_cycles[i][3], 'z_HP': N_LP_cycles[i][4]}
    initial_guess = np.log(N_LP_cycles[i][5:15])
    cycle = CAD(N_LP_cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, save_results = False, plot_figures = False)
    T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
    T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

if plot_figures :
    color = ['blue', 'red', 'green']

    heos = CoolProp.AbstractState("HEOS", list(states[0].values())[0].fluid)
    T_crit = heos.T_critical()
    p_crit = heos.p_critical()

    plt.figure(figsize=(8, 6))
    for i in range(2):
        for k, (label, state) in enumerate(states[i].items()):
            plt.scatter(state.s/1e3, state.T - 273.15, color = color[i], clip_on = False)

        for j in range(len(T_points[i])):
            if j == 0 :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', label= f'N_LP = {N_LP_cycles[i][1]} Hz', color = color[i], clip_on = False)
            else :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', color = color[i], clip_on = False)

    for j in range(len(T_points_ref)):
        if j == 0 :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', label= f'N_LP = {reference_cycle[1]} Hz (Reference)', color = color[2], clip_on = False)
        else :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', color = color[2], clip_on = False)

    for k, (label, state) in enumerate(states_ref.items()):
        plt.scatter(state.s/1e3, state.T - 273.15, color = color[2], clip_on = False)

    s_max_state = np.max(s_points)
    s_min_state = np.min(s_points)
    T_max_state = np.max(T_points)
    T_min_state = np.min(T_points)

    heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
    s_min_liq = heos.smass()
    heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
    #s_max_vap = heos.smass()

    plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
    plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

    T_sat = np.linspace(T_min_state, T_crit - 1, 100)
    s_liq = np.zeros(100)
    s_vap = np.zeros(100)
    for i, T in enumerate(T_sat):
        heos.update(CoolProp.QT_INPUTS, 0, T)
        s_liq[i] = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1,T)
        s_vap[i] = heos.smass()

    plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = False)
    plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = False)
    heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
    s_crit = heos.smass()
    plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point


    plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Temperature [°C]', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Improve readability
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
    plt.savefig("code/Figures/CAD/Ts_diagram_N_LP_variation.png", dpi = 300, bbox_inches = 'tight')
    plt.show()

# N_HP analysis
T_points = []
s_points = []
states = []
for i in range(2): 
    compressor_inputs = {'N_LP': N_HP_cycles[i][1], 'N_HP': N_HP_cycles[i][2]}
    valve_inputs = {'z_LP': N_HP_cycles[i][3], 'z_HP': N_HP_cycles[i][4]}
    initial_guess = np.log(N_HP_cycles[i][5:15])
    cycle = CAD(N_HP_cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, save_results = False, plot_figures = False)
    T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
    T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

if plot_figures :

    color = ['blue', 'red', 'green']

    heos = CoolProp.AbstractState("HEOS", list(states[0].values())[0].fluid)
    T_crit = heos.T_critical()
    p_crit = heos.p_critical()

    plt.figure(figsize=(8, 6))
    for i in range(2):
        for k, (label, state) in enumerate(states[i].items()):
            plt.scatter(state.s/1e3, state.T - 273.15, color = color[i], clip_on = False)

        for j in range(len(T_points[i])):
            if j == 0 :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', label= f'N_HP = {N_HP_cycles[i][2]} Hz', color = color[i], clip_on = False)
            else :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', color = color[i], clip_on = False)

    for j in range(len(T_points_ref)):
        if j == 0 :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', label= f'N_HP = {reference_cycle[2]} Hz (Reference)', color = color[2], clip_on = False)
        else :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', color = color[2], clip_on = False)

    for k, (label, state) in enumerate(states_ref.items()):
        plt.scatter(state.s/1e3, state.T - 273.15, color = color[2], clip_on = False)

    s_max_state = np.max(s_points)
    s_min_state = np.min(s_points)
    T_max_state = np.max(T_points)
    T_min_state = np.min(T_points)

    heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
    s_min_liq = heos.smass()
    heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
    #s_max_vap = heos.smass()

    plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
    plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

    T_sat = np.linspace(T_min_state, T_crit - 1, 100)
    s_liq = np.zeros(100)
    s_vap = np.zeros(100)
    for i, T in enumerate(T_sat):
        heos.update(CoolProp.QT_INPUTS, 0, T)
        s_liq[i] = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1,T)
        s_vap[i] = heos.smass()

    plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = False)
    plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = False)
    heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
    s_crit = heos.smass()
    plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point


    plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Temperature [°C]', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Improve readability
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
    plt.savefig("code/Figures/CAD/Ts_diagram_N_HP_variation.png", dpi = 300, bbox_inches = 'tight')
    plt.show()

# z_LP analysis
T_points = []
s_points = []
states = []
for i in range(2): 
    compressor_inputs = {'N_LP': z_LP_cycles[i][1], 'N_HP': z_LP_cycles[i][2]}
    valve_inputs = {'z_LP': z_LP_cycles[i][3], 'z_HP': z_LP_cycles[i][4]}
    initial_guess = np.log(z_LP_cycles[i][5:15])
    cycle = CAD(z_LP_cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, save_results = False, plot_figures = False)
    T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
    T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

if plot_figures :
    color = ['blue', 'red', 'green']

    heos = CoolProp.AbstractState("HEOS", list(states[0].values())[0].fluid)
    T_crit = heos.T_critical()
    p_crit = heos.p_critical()

    plt.figure(figsize=(8, 6))
    for i in range(2):
        for k, (label, state) in enumerate(states[i].items()):
            plt.scatter(state.s/1e3, state.T - 273.15, color = color[i], clip_on = False)

        for j in range(len(T_points[i])):
            if j == 0 :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', label= f'z_LP = {z_LP_cycles[i][3]} %', color = color[i], clip_on = False)
            else :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', color = color[i], clip_on = False)

    for j in range(len(T_points_ref)):
        if j == 0 :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', label= f'z_LP = {reference_cycle[3]} % (Reference)', color = color[2], clip_on = False)
        else :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', color = color[2], clip_on = False)

    for k, (label, state) in enumerate(states_ref.items()):
        plt.scatter(state.s/1e3, state.T - 273.15, color = color[2], clip_on = False)

    s_max_state = np.max(s_points)
    s_min_state = np.min(s_points)
    T_max_state = np.max(T_points)
    T_min_state = np.min(T_points)

    heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
    s_min_liq = heos.smass()
    heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
    #s_max_vap = heos.smass()

    plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
    plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

    T_sat = np.linspace(T_min_state, T_crit - 1, 100)
    s_liq = np.zeros(100)
    s_vap = np.zeros(100)
    for i, T in enumerate(T_sat):
        heos.update(CoolProp.QT_INPUTS, 0, T)
        s_liq[i] = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1,T)
        s_vap[i] = heos.smass()

    plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = False)
    plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = False)
    heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
    s_crit = heos.smass()
    plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point


    plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Temperature [°C]', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Improve readability
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
    plt.savefig("code/Figures/CAD/Ts_diagram_z_LP_variation.png", dpi = 300, bbox_inches = 'tight')
    plt.show()


# z_HP analysis
T_points = []
s_points = []
states = []
for i in range(2): 
    compressor_inputs = {'N_LP': z_HP_cycles[i][1], 'N_HP': z_HP_cycles[i][2]}
    valve_inputs = {'z_LP': z_HP_cycles[i][3], 'z_HP': z_HP_cycles[i][4]}
    initial_guess = np.log(z_HP_cycles[i][5:15])
    cycle = CAD(z_HP_cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, initial_guess, save_results = False, plot_figures = False)
    T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
    T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

if plot_figures :
    color = ['blue', 'red', 'green']

    heos = CoolProp.AbstractState("HEOS", list(states[0].values())[0].fluid)
    T_crit = heos.T_critical()
    p_crit = heos.p_critical()

    plt.figure(figsize=(8, 6))
    for i in range(2):
        for k, (label, state) in enumerate(states[i].items()):
            plt.scatter(state.s/1e3, state.T - 273.15, color = color[i], clip_on = False)

        for j in range(len(T_points[i])):
            if j == 0 :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', label= f'z_HP = {z_HP_cycles[i][4]} %', color = color[i], clip_on = False)
            else :
                plt.plot(s_points[i][j]/1e3, T_points[i][j] - 273.15, linestyle = '-', color = color[i], clip_on = False)

    for j in range(len(T_points_ref)):
        if j == 0 :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', label= f'z_HP = {reference_cycle[4]} % (Reference)', color = color[2], clip_on = False)
        else :
            plt.plot(s_points_ref[j]/1e3, T_points_ref[j] - 273.15, linestyle = '-', color = color[2], clip_on = False)

    for k, (label, state) in enumerate(states_ref.items()):
        plt.scatter(state.s/1e3, state.T - 273.15, color = color[2], clip_on = False)

    s_max_state = np.max(s_points)
    s_min_state = np.min(s_points)
    T_max_state = np.max(T_points)
    T_min_state = np.min(T_points)

    heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
    s_min_liq = heos.smass()
    heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
    #s_max_vap = heos.smass()

    plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
    plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

    T_sat = np.linspace(T_min_state, T_crit - 1, 100)
    s_liq = np.zeros(100)
    s_vap = np.zeros(100)
    for i, T in enumerate(T_sat):
        heos.update(CoolProp.QT_INPUTS, 0, T)
        s_liq[i] = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1,T)
        s_vap[i] = heos.smass()

    plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = False)
    plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = False)
    heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
    s_crit = heos.smass()
    plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point


    plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Temperature [°C]', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Improve readability
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
    plt.savefig("code/Figures/CAD/Ts_diagram_z_HP_variation.png", dpi = 300, bbox_inches = 'tight')
    plt.show()
'''