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
# Parameters of the reference cycle
############################################################

# Condenser load
Q_cond_ref = 25e3                                  # Condenser load of the reference cycle [W

# Frequencies of the compressors
N_LP_ref = 38                                       # Frequency of the low-pressure compressor [Hz]
N_HP_ref = 38                                       # Frequency of the high-pressure compressor [Hz]

# Valve openings
z_LP_ref = 13.5                                     # Opening of the low-pressure valve [%]
z_HP_ref = 20                                       # Opening of the high-pressure valve [%]

# Heat sources parameters

    # 1. LT source
T1_prime = 15 + 273.75                          # Inlet temperature of the external fluid in the heat source [K] 
p1_prime = 3e5                                  # Inlet pressure of the external fluid in the heat source [Pa]
glide_external_fluid_LT = 5                      # Mass flow rate of the external fluid in the heat source [kg/s] 
external_fluid_LT_param = {'T1_prime': T1_prime, 'p1_prime': p1_prime, 'glide_external_fluid_LT': glide_external_fluid_LT}

    # 2. MT source
T3_prime = 60 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
p3_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
glide_external_fluid_MT = 5                      # Mass flow rate of the external fluid in the heat sink [kg/s]
external_fluid_MT_param = {'T3_prime': T3_prime, 'p3_prime': p3_prime, 'glide_external_fluid_MT': glide_external_fluid_MT}

# Heat sink parameters
T6_prime = 120 + 273.15                          # Inlet temperature of the external fluid in the heat sink [K] 
p5_prime = 3e5                                  # Inlet pressure of the external fluid in the heat sink [Pa]
glide_external_fluid_HT = 40                    # Mass flow rate of the external fluid in the heat sink [kg/s]
T5_prime = T6_prime - glide_external_fluid_HT  # Assuming a temperature drop of 10 K in the condenser
external_fluid_HT_param = {'T5_prime': T5_prime, 'p5_prime': p5_prime, 'glide_external_fluid_HT': glide_external_fluid_HT}

recuperator_LP = True

############################################################
# Parameters for the OAT analysis
############################################################

dN_max = 20                                       # Maximum variation of the compressor frequencies [Hz]
dz_max = 15                                        # Maximum variation of the valve openings [%]
n_points = 5                                      # Number of points for the OAT analysis (n_points = 1 means only maximum and minimum values, n_points = 2 means maximum, minimum and reference values, etc.)
file_results = "code/Figures/CAD/results_super.txt"
file_sensitivity_analysis = "code/Figures/CAD/sensitivity_analysis.txt"


############################################################
# Computing the cycles for the OAT analysis
############################################################

# Reference cycle
initial_guess = np.log(np.array([1781480.99, 4473182.32, 725165.01, 575682.12, 1783977.57, 0.0297, 0.0687, 0.4051, 0.3847, 0.1482, 574457.81, 598348.64]))
#cycle_ref = CAD(Q_cond_ref, {'N_LP': N_LP_ref, 'N_HP': N_HP_ref}, {'z_LP': z_LP_ref, 'z_HP': z_HP_ref}, external_fluid_LT_param, external_fluid_MT_param, \
                #external_fluid_HT_param, recuperator_LP, initial_guess, file_results = file_results, plot_figures = False, solve = False)

initial_guess_up = np.zeros((4, len(initial_guess)))
initial_guess_low = np.zeros((4, len(initial_guess)))
for i in range(4) :
    initial_guess_up[i] = initial_guess 
    initial_guess_low[i] = initial_guess 

for n in range(n_points) :
        if n != 0 :        
            print(f"Computing cycle for point {n+1}/{n_points} of the OAT analysis...")
            N_LP = N_LP_ref + dN_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP, 'N_HP': N_HP_ref}
            valve_inputs = {'z_LP': z_LP_ref, 'z_HP': z_HP_ref}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_up[0], file_results = file_results, plot_figures = False, solve = False)
            
            N_LP = N_LP_ref - dN_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP, 'N_HP': N_HP_ref}
            valve_inputs = {'z_LP': z_LP_ref, 'z_HP': z_HP_ref}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_low[0], file_results = file_results, plot_figures = False, solve = False)
            
            N_HP = N_HP_ref + dN_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP_ref, 'N_HP': N_HP}
            valve_inputs = {'z_LP': z_LP_ref, 'z_HP': z_HP_ref}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_up[1], file_results = file_results, plot_figures = False, solve = False)
            
            N_HP = N_HP_ref - dN_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP_ref, 'N_HP': N_HP}
            valve_inputs = {'z_LP': z_LP_ref, 'z_HP': z_HP_ref}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_low[1], file_results = file_results, plot_figures = False, solve = False)
            
            z_LP = z_LP_ref + dz_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP_ref, 'N_HP': N_HP_ref}
            valve_inputs = {'z_LP': z_LP, 'z_HP': z_HP_ref}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_up[2], file_results = file_results, plot_figures = False, solve = False)            
            
            z_LP = z_LP_ref - dz_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP_ref, 'N_HP': N_HP_ref}
            valve_inputs = {'z_LP': z_LP, 'z_HP': z_HP_ref}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_low[2], file_results = file_results, plot_figures = False, solve = False)
            
            z_HP = z_HP_ref + dz_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP_ref, 'N_HP': N_HP_ref}
            valve_inputs = {'z_LP': z_LP_ref, 'z_HP': z_HP}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_up[3], file_results = file_results, plot_figures = False, solve = False)            
            
            z_HP = z_HP_ref - dz_max * (n + 1) / n_points
            compressor_inputs = {'N_LP': N_LP_ref, 'N_HP': N_HP_ref}
            valve_inputs = {'z_LP': z_LP_ref, 'z_HP': z_HP}
            CAD(Q_cond_ref, compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                        recuperator_LP, initial_guess_low[3], file_results = file_results, plot_figures = False, solve = False)  

        data = np.loadtxt(file_results, skiprows=1, usecols = np.arange(5, 5 + len(initial_guess)))  # Skip the header row
        for i in range(4) :
            initial_guess_up[i] = np.log(data[-2*4 +2*i])  # Get the last 4 lines of the file (corresponding to the last 4 cycles computed) and assign the values to the initial guesses for the next cycle
            initial_guess_low[i] = np.log(data[-2*4 +2*i + 1])



'''       


file_name = "code/Figures/CAD/results_sub.txt"
data = np.loadtxt(file_name, skiprows=1)  # Skip the header row
reference_cycle = data[0]  # Assuming the first row corresponds to the reference cycle
N_LP_cycles = data[1:3]
N_HP_cycles = data[3:5]
z_LP_cycles = data[5:7]
z_HP_cycles = data[7:9]

def OAT_analysis(cycles, variable_name, threshold = 0.5, save_analysis = False, plot_figure = False, save_figure = False) : 
    # Reference cycle analysis
    compressor_inputs_ref = {'N_LP': reference_cycle[1], 'N_HP': reference_cycle[2]}
    valve_inputs_ref = {'z_LP': reference_cycle[3], 'z_HP': reference_cycle[4]}
    initial_guess_ref = np.log(reference_cycle[5:15])
    cycle_ref = CAD(reference_cycle[0], compressor_inputs_ref, valve_inputs_ref, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, recuperator_LP, initial_guess_ref, save_results = False, plot_figures = False, solve = True)
    T_points_ref, s_points_ref, states_ref = cycle_ref.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram

    T_ref = [cycle_ref.state_1.T, cycle_ref.state_3.T, cycle_ref.state_5.T, cycle_ref.state_7.T, cycle_ref.state_8.T, cycle_ref.state_10.T, cycle_ref.state_3_evap.T,\
            cycle_ref.state_3_comp.T, cycle_ref.state_2_prime.T, cycle_ref.state_4_prime.T, cycle_ref.state_6_prime.T]
    p_ref = [cycle_ref.state_1.p, cycle_ref.state_3.p, cycle_ref.state_5.p, cycle_ref.state_7.p, cycle_ref.state_8.p, cycle_ref.state_10.p, cycle_ref.state_3_evap.p,\
            cycle_ref.state_3_comp.p, cycle_ref.state_2_prime.p, cycle_ref.state_4_prime.p, cycle_ref.state_6_prime.p]
    mass_flow_rates_ref = [cycle_ref.mdot_wf_top, cycle_ref.mdot_wf_bottom, cycle_ref.mdot_LT, cycle_ref.mdot_MT, cycle_ref.mdot_HT]
    Q_ref = [cycle_ref.Q_LT, cycle_ref.Q_MT, cycle_ref.Q_HT]
    P_ref = [cycle_ref.P_comp_top, cycle_ref.P_comp_bottom]
    COP_ref = cycle_ref.COP
    alpha_ref = cycle_ref.alpha

    # Variable cycle analysis
    T_points = []
    s_points = []
    states = []
    T = []
    p = []
    mass_flow_rates = []
    Q = []
    P = []
    COP = []
    alpha = []

    for i in range(2): 
        compressor_inputs = {'N_LP': cycles[i][1], 'N_HP': cycles[i][2]}
        valve_inputs = {'z_LP': cycles[i][3], 'z_HP': cycles[i][4]}
        initial_guess = np.log(cycles[i][5:15])
        cycle = CAD(cycles[i][0], compressor_inputs, valve_inputs, external_fluid_LT_param, external_fluid_MT_param, external_fluid_HT_param, \
                    recuperator_LP, initial_guess, save_results = False, plot_figures = False, solve = True)
        
        T_points_cycle, s_points_cycle, states_cycle = cycle.Ts_diagram(save = False, only_points = True)  # Get the points for the Ts diagram
        T_points.append(T_points_cycle) ; s_points.append(s_points_cycle) ; states.append(states_cycle)

        T.append([cycle.state_1.T, cycle.state_3.T, cycle.state_5.T, cycle.state_7.T, cycle.state_8.T, cycle.state_10.T, cycle.state_3_evap.T,\
                cycle.state_3_comp.T, cycle.state_2_prime.T, cycle.state_4_prime.T, cycle.state_6_prime.T])
        p.append([cycle.state_1.p, cycle.state_3.p, cycle.state_5.p, cycle.state_7.p, cycle.state_8.p, cycle.state_10.p, cycle.state_3_evap.p,\
                cycle.state_3_comp.p, cycle.state_2_prime.p, cycle.state_4_prime.p, cycle.state_6_prime.p])
        mass_flow_rates.append([cycle.mdot_wf_top, cycle.mdot_wf_bottom, cycle.mdot_LT, cycle.mdot_MT, cycle.mdot_HT])
        Q.append([cycle.Q_LT, cycle.Q_MT, cycle.Q_HT])
        P.append([cycle.P_comp_top, cycle.P_comp_bottom])
        COP.append(cycle.COP)
        alpha.append(cycle.alpha)

    if save_analysis :
        index_variable = 0
        if variable_name == "N_LP" :
            index_variable = 1
        elif variable_name == "N_HP" :
            index_variable = 2
        elif variable_name == "z_LP" :
            index_variable = 3
        elif variable_name == "z_HP" :
            index_variable = 4

        T = np.array(T) ; p = np.array(p) ; mass_flow_rates = np.array(mass_flow_rates) ; Q = np.array(Q) ; P = np.array(P) ; COP = np.array(COP) ; alpha = np.array(alpha)
        SI_T = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], T[1], T[0])
        SI_p = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], p[1], p[0])
        SI_mass_flow_rates = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], mass_flow_rates[1], mass_flow_rates[0])
        SI_Q = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], Q[1], Q[0])
        SI_P = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], P[1], P[0])
        SI_COP = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], COP[1], COP[0])
        SI_alpha = sensitivity_index(cycles[1][index_variable], cycles[0][index_variable], alpha[1], alpha[0])

        with open(file_sensitivity_analysis, "a") as f:
            """
            f.write("Variable \t T_1 \t T_3 \t T_5 \t T_7 \t T_8 \t T_10 \t T_3_evap \t T_3_comp \t T_2_prime \t T_4_prime \t T_6_prime" \
                    "\t p_1 \t p_3 \t p_5 \t p_7 \t p_8 \t p_10 \t p_3_evap \t p_3_comp \t p_2_prime \t p_4_prime \t p_6_prime \t mdot_wf_top"\
                    "\t mdot_wf_bottom \t mdot_LT \t mdot_MT \t mdot_HT \t Q_LT \t Q_MT \t Q_HT \t P_comp_top \t P_comp_bottom \t COP \t alpha\n")
            """
            str_T = "\t".join([f"{SI:.2f}" for SI in SI_T])
            str_p = "\t".join([f"{SI:.2f}" for SI in SI_p])
            str_mass_flow_rates = "\t".join([f"{SI:.2f}" for SI in SI_mass_flow_rates])
            str_Q = "\t".join([f"{SI:.2f}" for SI in SI_Q])
            str_P = "\t".join([f"{SI:.2f}" for SI in SI_P])
            str_COP = "\t".join([f"{SI_COP:.2f}"])
            str_alpha = "\t".join([f"{SI_alpha:.2f}"])
            f.write(variable_name + f" \t {str_T} \t {str_p} \t {str_mass_flow_rates} \t {str_Q} \t {str_P} \t {str_COP} \t {str_alpha}\n")


    if plot_figure : 

        def draw_radar(ax, variables, values_min, values_max, max_value) :
            num_vars = len(variables)
            values_min = values_min.tolist()
            values_max = values_max.tolist()
            
            # Calcul des angles
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            
            # Fermeture des boucles (répéter le premier élément)
            angles_c = angles + [angles[0]]
            v_min_c = values_min + [values_min[0]]
            v_max_c = values_max + [values_max[0]]
            
            # 1. Tracé des ombres (Min et Max)
            #ax.fill(angles_c, v_max_c, color=color, alpha=0.2)
            ax.plot(angles_c, v_max_c, linewidth=1, linestyle='-', color = 'red')
            
            #ax.fill(angles_c, v_min_c, color=color, alpha=0.4)
            ax.plot(angles_c, v_min_c, linewidth=1, linestyle='-', color = 'blue')
            ax.plot(angles_c, np.ones(len(angles_c)), linestyle = '--', color = 'green', linewidth=1)
            
            # 2. Configuration esthétique
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Gestion des labels
            ax.set_thetagrids(np.degrees(angles), variables)

            # Cacher les graduations radiales
            ax.set_yticklabels([])
            ax.spines['polar'].set_visible(False)
            
            """
            # Ajout des flèches d'axes
            limit = max(v_max_c) * 1.2
            for angle in angles:
                ax.annotate('', xy=(angle, limit), xytext=(0,0),
                            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
            """
            ax.yaxis.grid(False)
            ax.set_ylim(0, max_value * 1.1)  # Ajuster la limite radiale pour éviter les chevauchements
            

        # --- DONNÉES POUR LES 4 GRAPHIQUES ---

        # Graphe 1 (Températures)
        vars1 = [r'$T_1$', r'$T_3$', r'$T_5$', r'$T_{3,evap}$', r'$T_{3,comp}$']
        min1 = np.array(T[1][:3] + [T[0][6], T[0][7]]) / np.array(T_ref[:3] + [T_ref[6], T_ref[7]]) 
        max1 = np.array(T[0][:3] + [T[1][6], T[1][7]]) / np.array(T_ref[:3] + [T_ref[6], T_ref[7]])

        # Graphe 2 (Pressions)
        vars2 = [r'$p_1$', r'$p_3$', r'$p_5$']
        min2 = np.array([p[1][0], p[1][1], p[1][2]]) / np.array([p_ref[0], p_ref[1], p_ref[2]])
        max2 = np.array([p[0][0], p[0][1], p[0][2]]) / np.array([p_ref[0], p_ref[1], p_ref[2]])

        # Graphe 3 (Masses)
        vars3 = [r'$\dot{m}_{wf,top}$', r'$\dot{m}_{wf,bottom}$', r'$\dot{m}_{LT}$', r'$\dot{m}_{MT}$', r'$\dot{m}_{HT}$']
        min3 = np.array([mass_flow_rates[1][0], mass_flow_rates[1][1], mass_flow_rates[1][2], mass_flow_rates[1][3], mass_flow_rates[1][4]]) / np.array([mass_flow_rates_ref[0], mass_flow_rates_ref[1], mass_flow_rates_ref[2], mass_flow_rates_ref[3], mass_flow_rates_ref[4]])
        max3 = np.array([mass_flow_rates[0][0], mass_flow_rates[0][1], mass_flow_rates[0][2], mass_flow_rates[0][3], mass_flow_rates[0][4]]) / np.array([mass_flow_rates_ref[0], mass_flow_rates_ref[1], mass_flow_rates_ref[2], mass_flow_rates_ref[3], mass_flow_rates_ref[4]])

        # Graphe 4 (Puissance/COP)
        vars4 = [r'$Q_{LT}$', r'$Q_{MT}$', r'$Q_{HT}$', r'$P_{comp,top}$', r'$P_{comp,bottom}$', r'$COP$', r'$\alpha$']
        min4 = np.array([Q[1][0], Q[1][1], Q[1][2], P[1][0], P[1][1], COP[1], alpha[1]]) / np.array([Q_ref[0], Q_ref[1], Q_ref[2], P_ref[0], P_ref[1], COP_ref, alpha_ref])
        max4 = np.array([Q[0][0], Q[0][1], Q[0][2], P[0][0], P[0][1], COP[0], alpha[0]]) / np.array([Q_ref[0], Q_ref[1], Q_ref[2], P_ref[0], P_ref[1], COP_ref, alpha_ref])

        max_value = 1.2  # Valeur maximale pour les graphiques (pour éviter les chevauchements)
        # --- CRÉATION DES SUBPLOTS ---
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), subplot_kw=dict(polar=True))
        if variable_name == "Q_cond" :
            variable_name_title = r'$Q_{cond}$'
        elif variable_name == "N_LP" :
            variable_name_title = r'$N_{LP}$'
        elif variable_name == "N_HP" :
            variable_name_title = r'$N_{HP}$'
        elif variable_name == "z_LP" :
            variable_name_title = r'$z_{LP}$'
        elif variable_name == "z_HP" :
            variable_name_title = r'$z_{HP}$'

        fig.suptitle(fr"Sensitivity analysis for {variable_name_title}", size=12)
        draw_radar(axes[0][0], vars1, min1, max1, max_value = max_value)
        draw_radar(axes[0][1], vars2, min2, max2, max_value = max_value)
        draw_radar(axes[1][0], vars3, min3, max3, max_value = max_value)
        draw_radar(axes[1][1], vars4, min4, max4, max_value = max_value)
        plt.tight_layout()
        if save_figure :
            plt.savefig(f"code/Figures/CAD/OAT_analysis_{variable_name}.png", dpi=300, bbox_inches='tight')
        plt.show()

        color = ['red', 'blue', 'green']

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

        #plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
        #plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

        plt.xlim((1.2, 2.6))
        plt.ylim((0, 120))

        T_sat = np.linspace(T_min_state, T_crit - 1, 100)
        s_liq = np.zeros(100)
        s_vap = np.zeros(100)
        for i, T in enumerate(T_sat):
            heos.update(CoolProp.QT_INPUTS, 0, T)
            s_liq[i] = heos.smass()
            heos.update(CoolProp.QT_INPUTS, 1,T)
            s_vap[i] = heos.smass()

        plt.plot(s_liq/1e3, T_sat-273.15, 'black', clip_on = True)
        plt.plot(s_vap/1e3, T_sat-273.15, 'black', clip_on = True)
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
        order = [1, 2, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon = False, fontsize = 10)
        if save_figure :
            plt.savefig("code/Figures/CAD/Ts_diagram_" + variable_name + "_variation.png", dpi = 300, bbox_inches = 'tight')
        plt.show()
        
    

    return 
'''

OAT_analysis(N_LP_cycles, "N_LP", plot_figure = True, save_figure = True, save_analysis = False)
OAT_analysis(N_HP_cycles, "N_HP", plot_figure = True, save_figure = True, save_analysis = False)
OAT_analysis(z_LP_cycles, "z_LP", plot_figure = True, save_figure = True, save_analysis = False)
OAT_analysis(z_HP_cycles, "z_HP", plot_figure = True, save_figure = True, save_analysis = False)

