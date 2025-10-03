# -*- coding: utf-8 -*-
"""
===============================================================================
@ LELME2240 - Energy systems lab. 

Script for the simulation of a thermodynamic heat pump cycle.

-> Students version.
===============================================================================
- Date:     February 2022
- Author:   Antoine Laterre
===============================================================================
    Legend:
        p_ev    = evaporation pressure          [Pa]
        T_1     = compressor supply temperature [K]
        p_cd    = condensation pressure         [Pa]
        T_2     = compressor exit temperature   [K]
        T_3     = valve supply temperature      [K]
        T_0     = evaporation temperature       [K]
        T_c     = condensation temperature      [K]
        fluid   = refrigerant                   [K]
===============================================================================
"""

#%%
#===IMPORT PACKAGES============================================================
#

import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

#%%
#===FUNCTIONS==================================================================
#

def heat_pump(inputs,options, other_inputs = None):
    p_ev, T_1, p_cd, T_2, T_3, T_0, T_c, fluid = inputs
    plot_option, print_option = options
    
    #===YOUR CODE=============================================================
    # 1. Compression

    x_1 = PropsSI('Q', 'P', p_ev, 'T', T_1, fluid) 
    if x_1 == -1:
        s_1 = PropsSI('S', 'P', p_ev, 'T', T_1, fluid) 
        h_1 = PropsSI('H', 'P', p_ev, 'T', T_1, fluid)
    else: 
        s_1 = PropsSI('S', 'P', p_ev, 'Q', x_1, fluid) 
        h_1 = PropsSI('H', 'P', p_ev, 'Q', x_1, fluid)

    T_2s = PropsSI('T', 'P', p_cd, 'S', s_1, fluid)
    h_2s = PropsSI('H', 'P', p_cd, 'T', T_2s, fluid)
    h_2 = PropsSI('H', 'P', p_cd, 'T', T_2, fluid)
    s_2 = PropsSI('S', 'P', p_cd, 'T', T_2, fluid)

    eta_c_is = (h_2s - h_1) / (h_2 - h_1)
    
    # 2. Condensation
    x_3 = PropsSI('Q', 'P', p_cd, 'T', T_3, fluid)
    if x_3 == -1:
        s_3 = PropsSI('S', 'P', p_cd, 'T', T_3, fluid) 
        h_3 = PropsSI('H', 'P', p_cd, 'T', T_3, fluid)
    else: 
        s_3 = PropsSI('S', 'P', p_cd, 'Q', x_3, fluid) 
        h_3 = PropsSI('H', 'P', p_cd, 'Q', x_3, fluid)
    
    # 3. Expansion
    x_4 = PropsSI('Q', 'P', p_ev, 'H', h_3, fluid)
    T_4 = PropsSI('T', 'P', p_ev, 'Q', x_4, fluid)
    h_4 = h_3
    s_4 = PropsSI('S', 'P', p_ev, 'Q', x_4, fluid)

    # COP calculation
    COP         = 1 + (h_1 - h_3) / (h_2 - h_1)

    # Lift 
    lift = T_c - T_0
    
    if other_inputs != None:
        P = other_inputs[0]
        eta_mec = other_inputs[3]/ COP
        rho_gw = 1 * 62 /100 + 38/100 * 1.04 
        print('rho_gw = ', rho_gw, 'kg/m3')
        m_c = rho_gw * 21.6 / 60
        m_h = 22.3 / 60
        delta_T_c = other_inputs[1]
        delta_T_h = other_inputs[2]

        cp_c = 0.9 * 4.184 * 1000 # Specific heat capacity of the glycol water [J/kg.K]
        cp_h = 4.184 * 1000 # Specific heat capacity of the water [J/kg.K]
        
        # efficiencies
        m_cy = eta_mec * P / (h_2 - h_1)  # mass flow rate of the cycle [kg/s]
        p_mec = (1-eta_mec) * P  # mechanical power [W]

        eta_ev = m_cy * (h_1 - h_4) / (m_c * cp_c * delta_T_c)  # evaporation efficiency
        P_ev = (1-eta_ev) * m_c * cp_c * delta_T_c  # evaporation power [W]

        eta_cd = m_h * cp_h * delta_T_h / (m_cy * (h_2 - h_3))  # condensation efficiency
        P_cd = (1/eta_cd-1) * m_h * cp_h * delta_T_h  # condensation power [W]
    
    states = [
        [s_1, T_1],
        [s_2, T_2],
        [s_3, T_3],
        [s_4, T_4],
        [s_1, T_1], 
    ]
    pressure = [p_ev, p_cd]

    data_plot = {
        'states': states,
        'pressure': pressure,
        'lift': lift
    }

    if print_option:    
        print('COP = ', COP)
        print('eta_c_is = ', eta_c_is)
        print('Lift = ', lift, '°C')
        
        if other_inputs is not None:
            print('P_e = ', P, 'W')
            print('m_cy = ', m_cy, 'kg/s')
            print('p_mec = ', p_mec, 'W')
            print('eta_ev = ', eta_ev)
            print('p_ev = ', P_ev, 'W')
            print('eta_cd = ', eta_cd)
            print('p_cd = ', P_cd, 'W')
            print('eta_mec = ', eta_mec)
        
        states = [
            {"State": "1", "T (K)": T_1, "P (Pa)": p_ev, "x": x_1, "s (J/kg.K)": s_1, "h (J/kg)": h_1},
            {"State": "2", "T (K)": T_2, "P (Pa)": p_cd, "x": "-", "s (J/kg.K)": s_2, "h (J/kg)": h_2},
            {"State": "3", "T (K)": T_3, "P (Pa)": p_cd, "x": x_3, "s (J/kg.K)": s_3, "h (J/kg)": h_3},
            {"State": "4", "T (K)": T_4, "P (Pa)": p_ev, "x": x_4, "s (J/kg.K)": s_4, "h (J/kg)": h_4},
        ]

        
        

        
        print(f"{'State':<6} {'T (K)':<10} {'P (Pa)':<12} {'x':<8} {'s (J/kg.K)':<15} {'h (J/kg)':<15}")
        print("-" * 70)
        for state in states:
            print(f"{state['State']:<6} {state['T (K)']:<10.2f} {state['P (Pa)']:<12.2e} {state['x']:<8} {state['s (J/kg.K)']:<15.2f} {state['h (J/kg)']:<15.2f}")
    if plot_option:     fig = plot_diag_Ts(data_plot)


    
def plot_diag_Ts(data_plot):
    import numpy as np
    from CoolProp.CoolProp import PropsSI

    fluid = 'R410A'
    
    # === SATURATION DOME ===
    p_min = PropsSI('P_MIN', fluid)
    p_crit = PropsSI('P_CRITICAL', fluid)
    
    T_range = np.linspace(PropsSI('T_TRIPLE', fluid), PropsSI('T_CRITICAL', fluid), 500)
    s_l, s_v, T_sat = [], [], []

    for T in T_range:
        try:
            s_l.append(PropsSI('S', 'T', T, 'Q', 0, fluid)/1000)
            s_v.append(PropsSI('S', 'T', T, 'Q', 1, fluid)/1000)
            T_sat.append(T - 273.15)
        except:
            continue

    fig = plt.figure()

    plt.plot(s_l, T_sat, 'black', linewidth=1, label='Saturation curve')
    plt.plot(s_v, T_sat,'black', linewidth=1)

    # === CYCLE ===
    n_points = 100
    states = data_plot['states']

    p_ev = data_plot['pressure'][0]
    p_cd = data_plot['pressure'][1]

    # 1 -> 2 
    p_vals = np.linspace(p_ev, p_cd, n_points)
    s_vals, T_vals = [], []

    s1 = states[0][0]
    s2 = states[1][0]

    for i, p in enumerate(p_vals):
        s_interp = (1 - i/(n_points - 1)) * s1 + (i/(n_points - 1)) * s2
        try:
            T = PropsSI('T', 'P', p, 'S', s_interp, fluid)
            s_vals.append(s_interp / 1000)
            T_vals.append(T - 273.15)
        except:
            continue

    plt.plot(s_vals, T_vals, 'red')

    # Compression isentropique idéale (pointillée)
    s1 = states[0][0]
    p_vals = np.linspace(p_ev, p_cd, n_points)
    s_vals_ideal, T_vals_ideal = [], []

    for p in p_vals:
        try:
            T = PropsSI('T', 'P', p, 'S', s1, fluid)
            s_vals_ideal.append(s1 / 1000)
            T_vals_ideal.append(T - 273.15)
        except:
            continue

    plt.plot(s_vals_ideal, T_vals_ideal, 'k--', linewidth=1)

    # 2 -> 3
    h2 = PropsSI('H', 'S', states[1][0], 'T', states[1][1], fluid)
    h3 = PropsSI('H', 'S', states[2][0], 'T', states[2][1], fluid)
    h_vals = np.linspace(h2, h3, n_points)
    s_vals, T_vals = [], []
    for h in h_vals:
        try:
            T = PropsSI('T', 'P', p_cd, 'H', h, fluid)
            s = PropsSI('S', 'P', p_cd, 'H', h, fluid)
            s_vals.append(s/1000)
            T_vals.append(T - 273.15)
        except:
            continue
    plt.plot(s_vals, T_vals, 'r')

    # 3 -> 4
    p_vals = np.linspace(p_cd, p_ev, n_points)
    s_vals, T_vals = [], []
    h_3 = PropsSI('H', 'S', states[2][0], 'T', states[2][1], fluid)
    for p in p_vals:
        try:
            T = PropsSI('T', 'P', p, 'H', h_3, fluid)
            s = PropsSI('S', 'P', p, 'H', h_3, fluid)
            s_vals.append(s/1000)
            T_vals.append(T - 273.15)
        except:
            continue
    plt.plot(s_vals, T_vals, 'red')

    # 4 -> 1
    h4 = PropsSI('H', 'S', states[3][0], 'T', states[3][1], fluid)
    h1 = PropsSI('H', 'S', states[0][0], 'T', states[0][1], fluid)
    h_vals = np.linspace(h4, h1, n_points)
    s_vals, T_vals = [], []
    for h in h_vals:
        try:
            T = PropsSI('T', 'P', p_ev, 'H', h, fluid)
            s = PropsSI('S', 'P', p_ev, 'H', h, fluid)
            s_vals.append(s/1000)
            T_vals.append(T - 273.15)
        except:
            continue
    plt.plot(s_vals, T_vals, 'red')

    # Détente isentropique idéale (pointillée)
    s3 = states[2][0]
    p_vals = np.linspace(p_cd, p_ev, n_points)
    s_vals_ideal, T_vals_ideal = [], []

    for p in p_vals:
        try:
            T = PropsSI('T', 'P', p, 'S', s3, fluid)
            s_vals_ideal.append(s3 / 1000)
            T_vals_ideal.append(T - 273.15)
        except:
            continue
    
    #plt.plot(s_vals_ideal, T_vals_ideal, 'k--', linewidth=1)

    # === Points ===
    for i, (s, T) in enumerate(states[:-1]):
        plt.plot(s/1000, T - 273.15, 'ro')
        plt.text(s/1000, T - 273.15, f"{i+1}", fontsize=12, ha='right')
    

    plt.xlabel("Entropy [kJ/kg·K]")
    plt.ylabel("Temperature [°C]")
    plt.title("R410A with lift = {:.1f} °C".format(data_plot['lift']))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Move bottom and left spines away
    plt.gca().spines['bottom'].set_position(('outward', 10))
    plt.gca().spines['left'].set_position(('outward', 10))

    plt.tick_params(axis='both', which='major', labelsize=12)

    # Only show ticks of the points in states
    plt.yticks([T - 273.15 for s, T in states[:-1]])
    
    plt.ylim(min([T - 273.15 for s, T in states[:-1]]) - 5, max(max([T - 273.15 for s, T in states[:-1]]), T_range[-1]-273.15) + 5)
    plt.xlim(PropsSI('S', 'T', min([T for s, T in states[:-1]]) - 5, 'Q', 0, fluid)/1000, 
               max([s/1000 for s, T in states[:-1]])+ 0.01)
    
    plt.xticks([PropsSI('S', 'T', min([T for s, T in states[:-1]]) - 5, 'Q', 0, fluid)/1000, 
               max([s/1000 for s, T in states[:-1]])+0.01])

    
    plt.tight_layout()
    #plt.savefig('heat_pump_basic/heat_pump_' + str(int(data_plot['lift'])) + '.png', dpi=300)
    plt.show()

    return fig



    
#%%
#===TEST YOUR CODE=============================================================
#

# p_ev, T_1, p_cd, T_2, T_3, T_0, T_c, fluid
cycles = {
    '10°C' : {
        'inputs': (9.8e5, 10.7 + 273.15, 24.2e5, 65.2 + 273.15, 36.2 + 273.15, 6.6 + 273.15, 40.1 + 273.15, 'R410A'),
        'P_e' : 2.035e3,
        'delta_T_c' : 8.5-3,
        'delta_T_h' : 40.2-34.4, 
        'COP_r' : 4.546
    }, 
    '0°C' : {
        'inputs': (9.3e5, 9.4 + 273.15, 33e5, 90 + 273.15, 47.6 + 273.15, 4.8 + 273.15, 53.4 + 273.15, 'R410A'),
        'P_e' : 2.679e3,
        'delta_T_c' : 9.2 - 4.6,
        'delta_T_h' : 52.2-46.7,
        'COP_r' : 3.408
    },
}


inputs = {
    '10°C': (9.8e5, 10.7 + 273.15, 24.2e5, 65.2 + 273.15, 36.2 + 273.15, 6.6 + 273.15, 40.1 + 273.15, 'R410A'),
    '0°C': (9.3e5, 9.4 + 273.15, 33e5, 90 + 273.15, 47.6 + 273.15, 4.8 + 273.15, 53.4 + 273.15, 'R410A'),
    '-5°C': (9.e5, 8.4 + 273.15, 37.1e5, 100.7 + 273.15, 52.1 + 273.15, 4 + 273.15, 58.7 + 273.15, 'R410A')
}

other_inputs = {
    '10°C': (2.035e3, 8.5-3, 40.2-34.4, 4.546),
    '0°C': (2.679e3, 9.2 - 4.6, 52.2-46.7, 3.408),
    '-5°C': None
}

for label, input_data in inputs.items():
    print(f"Running heat pump simulation for T_ext = {label}")
    heat_pump(input_data, (True, True), other_inputs.get(label, None))
    print("\n")

print(9.e5)

