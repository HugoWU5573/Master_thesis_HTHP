import CoolProp
from state import State
from compressor import Compressor_3_params
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from seaborn import color_palette
from CoolProp.CoolProp import PropsSI
from math import floor, ceil

fluid = 'R290'
heos = CoolProp.AbstractState("HEOS&TTSE", fluid)
weights = (0.5, 0.5)
weights = None

LP_compressor = 1
HP_compressor = not LP_compressor

################################################################################################################
# Objective function for fitting
################################################################################################################

def fitting(params, inputs, data, weights = (0.5, 0.5)) :
    N, T_1, p_1, v_1, p_2, mdot = inputs
    BVR = params[0]
    eta_is_max = [params[1+i] for i in range(len(N))]
    eta_elme = [params[1+len(N)+i] for i in range(len(N))]
    P_meas, T_2_meas = data
    I, J, K = len(N), len(T_1[0]), len(T_1[0,0])
    sum_squared_errors_P = 0
    sum_squared_errors_T = 0
    T_2_meas_min, T_2_meas_max = np.nanmin(T_2_meas), np.nanmax(T_2_meas)

    compressors = []
    for i in range(len(N)) :
        compressor = Compressor_3_params(BVR = BVR, eta_v = None, eta_is_max = eta_is_max[i], eta_elme = eta_elme[i])
        compressors.append(compressor)
    
    for i in range(I) :
        for j in range(J) :
            for k in range(K) :
                if not np.isnan(T_2_meas[i,j,k]) :
                    compressors[i].eta_v = (mdot[i,j,k] * v_1[i,j,k]) / (V_s * N[i]/2)
                    #print("Fitting point: N =", N[i,j,k], "T_1 =", T_1[i,j,k], "p_1 =", p_1[i,j,k], "p_2 =", p_2[i,j,k], "mdot =", mdot[i,j,k])
                    state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                    P_calc, T_2_calc = compressors[i].Solve(state_1, p_2[i,j,k], mdot[i,j,k])
                    sum_squared_errors_P += ((P_calc - P_meas[i,j,k])/P_calc)**2
                    sum_squared_errors_T += ((T_2_calc - T_2_meas[i,j,k])/(T_2_meas_max - T_2_meas_min))**2
    
    objective = weights[0] * np.sqrt(sum_squared_errors_P) + weights[1] * np.sqrt(sum_squared_errors_T)
    return objective

def fitting_aggregation(params, inputs, data) :
    N, T_1, p_1, v_1, p_2, mdot = inputs
    BVR = params[0]
    eta_is_max = [params[1+i] for i in range(len(N))]
    P_meas, T_2_meas = data
    I, J, K = len(N), len(T_1[0]), len(T_1[0,0])
    sum_squared_errors_T = 0
    T_2_meas_min, T_2_meas_max = np.nanmin(T_2_meas), np.nanmax(T_2_meas)
    total_weight = 0
    
    compressors = []

    for i in range(len(N)) :
        compressor = Compressor_3_params(BVR = BVR, eta_v = 1, eta_is_max = eta_is_max[i], eta_elme = 1)
        compressors.append(compressor)
    
    for i in range(I) :
        for j in range(J) :
            for k in range(K) :
                if not np.isnan(T_2_meas[i,j,k]) :
                    state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                    heos.update(CoolProp.PT_INPUTS, p_1[i,j,k], T_1[i,j,k])
                    v1 = 1/heos.rhomass()
                    P_calc, T_2_calc = compressors[i].Solve(state_1, p_2[i,j,k], mdot[i,j,k])
                    heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_calc)
                    v2 = 1/heos.rhomass()
                    weight_data = 1
                    if LP_compressor and v1/v2 > 2.5 :
                        weight_data = 20
                    total_weight += weight_data
                    sum_squared_errors_T += weight_data * ((T_2_calc - T_2_meas[i,j,k])/(T_2_meas_max - T_2_meas_min))**2
    
    objective = np.sqrt(sum_squared_errors_T/total_weight)
    return objective

def fitting_mdot(params, T_cond, T_evap, mdot_meas) : 
    mdot_calc = np.zeros_like(mdot_meas)
    for i in range(len(T_cond)) :    
        for j in range(len(T_evap)) :
            mdot_calc[i,j] = params @ np.array([1, T_evap[j], T_cond[i], T_evap[j]**2, T_cond[i]**2, T_evap[j]*T_cond[i], 
                                                T_evap[j]**3, T_cond[i]**3, T_evap[j]**2 * T_cond[i], T_evap[j] * T_cond[i]**2])
            
    return np.sqrt(np.nansum(((mdot_calc - mdot_meas) / mdot_calc)**2))

def fitting_eta_v(inputs, data) :
    N, v_1, p_2 = inputs
    T_2_meas, mdot_meas = data

    v_2_meas = np.zeros_like(T_2_meas)
    eta_v_meas = np.zeros_like(T_2_meas)
    ratio_volume_eta_v = np.zeros_like(T_2_meas)
    for i in range(len(N)) :
        for j in range(len(T_2_meas[0])) :
            for k in range(len(T_2_meas[0,0])) :
                if not np.isnan(T_2_meas[i,j,k]) :
                    heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_meas[i,j,k])
                    v_2_meas[i,j,k] = 1/heos.rhomass()
                    eta_v_meas[i,j,k] = mdot_meas[i,j,k] * v_1[i,j,k] / (V_s * N[i]/2)
                    ratio_volume_eta_v[i,j,k] = v_1[i,j,k] / v_2_meas[i,j,k]
                else :
                    v_2_meas[i,j,k] = np.nan
                    eta_v_meas[i,j,k] = np.nan
                    ratio_volume_eta_v[i,j,k] = np.nan

    slope, offset = np.polyfit(ratio_volume_eta_v.flatten()[~np.isnan(ratio_volume_eta_v.flatten())], eta_v_meas.flatten()[~np.isnan(eta_v_meas.flatten())], 1)
    return slope, offset

################################################################################################################
# LP compressor characterisation
################################################################################################################

if LP_compressor :
    print("=================LP compressor characterisation=====================")

    bore = 41e-3
    stroke = 39.3e-3
    n_cylinders = 4 
    V_s = bore**2 * np.pi/4 * stroke * n_cylinders

    T_cond = np.array([60, 55, 50, 45, 40, 35, 30]) + 273.15
    T_evap = np.array([7,12,17,18]) + 273.15
    T_sup = 10

    p_cond = np.zeros_like(T_cond)
    p_evap = np.zeros_like(T_evap)

    for i in range(len(T_cond)):
        heos.update(CoolProp.QT_INPUTS, 1, T_cond[i])
        p_cond[i] = heos.p()
    for j in range(len(T_evap)):
        heos.update(CoolProp.QT_INPUTS, 0, T_evap[j])
        p_evap[j] = heos.p()

    T_cond_other = np.array([55, 50, 45, 40, 35, 30]) + 273.15
    T_evap_other = np.array([7]) + 273.15
    
    p_cond_other = np.zeros_like(T_cond_other)
    p_evap_other = np.zeros_like(T_evap_other)

    for i in range(len(T_cond_other)):
        heos.update(CoolProp.QT_INPUTS, 1, T_cond_other[i])
        p_cond_other[i] = heos.p()
    for j in range(len(T_evap_other)):
        heos.update(CoolProp.QT_INPUTS, 0, T_evap_other[j])
        p_evap_other[j] = heos.p()

    N = [25, 50] 

    P_25 = 1e3 * np.array([
        [2.1, 2.2, 2.3, 2.3],      # 60°C
        [2.0, 2.1, 2.2, 2.2],      # 55°C
        [1.9, 1.9, 2.0, 2.0],      # 50°C
        [1.7, 1.8, 1.8, 1.9],      # 45°C
        [1.6, 1.6, 1.7, 1.7],      # 40°C
        [1.5, 1.5, 1.5, 1.5],      # 35°C
        [1.3, 1.3, np.nan, np.nan] # 30°C
    ])
    P_50 = 1e3 * np.array([
        [4.6, 4.9, 5.2, 5.2],  # Tc = 60°C
        [4.4, 4.7, 4.9, 4.9],  # Tc = 55°C
        [4.2, 4.4, 4.6, 4.6],  # Tc = 50°C
        [4.0, 4.1, 4.2, 4.3],  # Tc = 45°C
        [3.7, 3.8, 3.9, 3.9],  # Tc = 40°C
        [3.4, 3.5, 3.5, 3.5],  # Tc = 35°C
        [3.1, 3.1, np.nan, np.nan]  # Tc = 30°C
    ])
    


    P = np.array([P_25, P_50])

    mdot_25 = 1/3600 * np.array([
        [83.3, 98.1, 115.1, 118.8],   # 60°C
        [85.6, 100.5, 117.7, 121.5],  # 55°C
        [87.8, 102.9, 120.1, 123.9],  # 50°C
        [89.9, 105.1, 122.5, 126.3],  # 45°C
        [91.9, 107.2, 124.7, 128.5],  # 40°C
        [93.7, 109.2, 126.8, 130.6],  # 35°C
        [95.5, 111.1, np.nan, np.nan] # 30°C
    ])
    mdot_50 = 1/3600 * np.array([
        [176.2, 207.5, 243.4, 251.4],  # Tc = 60°C
        [181.2, 212.7, 248.9, 256.9],  # Tc = 55°C
        [185.8, 217.6, 254.1, 262.2],  # Tc = 50°C
        [190.2, 222.3, 259.0, 267.2],  # Tc = 45°C
        [194.4, 226.8, 263.7, 271.9],  # Tc = 40°C
        [198.3, 231.0, 268.2, 276.4],  # Tc = 35°C
        [202.0, 235.0, np.nan, np.nan]  # Tc = 30°C
    ])
    mdot = np.array([mdot_25, mdot_50])

    T_2_25 = 273.15 + np.array([
        [84.6, 83.0, 81.9, 81.7],     # 60°C
        [79.0, 77.5, 76.5, 76.4],     # 55°C
        [73.5, 72.1, 71.3, 71.2],     # 50°C
        [68.0, 66.8, 66.1, 66.0],     # 45°C
        [62.7, 61.6, 61.0, 60.9],     # 40°C
        [57.3, 56.5, 55.9, 55.9],     # 35°C
        [52.1, 51.3, np.nan, np.nan]  # 30°C
    ])
    T_2_50 = 273.15 + np.array([
        [86.4, 85.0, 83.8, 83.7],  # Tc = 60°C
        [81.1, 79.7, 78.6, 78.4],  # Tc = 55°C
        [75.7, 74.4, 73.4, 73.3],  # Tc = 50°C
        [70.4, 69.2, 68.2, 68.1],  # Tc = 45°C
        [65.0, 63.9, 63.1, 62.9],  # Tc = 40°C
        [59.7, 58.7, 57.9, 57.8],  # Tc = 35°C
        [54.4, 53.5, np.nan, np.nan]  # Tc = 30°C
    ])
    T_2 = np.array([T_2_25, T_2_50])

    N_other = [30, 40, 60]
    P_other = np.zeros((len(N_other), len(T_cond), len(T_evap)))
    mdot_other = np.zeros((len(N_other), len(T_cond), len(T_evap)))
    T2_other = np.zeros((len(N_other), len(T_cond), len(T_evap)))

    for i, n in enumerate(N_other) :
        for j, Tc in enumerate(T_cond) :
            for k, To in enumerate(T_evap) :
                if Tc not in T_cond_other or To not in T_evap_other :
                    P_other[i,j,k] = np.nan
                    mdot_other[i,j,k] = np.nan
                    T2_other[i,j,k] = np.nan
                else :
                    data = np.loadtxt(f'code/fitting/compressor/data_LP/{int(To-273.15)}_{int(Tc-273.15)}_{n}.csv', delimiter=';', skiprows = 30, max_rows = 9, usecols = 3, dtype = str)
                    P_other[i,j,k] = float(data[0].replace(',', '.').strip()) * 1e3
                    mdot_other[i,j,k] = float(data[5].replace(',', '.')) / 3600
                    T2_other[i,j,k] = float(data[-1].replace(',', '.')) + 273.15
    
    '''
    N = np.concatenate((N, N_other))
    ratio_sorted_indices = np.argsort(N)
    N = N[ratio_sorted_indices]
    P = np.concatenate((P, P_other), axis = 0)[ratio_sorted_indices,:,:]
    mdot = np.concatenate((mdot, mdot_other), axis = 0)[ratio_sorted_indices,:,:]
    T_2 = np.concatenate((T_2, T2_other), axis = 0)[ratio_sorted_indices,:,:]
    '''
    T_1 = np.zeros_like(T_2)
    for i, T_e in enumerate(T_evap):
        T_1[:,:,i] = T_e + T_sup
    p_1 = np.zeros_like(T_2)
    for j, p_e in enumerate(p_evap):
        p_1[:,:,j] = p_e
    v_1 = np.zeros_like(T_2)
    for i in range(len(T_1)) :
        for j in range(len(T_1[0])) :
            for k in range(len(T_1[0,0])) :
                heos.update(CoolProp.PT_INPUTS, p_1[i,j,k], T_1[i,j,k])
                v_1[i,j,k] = 1/heos.rhomass()
    p_2 = np.zeros_like(T_2)
    for i, p_c in enumerate(p_cond):
        p_2[:,i,:] = p_c


    ###############################################################################
    # Mass flow rate fitting
    ###############################################################################

    slope, offset = fitting_eta_v((N, v_1, p_2), (T_2, mdot))
    print(f"Slope: {slope}, Offset: {offset}")

    mdot_calc = np.zeros_like(mdot)
    for i in range(len(N)) :
        params_mdot = np.array([163.7083028,5.443667331,0.177842814,0.064567073,0.00458696,-0.008491604,0.000286565,
                       -0.000115356,-0.000177236,-1.52749e-05]) * 1/3600
        params_mdot_opt = minimize(fitting_mdot, x0 = params_mdot, args = (T_cond, T_evap, mdot[i,:,:]), method = 'Nelder-Mead').x
        #print(f"Objective function value for N={N[i]} Hz:", fitting_mdot(params_mdot_opt, T_cond, T_evap, mdot[i,:,:]))

        for j in range(len(T_cond)) :
            for k in range(len(T_evap)) :
                mdot_calc[i,j,k] = params_mdot_opt @ np.array([1, T_evap[k], T_cond[j], T_evap[k]**2, T_cond[j]**2, T_evap[k]*T_cond[j], 
                                                T_evap[k]**3, T_cond[j]**3, T_evap[k]**2 * T_cond[j], T_evap[k] * T_cond[j]**2])
    
    ###############################################################################
    # Temperature fitting
    ###############################################################################

    inputs = (N, T_1, p_1, v_1, p_2, mdot_calc)
    data = (P, T_2)

    T_2_calc = np.zeros_like(T_2)
    v_2_calc = np.zeros_like(T_2)
    P_calc = np.zeros_like(T_2)
    eta_is_calc = np.zeros_like(T_2)
    eta_is_max = np.zeros(len(N))
    BVR = 0
    eta_v = np.zeros_like(T_2)
    eta_total = np.zeros_like(T_2)

    print("\n 1. Optimized Parameters")
    if weights is not None :

        BVR_initial = 3
        eta_is_max_initial = 0.8
        eta_elme_initial = 0.8
        params_initial = [BVR_initial] + [eta_is_max_initial] * len(N) + [eta_elme_initial] * len(N)
        params_opt = minimize(fitting, x0 = params_initial, args = (inputs, data, weights), method = 'Nelder-Mead').x #bounds=((2,2.5), (0.6,0.9), (None, None), (0.9, 1))).x
        print("BVR:", params_opt[0])
        for i in range(len(N)) :
            print(f"eta_is_max for N={N[i]} Hz:", params_opt[1+i])
            print(f"eta_elme for N={N[i]} Hz:", params_opt[1+len(N)+i])
        
        BVR = params_opt[0]
        eta_is_max = [params_opt[1+i] for i in range(len(N))]
        eta_elme = [params_opt[1+len(N)+i] for i in range(len(N))]

        for i in range(len(T_2)) :
            for j in range(len(T_2[0])) :
                for k in range(len(T_2[0,0])) :
                    if not np.isnan(T_2[i,j,k]) :
                        eta_v[i,j,k] = (mdot_calc[i,j,k] * v_1[i,j,k]) / (V_s * N[i]/2)
                        compressor = Compressor_3_params(params_opt[0], eta_v[i,j,k], eta_is_max=eta_is_max[i], eta_elme = eta_elme[i])
                        state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                        P_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_1, p_2[i,j,k], mdot_calc[i,j,k])
                        heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_calc[i,j,k])
                        v_2_calc[i,j,k] = 1/heos.rhomass()
                        h_2_calc = heos.hmass()
                        heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], state_1.s)
                        h_2is = heos.hmass()
                        eta_is_calc[i,j,k] = (h_2is - state_1.h) / (h_2_calc - state_1.h)
                        eta_total[i,j,k] = eta_v[i,j,k] * eta_elme[i]
                    else :
                        T_2_calc[i,j,k] = np.nan
                        v_2_calc[i,j,k] = np.nan
                        P_calc[i,j,k] = np.nan
                        eta_is_calc[i,j,k] = np.nan

    elif weights is None :

        BVR_initial = 3
        eta_is_max_initial = 0.8
        params_initial = [BVR_initial] + [eta_is_max_initial] * len(N)
        params_opt = minimize(fitting_aggregation, x0 = params_initial, args = (inputs, data), method = 'Nelder-Mead').x #bounds=((2,2.5), (0.6,0.9), (None, None), (0.9, 1))).x
        print("BVR:", params_opt[0])
        for i in range(len(N)) :
            print(f"eta_is_max for N={N[i]} Hz:", params_opt[1+i])
        
        BVR = params_opt[0]
        eta_is_max = [params_opt[1+i] for i in range(len(N))]

        for i in range(len(T_2)) :
            for j in range(len(T_2[0])) :
                for k in range(len(T_2[0,0])) :
                    if not np.isnan(T_2[i,j,k]) :
                        compressor = Compressor_3_params(params_opt[0], 1, eta_is_max=eta_is_max[i], eta_elme = 1)
                        state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                        P_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_1, p_2[i,j,k], mdot_calc[i,j,k])
                        heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_calc[i,j,k])
                        v_2_calc[i,j,k] = 1/heos.rhomass()
                        h_2_calc = heos.hmass()
                        heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], state_1.s)
                        h_2is = heos.hmass()
                        eta_is_calc[i,j,k] = (h_2is - state_1.h) / (h_2_calc - state_1.h)
                        eta_total[i,j,k] = P_calc[i,j,k] / P[i,j,k]
                        eta_v[i,j,k] = ( mdot[i,j,k] * v_1[i,j,k] ) / (V_s * N[i]/2)
                    else :
                        T_2_calc[i,j,k] = np.nan
                        v_2_calc[i,j,k] = np.nan
                        P_calc[i,j,k] = np.nan
                        eta_is_calc[i,j,k] = np.nan
                        eta_total[i,j,k] = np.nan
    
    eta_is = np.zeros_like(T_2)
    v_2 = np.zeros_like(T_2)
    for i in range(len(T_2)) :
        for j in range(len(T_2[0])) :
            for k in range(len(T_2[0,0])) :
                if not np.isnan(T_2[i,j,k]) :
                    heos.update(CoolProp.PT_INPUTS, p_1[i,j,k], T_1[i,j,k])
                    h_1 = heos.hmass()
                    s_1 = heos.smass()
                    
                    heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2[i,j,k])
                    h_2 = heos.hmass()
                    s_2 = heos.smass()
                    v_2[i,j,k] = 1/heos.rhomass()
                    
                    heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], s_1)
                    h_2is = heos.hmass()
                    
                    eta_is[i,j,k] = (h_2is - h_1) / (h_2 - h_1)
                else :
                    eta_is[i,j,k] = np.nan
                    v_2[i,j,k] = np.nan


                
    ###############################################################################
    # Total efficiency fitting
    ###############################################################################

    degree_of_polynomial = 3 
    eta_total_calc = np.zeros_like(eta_total)
    coeffs_eta_total = np.zeros((len(N), degree_of_polynomial+1))

    for i in range(len(N)) :
        volume_ratio = v_1[i,:,:].flatten() / v_2_calc[i,:,:].flatten()
        eta_total_meas = eta_total[i,:,:].flatten()
        valid_indices = ~np.isnan(eta_total_meas) & ~np.isnan(volume_ratio)
        coeffs = np.polyfit(volume_ratio[valid_indices], eta_total_meas[valid_indices], degree_of_polynomial)
        coeffs_eta_total[i,:] = coeffs
        eta_total_calc[i,:,:] = np.polyval(coeffs, v_1[i,:,:] / v_2_calc[i,:,:])
    
    if weights is None :
        
        for i in range(len(T_2)) :
            for j in range(len(T_2[0])) :
                for k in range(len(T_2[0,0])) :
                    if not np.isnan(T_2[i,j,k]) :
                        P_calc[i,j,k] = P_calc[i,j,k] / eta_total_calc[i,j,k]
                    else :
                        P_calc[i,j,k] = np.nan


    ###############################################################################
    # Plots
    ###############################################################################

    colors = color_palette("tab10", 10)

    # Calculate errors and their statistics

    error_P = 100 * (P_calc - P) / P
    error_T = 100 * (T_2_calc - T_2) / T_2
    error_mdot = 100 * (mdot_calc - mdot) / mdot
    mean_error_T = np.zeros(len(N))
    std_error_T = np.zeros(len(N))
    bins_T = []
    mean_error_P = np.zeros(len(N))
    std_error_P = np.zeros(len(N))
    bins_P = []
    mean_error_mdot = np.zeros(len(N))
    std_error_mdot = np.zeros(len(N))
    bins_mdot = []

    print("\n 2. Error statistics")
    for i in range(len(N)) :
        mean_error_T[i] = np.nanmean(error_T[i,:,:])
        std_error_T[i] = np.nanstd(error_T[i,:,:])
        print(f"N={N[i]} Hz: Mean Temperature Error = {mean_error_T[i]:.2f}%, Std Temperature Error = {std_error_T[i]:.2f}%")
        bins_T.append(np.arange(np.nanmin(error_T[i,:,:]), np.nanmax(error_T[i,:,:])*1.1, std_error_T[i]/2))

        mean_error_P[i] = np.nanmean(error_P[i,:,:])
        std_error_P[i] = np.nanstd(error_P[i,:,:])
        print(f"N={N[i]} Hz: Mean Power Error = {mean_error_P[i]:.2f}%, Std Power Error = {std_error_P[i]:.2f}%")
        bins_P.append(np.arange(np.nanmin(error_P[i,:,:]), np.nanmax(error_P[i,:,:])*1.1, std_error_P[i]/2))
        
        mean_error_mdot[i] = np.nanmean(error_mdot[i,:,:])
        std_error_mdot[i] = np.nanstd(error_mdot[i,:,:])
        print(f"N={N[i]} Hz: Mean Mass Flow Rate Error = {mean_error_mdot[i]:.2f}%, Std Mass Flow Rate Error = {std_error_mdot[i]:.2f}%")
        bins_mdot.append(np.arange(np.nanmin(error_mdot[i,:,:]), np.nanmax(error_mdot[i,:,:])*1.1, std_error_mdot[i]/2))

    # Temperature error 
    '''
    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.hist(error_T[i,:,:].flatten(),
                bins=bins_T[i], alpha=0.7/(i+1), color = colors[2*i],
                label=f'N={N[i]} Hz', edgecolor='black')

    plt.xlabel('Temperature Error [%]', fontsize=15)
    plt.ylabel('Occurrence', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), T_2[i,:,:].flatten()-273.15, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
        plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), T_2_calc[i,:,:].flatten()-273.15, 'x', color = colors[2*i], label=f'Fitted data N={N[i]} Hz')

    plt.xlabel(r'$v_1 / v_2$', fontsize=15)
    plt.ylabel(r'$T_2$ [°C]', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    
    plt.show()

    # Power error

    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.hist(error_P[i,:,:].flatten(),
                bins=bins_P[i], alpha=0.7/(i+1),
                label=f'N={N[i]} Hz', edgecolor='black', color = colors[2*i])

    plt.xlabel('Power Error [%]', fontsize=15)
    plt.ylabel('Occurrence', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), P[i,:,:].flatten()/1000, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
        plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), P_calc[i,:,:].flatten()/1000, 'x', color = colors[2*i], label=f'Fitted data N={N[i]} Hz')
    plt.xlabel(r'$v_1 / v_2$', fontsize=15)
    plt.ylabel(r'$P$ [kW]', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()

    # Mass flow rate error

    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.hist(error_mdot[i,:,:].flatten(),
                bins=bins_mdot[i], alpha=0.7/(i+1), color = colors[2*i],
                label=f'N={N[i]} Hz', edgecolor='black')
    plt.xlabel('Mass Flow Rate Error [%]', fontsize=15)
    plt.ylabel('Occurrence', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), mdot[i,:,:].flatten()*3600, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), mdot_calc[i,:,:].flatten()*3600, 'x', color = colors[2*i], label=f'Fitted data N={N[i]} Hz')
    plt.xlabel(r'$v_1 / v_2$', fontsize=15)
    plt.ylabel(r'$\dot{m}$ [kg/h]', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()
    '''
    # Isentropic efficiency

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_is[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
        ratio_sorted_indices = np.argsort(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten())
        plt.plot(v_1[i,:,:].flatten()[ratio_sorted_indices]/v_2_calc[i,:,:].flatten()[ratio_sorted_indices], eta_is_calc[i,:,:].flatten()[ratio_sorted_indices], '-', color = colors[2*i], label=f'Fitted data N={N[i]} Hz')
    plt.xlabel(r'$v_1 / v_2$', fontsize=12)
    plt.legend(fontsize=12)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'$\eta_{is}$', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Disable automatic tick locator
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

    ax.set_yticks([floor(np.nanmin(eta_is.flatten())*100)/100, ceil(max(eta_is.flatten())*100)/100, np.round(eta_is_max[0], 2), np.round(eta_is_max[1], 2)])
    plt.ylim(floor(np.nanmin(eta_is.flatten())*100)/100, ceil(max(eta_is.flatten())*100)/100)
    plt.xlim(1.4, 3.4)
    ax.set_xticks([1.4, 1.8, 2.2, 2.6, 3.0, 3.4])

    # Total efficiency
    plt.figure(figsize=(8,6))
    range_of_ratios = np.linspace(np.nanmin(v_1/v_2), np.nanmax(v_1/v_2), 100)
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_total[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz', clip_on = False)
        plt.plot(range_of_ratios, np.polyval(coeffs_eta_total[i,:], range_of_ratios), '-', color = colors[2*i], label=f'Fitted data N={N[i]} Hz', clip_on = False)
    plt.xlabel(r'$v_1 / v_2$', fontsize=12)
    plt.legend(fontsize=12, frameon=False)

    #plt.tight_layout()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'$\eta_{tot}$', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Disable automatic tick locator
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

    ax.set_yticks([floor(np.nanmin(eta_total.flatten())*100)/100, ceil(max(eta_total.flatten())*100)/100])
    plt.ylim(floor(np.nanmin(eta_total.flatten())*100)/100, ceil(max(eta_total.flatten())*100)/100)
    plt.xlim(1.4, 3.4)
    ax.set_xticks([1.4, 1.8, 2.2, 2.6, 3.0, 3.4])
    plt.show()

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_v[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
    plt.xlabel(r'$v_1 / v_2$', fontsize=12)
    plt.legend(fontsize=12)
    plt.title(r'$\eta_v$', loc='left', fontsize=12)
    #plt.xlim(1.4, 3.4)
    ax = plt.gca()
    #ax.set_xticks([1.4, 1.8, 2.2, 2.6, 3.0, 3.4])
    plt.show()

if HP_compressor :
    ##################################################################################################################
    # HP compressor characterisation
    ##################################################################################################################
    print("=================HP compressor characterisation=====================")

    bore = 41e-3
    stroke = 27.3e-3
    n_cylinders = 4 
    V_s = bore**2 * np.pi/4 * stroke * n_cylinders

    N = np.array([50])

    p_cond = np.array([50, 45, 40]) * 1e5
    T_evap = np.array([25, 30, 35, 40, 45, 50]) + 273.15
    T_sup = 10
    p_evap = np.zeros_like(T_evap)
    for j in range(len(T_evap)):
        heos.update(CoolProp.QT_INPUTS, 0, T_evap[j])
        p_evap[j] = heos.p()

    mdot = 1/3600 * np.array([
        [148, 179, 214, 253, 297, 346],     # 50 bar
        [158, 189, 225, 264, 309, 358],     # 45 bar
        [168, 200, 235, 126, 321, 371]      # 40 bar
    ])
    mdot[0, 4] = np.nan  # Removing data point with abnormal value
    mdot[1,2] = np.nan  # Removing data point with abnormal value
    mdot[2,3] = np.nan  # Removing data point with abnormal value
    mdot = np.array([mdot])

    P = 1e3 * np.array([
        [5.9, 6.2, 6.4, 6.7, 6.9, 7.2],     # 50 bar
        [5.5, 5.8, 6, 6.2, 6.4, 6.6],       # 45 bar
        [5.2, 5.4, 5.6, 5.8, 5.9, 6]      # 40 bar
    ])
    P[0,4] = np.nan  # Removing data point with abnormal value
    P[1,2] = np.nan  # Removing data point with abnormal value
    P[2,3] = np.nan  # Removing data point with abnormal value
    P = np.array([P])

    T_2 = 273.15 + np.array([
        [142, 138, 135, 132, 136, 129],     # 50 bar
        [132, 129, 129, 124, 122, 121],     # 45 bar
        [123, 120, 117, 116, 114, 113]      # 40 bar
    ])
    T_2[0,4] = np.nan  # Removing data point with abnormal value
    T_2[1,2] = np.nan  # Removing data point with abnormal value
    T_2[2,3] = np.nan  # Removing data point with abnormal value
    T_2 = np.array([T_2])

    T_1 = np.zeros_like(T_2)
    for i, T_e in enumerate(T_evap):
        T_1[:,:,i] = T_e + T_sup
        
    p_1 = np.zeros_like(T_2)
    for j, p_e in enumerate(p_evap):
        p_1[:,:,j] = p_e
    v_1 = np.zeros_like(T_2)
    for i in range(len(T_1)) :
        for j in range(len(T_1[0])) :
            for k in range(len(T_1[0,0])) :
                heos.update(CoolProp.PT_INPUTS, p_1[i,j,k], T_1[i,j,k])
                v_1[i,j,k] = 1/heos.rhomass()
    p_2 = np.zeros_like(T_2)
    for i, p_c in enumerate(p_cond):
        p_2[:,i,:] = p_c

    ###############################################################################
    # Mass flow rate fitting
    ###############################################################################

    '''
    mdot_calc = np.zeros_like(mdot)
    for i in range(len(N)) :
        params_mdot = np.array([163.7083028,5.443667331,0.177842814,0.064567073,0.00458696,-0.008491604,0.000286565,
                       -0.000115356,-0.000177236,-1.52749e-05]) * 1/3600
        params_mdot_opt = minimize(fitting_mdot, x0 = params_mdot, args = (T_cond, T_evap, mdot[i,:,:]), method = 'Nelder-Mead').x
        #print(f"Objective function value for N={N[i]} Hz:", fitting_mdot(params_mdot_opt, T_cond, T_evap, mdot[i,:,:]))

        for j in range(len(T_cond)) :
            for k in range(len(T_evap)) :
                mdot_calc[i,j,k] = params_mdot_opt @ np.array([1, T_evap[k], T_cond[j], T_evap[k]**2, T_cond[j]**2, T_evap[k]*T_cond[j], 
                                                T_evap[k]**3, T_cond[j]**3, T_evap[k]**2 * T_cond[j], T_evap[k] * T_cond[j]**2])
    '''
    mdot_calc = mdot.copy()  # Using measured mass flow rates directly for temperature fitting, as the dataset is small and the fitting of mass flow rate is not the main focus here
    ###############################################################################
    # Temperature fitting
    ###############################################################################

    inputs = (N, T_1, p_1, v_1, p_2, mdot_calc)
    data = (P, T_2)

    T_2_calc = np.zeros_like(T_2)
    v_2_calc = np.zeros_like(T_2)
    P_calc = np.zeros_like(T_2)
    eta_is_calc = np.zeros_like(T_2)
    eta_is_max = np.zeros(len(N))
    BVR = 0
    eta_v = np.zeros_like(T_2)
    eta_total = np.zeros_like(T_2)

    print("\n 1. Optimized Parameters")
    if weights is not None :

        BVR_initial = 3
        eta_is_max_initial = 0.8
        eta_elme_initial = 0.8
        params_initial = [BVR_initial] + [eta_is_max_initial] * len(N) + [eta_elme_initial] * len(N)
        params_opt = minimize(fitting, x0 = params_initial, args = (inputs, data, weights), method = 'Nelder-Mead').x #bounds=((2,2.5), (0.6,0.9), (None, None), (0.9, 1))).x
        print("BVR:", params_opt[0])
        for i in range(len(N)) :
            print(f"eta_is_max for N={N[i]} Hz:", params_opt[1+i])
            print(f"eta_elme for N={N[i]} Hz:", params_opt[1+len(N)+i])
        
        BVR = params_opt[0]
        eta_is_max = [params_opt[1+i] for i in range(len(N))]
        eta_elme = [params_opt[1+len(N)+i] for i in range(len(N))]

        for i in range(len(T_2)) :
            for j in range(len(T_2[0])) :
                for k in range(len(T_2[0,0])) :
                    if not np.isnan(T_2[i,j,k]) :
                        eta_v[i,j,k] = (mdot_calc[i,j,k] * v_1[i,j,k]) / (V_s * N[i]/2)
                        compressor = Compressor_3_params(params_opt[0], eta_v[i,j,k], eta_is_max=eta_is_max[i], eta_elme = eta_elme[i])
                        state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                        P_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_1, p_2[i,j,k], mdot_calc[i,j,k])
                        heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_calc[i,j,k])
                        v_2_calc[i,j,k] = 1/heos.rhomass()
                        h_2_calc = heos.hmass()
                        heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], state_1.s)
                        h_2is = heos.hmass()
                        eta_is_calc[i,j,k] = (h_2is - state_1.h) / (h_2_calc - state_1.h)
                        eta_total[i,j,k] = eta_v[i,j,k] * eta_elme[i]
                    else :
                        T_2_calc[i,j,k] = np.nan
                        v_2_calc[i,j,k] = np.nan
                        P_calc[i,j,k] = np.nan
                        eta_is_calc[i,j,k] = np.nan
                        eta_total[i,j,k] = np.nan

    elif weights is None :

        BVR_initial = 3
        eta_is_max_initial = 0.8
        params_initial = [BVR_initial] + [eta_is_max_initial] * len(N)
        params_opt = minimize(fitting_aggregation, x0 = params_initial, args = (inputs, data), method = 'Nelder-Mead').x #bounds=((2,2.5), (0.6,0.9), (None, None), (0.9, 1))).x
        print("BVR:", params_opt[0])
        for i in range(len(N)) :
            print(f"eta_is_max for N={N[i]} Hz:", params_opt[1+i])
        
        BVR = params_opt[0]
        eta_is_max = [params_opt[1+i] for i in range(len(N))]

        for i in range(len(T_2)) :
            for j in range(len(T_2[0])) :
                for k in range(len(T_2[0,0])) :
                    if not np.isnan(T_2[i,j,k]) :
                        compressor = Compressor_3_params(params_opt[0], 1, eta_is_max=eta_is_max[i], eta_elme = 1)
                        state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                        P_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_1, p_2[i,j,k], mdot_calc[i,j,k])
                        heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_calc[i,j,k])
                        v_2_calc[i,j,k] = 1/heos.rhomass()
                        h_2_calc = heos.hmass()
                        heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], state_1.s)
                        h_2is = heos.hmass()
                        eta_is_calc[i,j,k] = (h_2is - state_1.h) / (h_2_calc - state_1.h)
                        eta_total[i,j,k] = P_calc[i,j,k] / P[i,j,k]
                        eta_v[i,j,k] = ( mdot_calc[i,j,k] * v_1[i,j,k] ) / (V_s * N[i]/2)
                    else :
                        T_2_calc[i,j,k] = np.nan
                        v_2_calc[i,j,k] = np.nan
                        P_calc[i,j,k] = np.nan
                        eta_is_calc[i,j,k] = np.nan
                        eta_total[i,j,k] = np.nan

    eta_is = np.zeros_like(T_2)
    v_2 = np.zeros_like(T_2)
    for i in range(len(T_2)) :
        for j in range(len(T_2[0])) :
            for k in range(len(T_2[0,0])) :
                if not np.isnan(T_2[i,j,k]) :
                    heos.update(CoolProp.PT_INPUTS, p_1[i,j,k], T_1[i,j,k])
                    h_1 = heos.hmass()
                    s_1 = heos.smass()
                    v_1[i,j,k] = 1/heos.rhomass()
                    
                    heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2[i,j,k])
                    h_2 = heos.hmass()
                    s_2 = heos.smass()
                    v_2[i,j,k] = 1/heos.rhomass()
                    
                    heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], s_1)
                    h_2is = heos.hmass()
                    
                    eta_is[i,j,k] = (h_2is - h_1) / (h_2 - h_1)

                else :
                    eta_is[i,j,k] = np.nan
                    v_2[i,j,k] = np.nan


                
    ###############################################################################
    # Total efficiency fitting
    ###############################################################################

    degree_of_polynomial = 3 
    eta_total_calc = np.zeros_like(eta_total)
    coeffs_eta_total = np.zeros((len(N), degree_of_polynomial+1))

    for i in range(len(N)) :
        volume_ratio = v_1[i,:,:].flatten() / v_2_calc[i,:,:].flatten()
        eta_total_meas = eta_total[i,:,:].flatten()
        valid_indices = ~np.isnan(eta_total_meas) & ~np.isnan(volume_ratio)
        coeffs = np.polyfit(volume_ratio[valid_indices], eta_total_meas[valid_indices], degree_of_polynomial)
        coeffs_eta_total[i,:] = coeffs
        eta_total_calc[i,:,:] = np.polyval(coeffs, v_1[i,:,:] / v_2_calc[i,:,:])
    
    if weights is None :
        
        for i in range(len(T_2)) :
            for j in range(len(T_2[0])) :
                for k in range(len(T_2[0,0])) :
                    if not np.isnan(T_2[i,j,k]) :
                        P_calc[i,j,k] = P_calc[i,j,k] / eta_total_calc[i,j,k]
                    else :
                        P_calc[i,j,k] = np.nan


    ###############################################################################
    # Plots
    ###############################################################################

    colors = color_palette("tab10", 4)

    # Calculate errors and their statistics

    error_P = 100 * (P_calc - P) / P
    error_T = 100 * (T_2_calc - T_2) / T_2
    error_mdot = 100 * (mdot_calc - mdot) / mdot
    mean_error_T = np.zeros(len(N))
    std_error_T = np.zeros(len(N))
    bins_T = []
    mean_error_P = np.zeros(len(N))
    std_error_P = np.zeros(len(N))
    bins_P = []

    '''
    mean_error_mdot = np.zeros(len(N))
    std_error_mdot = np.zeros(len(N))
    bins_mdot = []
    '''

    print("\n 2. Error statistics")
    for i in range(len(N)) :
        mean_error_T[i] = np.nanmean(error_T[i,:,:])
        std_error_T[i] = np.nanstd(error_T[i,:,:])
        print(f"N={N[i]} Hz: Mean Temperature Error = {mean_error_T[i]:.2f}%, Std Temperature Error = {std_error_T[i]:.2f}%")
        bins_T.append(np.arange(np.nanmin(error_T[i,:,:]), np.nanmax(error_T[i,:,:])*1.1, std_error_T[i]/2))

        mean_error_P[i] = np.nanmean(error_P[i,:,:])
        std_error_P[i] = np.nanstd(error_P[i,:,:])
        print(f"N={N[i]} Hz: Mean Power Error = {mean_error_P[i]:.2f}%, Std Power Error = {std_error_P[i]:.2f}%")
        bins_P.append(np.arange(np.nanmin(error_P[i,:,:]), np.nanmax(error_P[i,:,:])*1.1, std_error_P[i]/2))
        
        '''
        mean_error_mdot[i] = np.nanmean(error_mdot[i,:,:])
        std_error_mdot[i] = np.nanstd(error_mdot[i,:,:])
        print(f"N={N[i]} Hz: Mean Mass Flow Rate Error = {mean_error_mdot[i]:.2f}%, Std Mass Flow Rate Error = {std_error_mdot[i]:.2f}%")
        bins_mdot.append(np.arange(np.nanmin(error_mdot[i,:,:]), np.nanmax(error_mdot[i,:,:])*1.1, std_error_mdot[i]/2))
        '''
    # Temperature error 
    '''
    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.hist(error_T[i,:,:].flatten(),
                bins=bins_T[i], alpha=0.7/(i+1), color = colors[2*i],
                label=f'N={N[i]} Hz', edgecolor='black')

    plt.xlabel('Temperature Error [%]', fontsize=15)
    plt.ylabel('Occurrence', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), T_2[i,:,:].flatten()-273.15, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
        plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), T_2_calc[i,:,:].flatten()-273.15, 'x', color = colors[2*i], label=f'Fitted data N={N[i]} Hz')

    plt.xlabel(r'$v_1 / v_2$', fontsize=15)
    plt.ylabel(r'$T_2$ [°C]', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    
    plt.show()

    # Power error

    plt.figure(figsize=(8,6))

    for i in range(len(N)) :
        plt.hist(error_P[i,:,:].flatten(),
                bins=bins_P[i], alpha=0.7/(i+1),
                label=f'N={N[i]} Hz', edgecolor='black', color = colors[2*i])

    plt.xlabel('Power Error [%]', fontsize=15)
    plt.ylabel('Occurrence', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), P[i,:,:].flatten()/1000, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
        plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), P_calc[i,:,:].flatten()/1000, 'x', color = colors[2*i], label=f'Fitted data N={N[i]} Hz')
    plt.xlabel(r'$v_1 / v_2$', fontsize=15)
    plt.ylabel(r'$P$ [kW]', fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()
    '''
    # Isentropic efficiency

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_is[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz', clip_on = False)
        ratio_sorted_indices = np.argsort(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten())
        plt.plot(v_1[i,:,:].flatten()[ratio_sorted_indices]/v_2_calc[i,:,:].flatten()[ratio_sorted_indices], eta_is_calc[i,:,:].flatten()[ratio_sorted_indices], '-', color = colors[2*i], label=f'Fitted data N={N[i]} Hz', clip_on = False)
    plt.xlabel(r'$v_1 / v_2$', fontsize=12)
    plt.legend(fontsize=12)
   
   # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'$\eta_{is}$', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Disable automatic tick locator
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

    ax.set_yticks([floor(np.nanmin(eta_is.flatten())*100)/100, ceil(max(eta_is_calc.flatten())*100)/100, np.round(eta_is_max[0], 2)])
    plt.ylim(floor(np.nanmin(eta_is.flatten())*100)/100, ceil(max(eta_is_calc.flatten())*100)/100)
    plt.xlim(2.4, 5.2)
    ax.set_xticks([2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2])

    # Total efficiency
    plt.figure(figsize=(8,6))
    range_of_ratios = np.linspace(np.nanmin(v_1/v_2), np.nanmax(v_1/v_2), 100)
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_total[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz', clip_on = False)
        plt.plot(range_of_ratios, np.polyval(coeffs_eta_total[i,:], range_of_ratios), '-', color = colors[2*i], label=f'Fitted data N={N[i]} Hz', clip_on = False)
    plt.xlabel(r'$v_1 / v_2$', fontsize=12)
    plt.legend(fontsize=14)
  # Add some text for labels, title and custom x-axis tick labels, etc.
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'$\eta_{tot}$', loc='left', fontsize=12)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and left spines away
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Disable automatic tick locator
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

    ax.set_yticks([floor(np.nanmin(eta_total.flatten())*100)/100, ceil(max(eta_total.flatten())*100)/100])
    plt.ylim(floor(np.nanmin(eta_total.flatten())*100)/100, ceil(max(eta_total.flatten())*100)/100)
    plt.xlim(2.4, 5.2)
    ax.set_xticks([2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2])
    plt.show()

    plt.figure(figsize=(8,6))
    for i in range(len(N)) :
        plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_v[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
    plt.xlabel(r'$v_1 / v_2$', fontsize=12)
    plt.legend(fontsize=12)
    plt.title(r'$\eta_v$', loc='left', fontsize=12)
    plt.xlim(2.4, 5.2)
    ax = plt.gca()
    ax.set_xticks([2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2])
    plt.show()






