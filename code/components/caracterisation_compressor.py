import CoolProp
from state import State
from compressor import Compressor_3_params
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from seaborn import color_palette
from CoolProp.CoolProp import PropsSI

fluid = 'R290'
heos = CoolProp.AbstractState("HEOS&TTSE", fluid)

################################################################################################################
# LP compressor characterisation
################################################################################################################
print("=================LP compressor characterisation=====================")

T_cond = np.array([60, 55, 50, 45, 40, 35, 30]) + 273.15
T_evap = np.array([7,12,17,18]) + 273.15
T_sup = 10
bore = 41e-3
stroke = 39.3e-3
n_cylinders = 4 
V_s = bore**2 * np.pi/4 * stroke * n_cylinders

p_cond = np.zeros_like(T_cond)
p_evap = np.zeros_like(T_evap)

for i in range(len(T_cond)):
    heos.update(CoolProp.QT_INPUTS, 1, T_cond[i])
    p_cond[i] = heos.p()
for j in range(len(T_evap)):
    heos.update(CoolProp.QT_INPUTS, 0, T_evap[j])
    p_evap[j] = heos.p()

N = [25, 50] 

Q_I_25 = 1e3 * np.array([
    [5.4, 6.5, 7.8, 8.1],      # 60°C
    [5.9, 7.1, 8.5, 8.8],      # 55°C
    [6.4, 7.7, 9.2, 9.5],      # 50°C
    [7.0, 8.3, 9.9, 10.2],     # 45°C
    [7.5, 8.9, 10.6, 10.9],    # 40°C
    [8.0, 9.5, 11.2, 11.6],    # 35°C
    [8.5, 10.1, np.nan, np.nan] # 30°C (données manquantes)
])
Q_I_50 = 1e3 * np.array([
    [11.4, 13.7, 16.5, 17.1],  # Tc = 60°C
    [12.5, 15.0, 18.0, 18.6],  # Tc = 55°C
    [13.6, 16.3, 19.4, 20.1],  # Tc = 50°C
    [14.7, 17.6, 20.9, 21.6],  # Tc = 45°C
    [15.8, 18.8, 22.3, 23.1],  # Tc = 40°C
    [16.9, 20.1, 23.8, 24.6],  # Tc = 35°C
    [18.0, 21.4, np.nan, np.nan]  # Tc = 30°C (valeurs manquantes dans le PDF)
])
Q = np.array([Q_I_25, Q_I_50])

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

inputs = (N, T_1, p_1, v_1, p_2, mdot)
data = (P, T_2)

#print(np.nanmax(0.8 * mdot * v_1 / N))
compressors = []
for i in range(len(N)) :
    compressor = Compressor_3_params(BVR = None, eta_v = None, eta_is_max = None, eta_elme = None)
    compressors.append(compressor)

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

    for i in range(len(N)) :
            compressors[i].BVR = BVR
            compressors[i].eta_is_max = eta_is_max[i]
            compressors[i].eta_elme = eta_elme[i]
    
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

BVR_initial = 3
eta_is_max_initial = 0.8
eta_elme_initial = 0.8
params_initial = [BVR_initial] + [eta_is_max_initial] * len(N) + [eta_elme_initial] * len(N)
params_opt = minimize(fitting, x0 = params_initial, args = (inputs, data), method = 'Nelder-Mead').x #bounds=((2,2.5), (0.6,0.9), (None, None), (0.9, 1))).x
print("Objective function value:", fitting(params_opt, inputs, data))
print("Optimized parameters:")
print("BVR:", params_opt[0])
for i in range(len(N)) :
    print(f"eta_is_max for N={N[i]} Hz:", params_opt[1+i])
    print(f"eta_elme for N={N[i]} Hz:", params_opt[1+len(N)+i])
    
eta_is_max = [params_opt[1+i] for i in range(len(N))]
eta_elme = [params_opt[1+len(N)+i] for i in range(len(N))]
eta_v = np.zeros_like(T_2)
T_2_calc = np.zeros_like(T_2)
v_2_calc = np.zeros_like(T_2)
P_calc = np.zeros_like(T_2)
eta_is_calc = np.zeros_like(T_2)

for i in range(len(T_2)) :
    for j in range(len(T_2[0])) :
        for k in range(len(T_2[0,0])) :
            if not np.isnan(T_2[i,j,k]) :
                eta_v[i,j,k] = (mdot[i,j,k] * v_1[i,j,k]) / (V_s * N[i]/2)
                compressor = Compressor_3_params(params_opt[0], eta_v[i,j,k], eta_is_max=eta_is_max[i], eta_elme = eta_elme[i])
                state_1 = State(heos, T = T_1[i,j,k], p = p_1[i,j,k])
                P_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_1, p_2[i,j,k], mdot[i,j,k])
                heos.update(CoolProp.PT_INPUTS, p_2[i,j,k], T_2_calc[i,j,k])
                v_2_calc[i,j,k] = 1/heos.rhomass()
                h_2_calc = heos.hmass()
                heos.update(CoolProp.PSmass_INPUTS, p_2[i,j,k], state_1.s)
                h_2is = heos.hmass()
                eta_is_calc[i,j,k] = (h_2is - state_1.h) / (h_2_calc - state_1.h)
            else :
                T_2_calc[i,j,k] = np.nan
                v_2_calc[i,j,k] = np.nan
                P_calc[i,j,k] = np.nan
                eta_is_calc[i,j,k] = np.nan
               

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

colors = color_palette("tab10", 4)

error_P = 100 * np.abs(P_calc - P) / P
error_T = 100 * np.abs(T_2_calc - T_2) / T_2
plt.figure(figsize=(8,6))
for i in range(len(N)) :
    plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), error_P[i,:,:].flatten(), 'o', color = colors[2*i], label = f'Power N={N[i]} Hz')
    plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), error_T[i,:,:].flatten(), 'x', color = colors[2*i+1], label = f'Temperature N={N[i]} Hz')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'Relative error [%]')
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(8,6))
for i in range(len(N)) :
    plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_is[i,:,:].flatten(), 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
    plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), eta_is_calc[i,:,:].flatten(), 'x', color = colors[2*i+1], label=f'Fitted data N={N[i]} Hz')

'''
plt.hlines(params_opt[1],
            xmin=np.nanmin(v_1[1,:,:].flatten()/v_2[1,:,:].flatten()),
            xmax=np.nanmax(v_1[1,:,:].flatten()/v_2[1,:,:].flatten()),
            colors='black', linestyles='dashed',
            label=r'Fitted $\eta_{is,max}$' + f' = {params_opt[1]:.2f}')
plt.vlines(params_opt[0], ymin=np.nanmin(eta_is[1,:,:].flatten()), ymax=np.nanmax(eta_is[0,:,:].flatten()), colors='black', 
           linestyles='dotted', label=r'Fitted BVR' + f' = {params_opt[0]:.2f}')
'''
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$\eta_{is}$')
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(8,6))
for i in range(len(N)) :
    plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), P[i,:,:].flatten()/1000, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
    plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), P_calc[i,:,:].flatten()/1000, 'x', color = colors[2*i+1], label=f'Fitted data N={N[i]} Hz')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$P$ [kW]')
plt.legend()
plt.tight_layout()
#plt.show()


plt.figure(figsize=(8,6))

for i in range(len(N)) :
    plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), T_2[i,:,:].flatten()-273.15, 'o', color = colors[2*i], label=f'Measured data N={N[i]} Hz')
    plt.plot(v_1[i,:,:].flatten()/v_2_calc[i,:,:].flatten(), T_2_calc[i,:,:].flatten()-273.15, 'x', color = colors[2*i+1], label=f'Fitted data N={N[i]} Hz')

plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$T_2$ [°C]')
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(8,6))
for i in range(len(N)) :
    plt.plot(v_1[i,:,:].flatten()/v_2[i,:,:].flatten(), eta_v[i,:,:].flatten() * eta_elme[i], 'o', color = colors[2*i], label=f'N={N[i]} Hz')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$\eta_{v} \cdot \eta_{elme}$')
plt.legend()
plt.tight_layout()
plt.show()
#plt.close('all')

'''
##################################################################################################################
# HP compressor characterisation
##################################################################################################################
print("=================HP compressor characterisation=====================")

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

P = 1e3 * np.array([
    [5.9, 6.2, 6.4, 6.7, 6.9, 7.2],     # 50 bar
    [5.5, 5.8, 6, 6.2, 6.4, 6.6],       # 45 bar
    [5.2, 5.4, 5.6, 5.8, 5.9, 6]      # 40 bar
])
P[0,4] = np.nan  # Removing data point with abnormal value
P[1,2] = np.nan  # Removing data point with abnormal value

T_2 = 273.15 + np.array([
    [142, 138, 135, 132, 136, 129],     # 50 bar
    [132, 129, 129, 124, 122, 121],     # 45 bar
    [123, 120, 117, 116, 114, 113]      # 40 bar
])
T_2[0,4] = np.nan  # Removing data point with abnormal value
T_2[1,2] = np.nan  # Removing data point with abnormal value

N = np.ones_like(T_2) * 50
T_1 = np.zeros_like(T_2)
for i, T_e in enumerate(T_evap):
    T_1[:,i] = T_e + T_sup
p_1 = np.zeros_like(T_2)
for j, p_e in enumerate(p_evap):
    p_1[:,j] = p_e
v_1 = np.zeros_like(T_2)
for i in range(len(T_1)) :
    for j in range(len(T_1[0])) :
        heos.update(CoolProp.PT_INPUTS, p_1[i,j], T_1[i,j])
        v_1[i,j] = 1/heos.rhomass()
p_2 = np.zeros_like(T_2)
for i, p_c in enumerate(p_cond):
    p_2[i,:] = p_c
inputs = (np.array([N]), np.array([T_1]), np.array([p_1]), np.array([v_1]), np.array([p_2]), np.array([mdot]))
data = (np.array([P]), np.array([T_2]))

params_initial = [2.25, 0.7, 40e-6]  # Initial guesses for BVR, eta_is_max, V_s, eta_elme
params_opt = minimize(fitting, x0 = params_initial, args = (inputs, data, (0.5, 0.5)), method = 'Nelder-Mead').x#, bounds=((1,3), (0.65,0.75), (None, None))).x
#print(params_opt)
print("Objective function value:", fitting(params_opt, inputs, data, (0.3, 0.7)))
print("Optimized parameters:")
print("BVR:", params_opt[0])
print("eta_is_max:", params_opt[1])
print("V_s:", params_opt[2])
print("eta_elme:", eta_elme)

eta_v = params_opt[2] * N / (mdot * v_1)
specific_work_calc = np.zeros_like(T_2)
T_2_calc = np.zeros_like(T_2)
v_2_calc = np.zeros_like(T_2)
P_calc = np.zeros_like(T_2)
eta_is_calc = np.zeros_like(T_2)

for i in range(len(T_2)) :
    for j in range(len(T_2[0])) :
        if not np.isnan(T_2[i,j]) :
            compressor = Compressor_3_params(params_opt[0], eta_v[i,j], eta_is_max=params_opt[1], eta_elme = 0.95)
            state_1 = State(heos, T = T_1[i,j], p = p_1[i,j])
            specific_work_calc[i,j], T_2_calc[i,j] = compressor.Solve(state_1, p_2[i,j])[0:2]
            P_calc[i,j] = specific_work_calc[i,j] * mdot[i,j]

            heos.update(CoolProp.PT_INPUTS, p_2[i,j], T_2_calc[i,j])
            v_2_calc[i,j] = 1/heos.rhomass()
            h_2_calc = heos.hmass()
            heos.update(CoolProp.PSmass_INPUTS, p_2[i,j], state_1.s)
            h_2is = heos.hmass()
            eta_is_calc[i,j] = (h_2is - state_1.h) / (h_2_calc - state_1.h)
        else :
            specific_work_calc[i,j] = np.nan
            T_2_calc[i,j] = np.nan
            v_2_calc[i,j] = np.nan
            P_calc[i,j] = np.nan
            eta_is_calc[i,j] = np.nan

eta_is = np.zeros_like(T_2)
v_2 = np.zeros_like(T_2)

for i in range(len(T_2)) :
    for j in range(len(T_2[0])) :
        if not np.isnan(T_2[i,j]) :
            heos.update(CoolProp.PT_INPUTS, p_1[i,j], T_1[i,j])
            h_1 = heos.hmass()
            s_1 = heos.smass()
            
            heos.update(CoolProp.PT_INPUTS, p_2[i,j], T_2[i,j])
            h_2 = heos.hmass()
            s_2 = heos.smass()
            v_2[i,j] = 1/heos.rhomass()
            
            heos.update(CoolProp.PSmass_INPUTS, p_2[i,j], s_1)
            h_2is = heos.hmass()

            eta_is[i,j] = (h_2is - h_1) / (h_2 - h_1)

        else :
            eta_is[i,j] = np.nan
            v_2[i,j] = np.nan


plt.figure(figsize=(8,6))
for i in range(len(T_evap)) :
    plt.plot(v_1[:,i]/v_2[:,i], eta_is[:,i], 'o', label=f'Measured data T_evap={T_evap[i]-273.15}°C')
plt.plot(v_1.flatten()/v_2_calc.flatten(), eta_is_calc.flatten(), 'x', label=f'Fitted data')
plt.legend()
#plt.show()

colors = color_palette("tab10", 4)
plt.figure(figsize=(8,6))
plt.plot(v_1.flatten()/v_2.flatten(), eta_is.flatten(), 'o', color = colors[2], label='Measured data')
plt.plot(v_1.flatten()/v_2_calc.flatten(), eta_is_calc.flatten(), 'x', color = colors[3], label='Fitted data')
plt.hlines(params_opt[1],
            xmin=np.nanmin(v_1.flatten()/v_2.flatten()),
            xmax=np.nanmax(v_1.flatten()/v_2.flatten()),
            colors='black', linestyles='dashed',
            label=r'Fitted $\eta_{is,max}$' + f' = {params_opt[1]:.2f}')
plt.vlines(params_opt[0], ymin=np.nanmin(eta_is.flatten()), ymax=np.nanmax(eta_is.flatten()), colors='black',
              linestyles='dotted', label=r'Fitted BVR' + f' = {params_opt[0]:.2f}')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$\eta_{is}$')
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(8,6))
plt.plot(v_1.flatten()/v_2.flatten(), P.flatten()/1000, 'o', color = colors[2], label='Measured data')
plt.plot(v_1.flatten()/v_2_calc.flatten(), P_calc.flatten()/1000, 'x', color = colors[3], label='Fitted data')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$P$ [kW]')
plt.legend()
plt.tight_layout()
#plt.show()

error_P = 100 * np.abs((P_calc - P) / P)
error_T = 100 * np.abs((T_2_calc - T_2) / T_2)
plt.figure(figsize=(8,6))
plt.plot(v_1.flatten()/v_2.flatten(), error_P.flatten(), 'o', color = colors[2], label = 'Power')
plt.plot(v_1.flatten()/v_2_calc.flatten(), error_T.flatten(), 'x', color = colors[3], label = 'Temperature')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'Relative error [%]')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
plt.plot(v_1.flatten()/v_2.flatten(), T_2.flatten()-273.15, 'o', color = colors[2], label='Measured data')
plt.plot(v_1.flatten()/v_2_calc.flatten(), T_2_calc.flatten()-273.15, 'x', color = colors[3], label='Fitted data')
plt.xlabel(r'$v_1 / v_2$')
plt.ylabel(r'$T_2$ [°C]')
plt.legend()
plt.tight_layout()
plt.show()
'''






