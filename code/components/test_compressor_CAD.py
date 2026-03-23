import CoolProp
from state import State
from compressor import Compressor_LP, Compressor_HP
import numpy as np
import matplotlib.pyplot as plt

fluid = 'R290'
heos = CoolProp.AbstractState("HEOS&TTSE", fluid)

compressor = Compressor_LP()

#"=================LP testing====================="

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

p_2_calc = np.zeros_like(T_2)
T_2_calc = np.zeros_like(T_2)

for i in range(len(N)) :
    for j in range(len(T_2[0])) :
        for k in range(len(T_2[0,0])) :
            if not np.isnan(T_2[i,j,k]) :
                state_in = State(heos, p = p_1[i,j,k], T = T_1[i,j,k])
                p_2_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_in, P[i,j,k], mdot[i,j,k], N[i])
            else : 
                p_2_calc[i,j,k] = np.nan
                T_2_calc[i,j,k] = np.nan

error_min_p = np.nanmin(p_2_calc.flatten() - p_2.flatten())
error_max_p = np.nanmax(p_2_calc.flatten() - p_2.flatten())
error_min_T = np.nanmin(T_2_calc.flatten() - T_2.flatten())
error_max_T = np.nanmax(T_2_calc.flatten() - T_2.flatten())
print(T_2.flatten() + error_min_T - 273.15, T_2.flatten() + error_max_T - 273.15)

sorted_indices = np.argsort(p_2.flatten())
p_2_sorted = p_2.flatten()[sorted_indices]
p_2_calc_sorted = p_2_calc.flatten()[sorted_indices]
plt.figure(figsize=(8, 6))
plt.plot(p_2_sorted/1e5, p_2_calc_sorted/1e5, 'kx')
plt.plot(p_2_sorted/1e5, p_2_sorted/1e5, 'k--', label='Ideal')
plt.fill_between(p_2_sorted/1e5, (p_2_sorted + error_min_p)/1e5, (p_2_sorted + error_max_p)/1e5, color = 'orange', alpha=0.3, label='Error Range')
plt.xlabel(r'$p_{meas}$ [bar]', fontsize=12)
plt.text(p_2_sorted[len(p_2_sorted)//2]/1e5, (p_2_sorted[len(p_2_sorted)//2] + error_min_p)/1e5 - 1, r'$\Delta p_{min}$' + f': {error_min_p/1e5:.2f} bar', fontsize=10)
plt.text(p_2_sorted[len(p_2_sorted)//2]/1e5, (p_2_sorted[len(p_2_sorted)//2] + error_max_p)/1e5 + 2.5, r'$\Delta p_{max}$' + f': {error_max_p/1e5:.2f} bar', fontsize=10)
#plt.axis('equal')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$p_{calc}$ [bar]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(10, 22)
plt.ylim(10, 22)
plt.show()

sorted_indices = np.argsort(T_2.flatten())
T_2_sorted = T_2.flatten()[sorted_indices]
T_2_calc_sorted = T_2_calc.flatten()[sorted_indices]

plt.figure(figsize=(8, 6))
plt.plot(T_2_sorted - 273.15, T_2_calc_sorted - 273.15, 'kx', clip_on=False)
plt.plot(T_2_sorted - 273.15, T_2_sorted - 273.15, 'k--', label='Ideal', clip_on=False)
plt.fill_between(T_2_sorted - 273.15, T_2_sorted + error_min_T - 273.15, T_2_sorted + error_max_T - 273.15, color = 'orange', alpha=0.3, label='Error Range')
plt.text(T_2_sorted[len(T_2_sorted)//2] - 273.15, T_2_sorted[len(T_2_sorted)//2] + error_min_T - 273.15 - 5, r'$\Delta T_{min}$' + f': {error_min_T:.2f} °C', fontsize=10)
plt.text(T_2_sorted[len(T_2_sorted)//2] - 273.15, T_2_sorted[len(T_2_sorted)//2] + error_max_T - 273.15 + 8, r'$\Delta T_{max}$' + f': {error_max_T:.2f} °C', fontsize=10)
plt.xlabel(r'$T_{meas}$ [°C]', fontsize=12)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$T_{calc}$ [°C]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(50, 90)
plt.ylim(50, 90)

plt.show()

#=================HP testing=====================

compressor = Compressor_HP()

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

p_2_calc = np.zeros_like(T_2)
T_2_calc = np.zeros_like(T_2)

for i in range(len(N)) :
    for j in range(len(T_2[0])) :
        for k in range(len(T_2[0,0])) :
            if not np.isnan(T_2[i,j,k]) :
                state_in = State(heos, p = p_1[i,j,k], T = T_1[i,j,k])
                p_2_calc[i,j,k], T_2_calc[i,j,k] = compressor.Solve(state_in, P[i,j,k], mdot[i,j,k], N[i])
            else : 
                p_2_calc[i,j,k] = np.nan
                T_2_calc[i,j,k] = np.nan

error_min_p = np.nanmin(p_2_calc.flatten() - p_2.flatten())
error_max_p = np.nanmax(p_2_calc.flatten() - p_2.flatten())
error_min_T = np.nanmin(T_2_calc.flatten() - T_2.flatten())
error_max_T = np.nanmax(T_2_calc.flatten() - T_2.flatten())

sorted_indices = np.argsort(p_2.flatten())
p_2_sorted = p_2.flatten()[sorted_indices]
p_2_calc_sorted = p_2_calc.flatten()[sorted_indices]

plt.figure(figsize=(8, 6))
plt.plot(p_2_sorted.flatten()/1e5, p_2_calc_sorted.flatten()/1e5, 'kx', clip_on=False)
plt.plot(p_2_sorted.flatten()/1e5, p_2_sorted.flatten()/1e5, 'k-', label='Ideal', clip_on=False)
plt.fill_between(p_2_sorted.flatten()/1e5, (p_2_sorted.flatten() + error_min_p)/1e5, (p_2_sorted.flatten() + error_max_p)/1e5, color = 'orange', alpha=0.3, label='Error Range')
plt.text(p_2_sorted[len(p_2_sorted)//2]/1e5, (p_2_sorted[len(p_2_sorted)//2] + error_min_p)/1e5 - 1, r'$\Delta p_{min}$' + f': {error_min_p/1e5:.2f} bar', fontsize=10)
plt.text(p_2_sorted[len(p_2_sorted)//2]/1e5, (p_2_sorted[len(p_2_sorted)//2] + error_max_p)/1e5 + 2.5, r'$\Delta p_{max}$' + f': {error_max_p/1e5:.2f} bar', fontsize=10)
plt.xlabel(r'$p_{meas}$ [bar]', fontsize=12)
#plt.axis('equal')
# Add some text for labels, title and custom x-axis tick labels, etc.   
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$p_{calc}$ [bar]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))
plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(40, 50)
plt.ylim(40, 50)
plt.show()

sorted_indices = np.argsort(T_2.flatten())
T_2_sorted = T_2.flatten()[sorted_indices]
T_2_calc_sorted = T_2_calc.flatten()[sorted_indices]

plt.figure(figsize=(8, 6))
plt.plot(T_2_sorted.flatten() - 273.15, T_2_calc_sorted.flatten() - 273.15, 'kx', clip_on=False)
plt.plot(T_2_sorted.flatten() - 273.15, T_2_sorted.flatten() - 273.15, 'k-', label='Ideal', clip_on=False)
plt.fill_between(T_2_sorted.flatten() - 273.15, T_2_sorted.flatten() + error_min_T - 273.15, T_2_sorted.flatten() + error_max_T - 273.15, color = 'orange', alpha=0.3, label='Error Range')
plt.text(T_2_sorted[len(T_2_sorted)//2] - 273.15, T_2_sorted[len(T_2_sorted)//2] + error_min_T - 273.15 - 5, r'$\Delta T_{min}$' + f': {error_min_T:.2f} °C', fontsize=10)
plt.text(T_2_sorted[len(T_2_sorted)//2] - 273.15, T_2_sorted[len(T_2_sorted)//2] + error_max_T - 273.15 + 8, r'$\Delta T_{max}$' + f': {error_max_T:.2f} °C', fontsize=10)
plt.xlabel(r'$T_{meas}$ [°C]', fontsize=12)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$T_{calc}$ [°C]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(110, 145)
plt.ylim(110, 145)

plt.show()

