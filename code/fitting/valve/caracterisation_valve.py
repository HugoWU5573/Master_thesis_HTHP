import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from components.valve import Valve

'''
opening = np.array([50, 100, 150, 200, 250, 400]) / 480                    #convert in openings
volume_flow_rate = np.array([0, 1.4, 2.6, 3.9, 4.6, 7.6 ])/3600         #convert in m3/s
delta_p = 1e5
valve = Valve(a_0 = None, a_1 = None)

# Assuming standard air condition 
T_0 = 20 + 273.15 
p_0 = 2e5
rho_in = PropsSI('D', 'T', T_0, 'P', p_0, 'Air')

flow_coeffcient = volume_flow_rate * rho_in / np.sqrt(2 * rho_in * delta_p)
print(flow_coeffcient)

# Fit the valve coefficients a_0 and a_1 using linear regression
A = np.polyfit(opening, flow_coeffcient, 1)
valve.a_0, valve.a_1 = A
print(f"Fitted valve coefficients: a_0 = {valve.a_0}, a_1 = {valve.a_1}")

linspace_opening = np.linspace(0, 1, 100)
fitted_flow_coeffcient = np.polyval(A, linspace_opening)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(opening, flow_coeffcient, color='red', label='Experimental Data')
plt.plot(linspace_opening, fitted_flow_coeffcient, color='blue', label='Fitted Line')
plt.xlabel('Valve Opening')
plt.ylabel('Air Flow Rate (kg/s)')
plt.legend()
plt.grid()
plt.show()
'''

fluid = 'R410a'

t_e = 5 + 273.15
t_c = 38 + 273.15

h_evap_out = PropsSI('H', 'T', t_e, 'Q', 1, fluid)
h_valve_in = PropsSI('H', 'T', t_c, 'Q', 0, fluid)
h_evap_in = h_valve_in
rho_valve_in = PropsSI('D', 'T', t_c, 'Q', 0, fluid)
p_valve_in = PropsSI('P', 'T', t_c, 'Q', 0, fluid)
p_valve_out = PropsSI('P', 'T', t_e, 'Q', 1, fluid)

opening = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450]) / 480 * 100                    
Q_e = np.array([4.5, 11.5, 17.5, 19.5, 23.0, 27.0, 30, 32, 36]) * 1e3
mdot = Q_e / (h_evap_out - h_evap_in)
flow_coefficent = mdot / np.sqrt(2 * rho_valve_in * (p_valve_in - p_valve_out))

A = np.polyfit(opening, flow_coefficent, 2)
print(f"Fitted valve coefficients: a_0 = {A[2]}, a_1 = {A[1]}, a_2 = {A[0]}")

linspace_opening = np.linspace(10, 100, 100)
fitted_flow_coefficent = np.polyval(A, linspace_opening)
fitted_mdot = fitted_flow_coefficent * np.sqrt(2 * rho_valve_in * (p_valve_in - p_valve_out))


plt.figure(figsize=(8, 6))
plt.scatter(opening, mdot * 3600, color='black', label='Experimental Data', clip_on=False)
plt.plot(linspace_opening, fitted_mdot * 3600, color='black', label='Fitted Line', clip_on=False)
plt.xlabel(r'$z$ [%]', fontsize = 12)
#plt.legend(fontsize = 12)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$\dot{m}$ [kg/s]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(10, 100)
#plt.ylim(0.025, 0.225)

plt.show()






    