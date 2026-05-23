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

# For ETS6 - 18
Q_e = np.array([1.25, 2.5, 4.17, 5.5, 6.5, 7.5, 8, 10.5, 12.5]) * 1e3

# For ETS6 - 25
Q_e = np.array([2.5, 7.5, 12, 15.5, 18, 21, 22, 23, 23.5]) * 1e3

# For ETS6 - 32 
#Q_e = np.array([4.5, 11.5, 17.5, 19.5, 23.0, 27.0, 30, 32, 36]) * 1e3
mdot = Q_e / (h_evap_out - h_evap_in)
print(mdot)
flow_coefficent = mdot / np.sqrt(2 * rho_valve_in * (p_valve_in - p_valve_out))

A = np.polyfit(opening, flow_coefficent, 2)
print(f"Fitted valve coefficients: A = {A}")

mdot_linspace = np.linspace(min(mdot), max(mdot), 100)
mdot_calc = np.polyval(A, opening) * np.sqrt(2 * rho_valve_in * (p_valve_in - p_valve_out))

print("Max error in mass flow rate: ", np.max(np.abs(mdot_calc - mdot) / mdot))

plt.figure()
plt.scatter(mdot, mdot_calc, color = 'black', label='Calculated', clip_on=False)
plt.plot(mdot_linspace, mdot_linspace, color='black', label='Experimental', clip_on=False)
plt.xlabel(r'$\dot{m}_{meas}$ [kg/s]', fontsize = 12)
#plt.legend(fontsize = 12)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$\dot{m}_{calc}$ [kg/s]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(0.01, 0.15)
plt.ylim(0.01, 0.15)
plt.tight_layout()
plt.savefig('code/fitting/valve/flow_coefficient.pdf', dpi=300)
#plt.show()

linspace_opening = np.linspace(10, 100, 100)
fitted_flow_coefficent = np.polyval(A, linspace_opening)
fitted_mdot = fitted_flow_coefficent * np.sqrt(2 * rho_valve_in * (p_valve_in - p_valve_out))

plt.figure()
plt.plot(linspace_opening, fitted_flow_coefficent, color='black', clip_on=False)
plt.xlabel(r'$z$ [%]', fontsize = 12)
#plt.legend(fontsize = 12)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$C_D$', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(10, 100)
plt.ylim(0, 3e-6)
plt.tight_layout()
plt.savefig('code/fitting/valve/flow_coefficient.pdf', dpi=300)
#plt.show()

print(PropsSI('D', 'P', 40e5, 'Q', 0, 'R290'))
print(PropsSI('D', 'P', 20e5, 'Q', 0, 'R290'))

rho_in = PropsSI('D', 'P', 40e5, 'Q', 0, 'R290')
dp = np.linspace(0, 40e5, 100)
openings = np.array([10, 30, 60, 90])
mdot = np.zeros((len(openings), len(dp)))
for i, opening in enumerate(openings):
    for j, delta_p in enumerate(dp):
        flow_coefficent = np.polyval(A, opening)
        mdot[i, j] = flow_coefficent * np.sqrt(2 * rho_in * delta_p)

plt.figure()
for i, opening in enumerate(openings):
    plt.plot(np.sqrt(dp/40e5), mdot[i, :], label=rf'$z = {{{opening}}}\%$', clip_on=False)
plt.xlabel(r'$\sqrt{\Delta p / p_{in}}$', fontsize = 12)
#plt.legend(fontsize = 12)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$\dot{m}$ [kg/s]', loc='left', fontsize=12)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.xlim(0, 1)
plt.ylim(0, 0.14)

plt.tight_layout()
plt.legend(frameon=False, loc='best', fontsize=12)
plt.savefig('code/fitting/valve/mass_flow_rate.pdf', dpi=300)
plt.show()






    