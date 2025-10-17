import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

working_fluid = 'R290'

# Plot T-s diagram with saturation curve

# Generate saturation curve for working fluid
T_min = 300
T_crit = PropsSI('Tcrit', working_fluid) - 1
p_crit = PropsSI('Pcrit', working_fluid)
print(f'Critical pressure of {working_fluid}: {p_crit/1e5:.2f} bar')
T_sat = np.linspace(T_min, T_crit, 500)
s_liq = [PropsSI('S', 'T', T, 'Q', 0, working_fluid) for T in T_sat]
s_vap = [PropsSI('S', 'T', T, 'Q', 1, working_fluid) for T in T_sat]

# Working fluid in condenser
pressure_level = 45e5  # Pa
T5 = 400
s5 = PropsSI('S', 'P', pressure_level, 'T', T5, working_fluid)
T7 = 340
s7 = PropsSI('S', 'P', pressure_level, 'T', T7, working_fluid)

temperature_range = np.linspace(T7, T5, 100)
s_range = [PropsSI('S', 'P', pressure_level, 'T', T, working_fluid) for T in temperature_range]


plt.figure(figsize=(8,6))
plt.plot(s_liq, T_sat, 'k-')
plt.plot(s_vap, T_sat, 'k-')
plt.plot(s5, T5, 'ro')
plt.plot(s7, T7, 'ro')
plt.plot(s_range, temperature_range, 'r--')
plt.xlabel('Entropy [J/kg-K]')
plt.ylabel('Temperature [K]')
plt.tight_layout()
plt.show()