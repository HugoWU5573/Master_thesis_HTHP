import sys
from pathlib import Path

# Add the parent directory (code) to sys.path to enable relative imports
code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from components.state import state
from components.compressor import Compressor
from components.HEX import HEX
from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt
import numpy as np

P_comp = 5e3  # Compressor electrical power [W]
p_1 = 2e5   # Evaporator pressure [Pa]
T_1 = 273.15 - 10 # Evaporator temperature [K]
p_3 = 2.2477 * p_1   # Condenser pressure [Pa]
fluid = 'R290' # Working fluid

Q_evap = 5e3 # Evaporator heat load [W]
T0 = 273.15 + 20 # Reference temperature for exergy calculation [K]


state_1 = state(T=T_1, p=p_1, fluid=fluid)

Compressor1 = Compressor(BVR= 1 / 2.1260, eta_v=0.9, eta_is_max=0.60, eta_elme=0.98)
mdot_wf, T_ex = Compressor1.modelCompressor2(P_el=P_comp, p_ex=p_3, state_in=state_1, fluid=fluid)
state_2 = state(p=p_3, T=T_ex, fluid=fluid)

# Calculate entropy for each state
s_1 = state_1.s
s_2 = state_2.s

fluid = 'R290'
T_min = PropsSI('TMIN', fluid)
T_max = PropsSI('TCRIT', fluid) - 1  # Avoid critical point

T_range = np.linspace(T_min, T_max, 300)
s_liq = [PropsSI('S', 'T', T, 'Q', 0, fluid) for T in T_range]
s_vap = [PropsSI('S', 'T', T, 'Q', 1, fluid) for T in T_range]

plt.figure(figsize=(8,6))
plt.plot(s_liq, T_range - 273.15, label='Saturated Liquid')
plt.plot(s_vap, T_range - 273.15, label='Saturated Vapor')
plt.xlabel('Entropy [J/kg-K]')
plt.ylabel('Temperature [°C]')
plt.title('T-s Diagram of Propane (R290)')

plt.scatter([s_1], [state_1.T - 273.15], color='red', label='State 1', zorder=5)
plt.annotate('1', (s_1, state_1.T - 273.15), textcoords="offset points", xytext=(10,10), ha='center', color='red')
plt.scatter([s_2], [state_2.T - 273.15], color='blue', label='State 2', zorder=5)
plt.annotate('2', (s_2, state_2.T - 273.15), textcoords="offset points", xytext=(10,10), ha='center', color='blue')

plt.xlim((0,3e3))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()