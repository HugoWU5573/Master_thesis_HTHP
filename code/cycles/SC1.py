import sys
from pathlib import Path

# Add the parent directory (code) to sys.path to enable relative imports
code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from components.state import State
from components.compressor import Compressor
from components.HEX import HEX
from components.cycle_hugo import Cycle
from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

############################################################
# Parameters
############################################################

# Cycle parameters
working_fluid = 'R290'          # Working fluid
dT_1 = 3                        # Superheating at the compressor inlet [K] 
p_1 = 10e5
p_3 = 2.2247 * p_1              # Compressor outlet pressure [Pa] 
P_comp = 10e3                    # Compressor power [W] 
dT_9 = 3                        # Subcooling at the condenser outlet [K]      

# Heat source parameters
mdot_LT = 4                   # Mass flow rate [kg/s]
dT_in_LT = 10                   # Inlet temperature difference [K]
p_in_LT = 1e5                   # Inlet pressure [Pa]    
N_LT = 5                      # Heat exchanger area [m²]

# Heat sink parameters
mdot_MT = 5                  # Mass flow rate [kg/s]
dT_in_MT = 20                   # Inlet temperature difference [K]
p_in_MT = 1e5                   # Inlet pressure [Pa]
N_MT = 30                      # Heat exchanger area [m²]

# Compressor parameters
BVR = 1/2.1260                  # Built-in Volume Ratio
eta_v = 0.8                     # Volumetric efficiency
eta_is_max = 0.7                # Maximum isentropic efficiency
eta_elme = 0.95                 # Electrical-mechanical efficiency


############################################################
# Simulation
############################################################

cycle = Cycle("TwoStage_R290")

# Mass flow rates of heat source and sink
cycle.mdot_LT = mdot_LT
cycle.mdot_MT = mdot_MT

# State 1 
T_sat_1 = PropsSI('T', 'P', p_1, 'Q', 1, working_fluid)
cycle.state_1 = State(T=T_sat_1 + dT_1, p=p_1, fluid=working_fluid)

# State 3
compressor = Compressor(BVR=BVR, eta_v=eta_v, eta_is_max=eta_is_max, eta_elme=eta_elme)
cycle.mdot_wf, T_ex = compressor.modelCompressor2(P_el=P_comp, p_ex=p_3, state_in=cycle.state_1, fluid=working_fluid)
cycle.state_3 = State(p=p_3, T=T_ex, fluid=working_fluid)

'''
def iteration(x):
    N_LT, N_MT = x
    print(f"N_LT: {N_LT}, N_MT: {N_MT}")
    #
    # State 9
    cycle.state_4_prime = State(T = cycle.state_3.T - dT_in_MT, p = p_in_MT, fluid='Water')
    HEX_MT = HEX(cycle.state_4_prime, cycle.state_3, [cycle.mdot_MT, cycle.mdot_wf], fluid = ['Water', working_fluid], N = N_MT)
    T_3_prime, T_9, h_3_prime, h_9 = HEX_MT.Solve()[:4]
    cycle.state_9 = State(h = h_9, p=cycle.state_3.p, fluid=working_fluid)
    cycle.state_3_prime = State(h = h_3_prime, p=p_in_MT, fluid='Water')

    # State 10
    cycle.state_10 = State(p = cycle.state_1.p, h = cycle.state_9.h, fluid=working_fluid)

    # State 1_comp 
    cycle.state_1_prime = State(T = cycle.state_10.T + dT_in_LT, p = p_in_LT, fluid='Water')
    HEX_LT = HEX(cycle.state_10, cycle.state_1_prime, [cycle.mdot_wf, cycle.mdot_LT], fluid = [working_fluid, 'Water'], N = N_LT)
    T_1_comp, T_2_prime, h_1_comp, h_2_prime = HEX_LT.Solve()[:4]
    cycle.state_2_prime = State(h = h_2_prime, p=p_in_LT, fluid='Water')
    state_1_comp = State(h = h_1_comp, p=cycle.state_10.p, fluid=working_fluid)
    print(state_1_comp.T, cycle.state_1.T)

    return abs(state_1_comp.T - cycle.state_1.T)

#N_LT, N_MT = minimize(iteration, x0=(N_LT, N_MT), bounds=((0, 50), (0, 50))).x
'''
T_9_true = PropsSI('T', 'P', cycle.state_3.p, 'Q', 0, working_fluid) - dT_9
h_9_true = PropsSI('H', 'P', cycle.state_3.p, 'T', T_9_true, working_fluid)

def get_N_MT(N_MT) : 
    # State 9
    cycle.state_4_prime = State(T = cycle.state_3.T - dT_in_MT, p = p_in_MT, fluid='Water')
    HEX_MT = HEX(cycle.state_4_prime, cycle.state_3, [cycle.mdot_MT, cycle.mdot_wf], fluid = ['Water', working_fluid], N = N_MT)
    T_3_prime, T_9, h_3_prime, h_9 = HEX_MT.Solve()[:4]
    cycle.state_9 = State(h = h_9, p=cycle.state_3.p, fluid=working_fluid)
    cycle.state_3_prime = State(h = h_3_prime, p=p_in_MT, fluid='Water')

    print(f"N_MT: {N_MT}, {abs(cycle.state_9.T - T_9_true)}")
    return abs(cycle.state_9.h - h_9_true)

N_MT = minimize(get_N_MT, x0=(N_MT), bounds=((3, 50),),method='L-BFGS-B', tol = 1e-1).x[0]

# State 10
cycle.state_10 = State(p = cycle.state_1.p, h = cycle.state_9.h, fluid=working_fluid)
cycle.state_1_prime = State(T = cycle.state_10.T + dT_in_LT, p = p_in_LT, fluid='Water')

def get_N_LT(N_LT) : 
    # State 1_comp 
    HEX_LT = HEX(cycle.state_10, cycle.state_1_prime, [cycle.mdot_wf, cycle.mdot_LT], fluid = [working_fluid, 'Water'], N = N_LT)
    T_1_comp, T_2_prime, h_1_comp, h_2_prime = HEX_LT.Solve()[:4]
    cycle.state_2_prime = State(h = h_2_prime, p=p_in_LT, fluid='Water')
    state_1_comp = State(h = h_1_comp, p=cycle.state_10.p, fluid=working_fluid)

    print(f"N_LT: {N_LT}, {abs(cycle.state_1.T - state_1_comp.T)}")
    print(abs(state_1_comp.h - cycle.state_1.h))
    return abs(state_1_comp.h - cycle.state_1.h)

N_LT = minimize(get_N_LT, x0=(N_LT), bounds=((3, 50),), tol = 1e-1).x[0]

    



#A_LT, A_MT = minimize(iteration, x0=(A_LT_0, A_MT_0), bounds=((0, 20), (0, 20))).x
#print(f"Optimized Heat Exchanger Areas: A_LT = {A_LT:.2f} m², A_MT = {A_MT:.2f} m²")

# Plot T-s diagram with saturation curve

# Generate saturation curve for working fluid
T_min = PropsSI('Tmin', working_fluid) + 1
T_crit = PropsSI('Tcrit', working_fluid) - 1
T_sat = np.linspace(T_min, T_crit, 500)
s_liq = [PropsSI('S', 'T', T, 'Q', 0, working_fluid) for T in T_sat]
s_vap = [PropsSI('S', 'T', T, 'Q', 1, working_fluid) for T in T_sat]


# Extract states for plotting
states = [
    cycle.state_1,
    cycle.state_3,
    cycle.state_9,
    cycle.state_10,
    #state_1_comp,
]
state_labels = ['1', '3', '9', '10', "1_comp"]

T_points = [s.T for s in states]
s_points = [s.s for s in states]


plt.figure(figsize=(8,6))
plt.plot(s_liq, T_sat, 'black')
plt.plot(s_vap, T_sat, 'black')
plt.scatter(PropsSI('S', 'T', T_crit, 'Q', 0.5, 'R290'), T_crit, color='black', s=10)  # Triple point
plt.plot(s_points, T_points, 'ko-', label='Cycle')
# Add labels to each state point
for i, (s_val, T_val, label) in enumerate(zip(s_points, T_points, state_labels)):
    plt.text(s_val, T_val, f' {label}', fontsize=10, verticalalignment='bottom', horizontalalignment='left')


plt.xlabel('Entropy [J/kg-K]')
plt.legend(frameon=False)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title('Temperature [°C]', loc='left')

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move bottom and left spines away
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))

'''
plt.xlim(cycle.state_9.s * 0.95, cycle.state_3.s * 1.05)
plt.ylim(273.15, T_crit)
'''

plt.tight_layout()
plt.show()