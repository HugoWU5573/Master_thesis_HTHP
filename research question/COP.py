import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def COP(T_H, T_C):
    COP_Carnot = T_H / (T_H - T_C)
    return 0.5 * COP_Carnot             # Real COP = 50% of Carnot (assumption)

# Constants
T_LT = 273.15 + 10              # Low temp source
T_HT = 273.15 + 130             # High temp sink
COP_ref = COP(T_HT, T_LT)       # COP of the reference cycle (one evaporator)

# Medium temperature source range
T_MT = np.linspace(273.15 + 10, 273.15 + 90, 1000)

# Variation of COP with T_MT
COP_multi_evap = COP(T_HT, T_MT)

# Fraction of heat available at MT
beta = np.linspace(0, 1, 1000)

# Variation of COP with T_MT and beta
COP_mult_evap_matrix = np.zeros((len(beta), len(T_MT)))

for i in range(len(beta)):
    COP_mult_evap_matrix[i, :] = (1 - beta[i]) * COP_ref + beta[i] * COP_multi_evap

# Ratio (in %)
result = (COP_mult_evap_matrix / COP_ref)

# Create meshgrid for contourf
T_MT_grid, beta_grid = np.meshgrid(T_MT, beta)

# Plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(
    T_MT_grid - 273.15,   # Convert K → °C for x-axis
    beta_grid,
    result,
    levels=40,
    cmap=matplotlib.cm.jet
)
cbar = plt.colorbar(contour, orientation='vertical')
cbar.set_label(r'$\frac{COP_{dual}}{COP_{single}}$', rotation=0, labelpad=60, fontsize=18, loc='top')
cbar.set_ticks([1, 1.5, 2, 2.5, 3])  # Set desired ticks here
plt.xlabel(r'$T_{MT}$ [$^\circ C$]',labelpad=10, fontsize=16)
plt.ylabel(r'$\beta$ $[-]$', rotation=0, labelpad=20, fontsize=16)
#plt.title(r'$\frac{COP_{dual}}{COP_{single}}$', loc='right', fontsize=12, pad=30)
plt.tight_layout()
plt.show()