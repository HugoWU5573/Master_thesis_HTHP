import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def COP(T_H, T_C):
    COP_Carnot = T_H / (T_H - T_C)
    return COP_Carnot * 0.62

# Constants
T_LT = 273.15 + 15              # Low temp source
T_HT = 273.15 + 115             # High temp sink
COP_ref = COP(T_HT, T_LT)       # COP of the reference cycle (one evaporator)

# Medium temperature source range
T_MT = np.linspace(273.15 + 15, 273.15 + 90, 1000)

# Variation of COP with T_MT
COP_multi_evap = COP(T_HT, T_MT)

# Fraction of heat available at MT
beta = np.linspace(0.01, 1, 1000)

# Variation of COP with T_MT and beta
COP_mult_evap_matrix = np.zeros((len(beta), len(T_MT)))

for i in range(len(beta)):
    COP_mult_evap_matrix[i, :] = (1 - beta[i]) * COP_ref + beta[i] * COP_multi_evap

# Ratio (in %)
result = (COP_mult_evap_matrix / COP_ref)

# Create meshgrid for contourf
T_MT_grid, beta_grid = np.meshgrid(T_MT, beta)

# Complete plot
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


# Simplified plot assuming T_MT is fixed at 45°C

    # Theoreticl results 
T_MT_45 = 273.15 + 45
idx_45 = np.argmin(np.abs(T_MT - T_MT_45))
COP_vs_beta_th = COP_mult_evap_matrix[:, idx_45]

    # Actual results from TC2 cycle simulation
beta_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
COP_vs_beta_TC2 = [2.415, 2.485, 2.57, 2.665, 2.773, 2.894, 3.018, 3.172, 3.334, 3.443, 3.636]

# Plot COP ratio vs beta
plt.figure()
plt.plot(beta, COP_vs_beta_th, label=r"$Carnot$-$based$ $model$", color='blue', zorder=3)
plt.scatter(beta_range, COP_vs_beta_TC2, color='red', label=r"$Simulation$ $results$", marker="o", zorder =4)
plt.xlabel(r'$\beta$  $[-]$', fontsize=12)
plt.xlim(0, 1)
plt.ylim(2, 4)
ax = plt.gca()
ax.tick_params(axis='both', which='major')
ax.set_title(r'$COP$  $[-]$', loc='left', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))
ax.set_xticks(np.arange(0, 1.1, 0.2))
ax.set_yticks(np.arange(2, 4.5, 1))
plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(frameon=False, fontsize=10, loc="lower right")
plt.tight_layout()
plt.show()