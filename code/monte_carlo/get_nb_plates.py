import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

save_figs = False
target_CDF_1 = 0.85
target_CDF_2 = 0.85
target_CDF_3 = 0.85
target_CDF_4 = 0.85
target_CDF_5 = 0.85

colors = {
    "SC1" : "#1f77b4",
    "SC1R": "#ff7f0e",
    "SC2": "#2ca02c",
    "SC2R": "#d62728",
    "TC1": "#9467bd",
    "TC1R": "#8c564b",
    "TC2": "#E900E9",
    "TC2R": "#7f7f7f",
}

######################################################
## 1. Evaporator LT
######################################################

# Parameters for the ACP70X heat exchanger
xmin = 4
xmax = 124

cycles_involved_1 = ["SC1", "SC1R", "SC2", "SC2R", "TC2", "TC2R"]
colors_1 = [colors[cycle] for cycle in cycles_involved_1]
objective_1 = "Evap_LT_CDF_data.csv"
Nb_plates_1 = []
cdf_1 = []

for i in range(len(cycles_involved_1)):
    file_name = "code/monte_carlo/data/" + cycles_involved_1[i] + "/" + objective_1
    data = pd.read_csv(file_name, skiprows=1, header=None).values
    Nb_plates_1.append(data[:, 0])
    cdf_1.append(data[:, 1])

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_1 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_1)):
    cdf_array_interpolated_1 = np.interp(nb_plates_interpolated, Nb_plates_1[i], cdf_1[i])
    if np.any(cdf_array_interpolated_1 >= target_CDF_1):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_1 >= target_CDF_1][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_1:.0%} is not reached for the cycle {cycles_involved_1[i]} in the Evaporator LT case.")
    min_Nb_plates_target_CDF_1.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_1_unrounded = np.nanmax(min_Nb_plates_target_CDF_1)
min_Nb_plates_target_CDF_1 = np.round(min_Nb_plates_target_CDF_1_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_1:.0%} for the Evaporator LT : {min_Nb_plates_target_CDF_1:.0f} plates")

plt.figure("Evaporator LT - CDF")

for i in range(len(cycles_involved_1)):
    plt.plot(Nb_plates_1[i], cdf_1[i],
            label=cycles_involved_1[i],
            clip_on=False, color=colors_1[i], zorder=2)
    
plt.axhline(y=target_CDF_1, color='k', linestyle=':', label=f'Target : {target_CDF_1:.0%}', zorder=1)

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Cumulated Distribution Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_1, xmax]
yticks = [0, 1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='lower right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs : plt.savefig("code/monte_carlo/figures/Evaporator_LT_CDFs.pdf")

plt.show()


######################################################
## 2. Evaporator MT
######################################################

# Geometric parameters for the ACP70X heat exchanger
xmin = 4
xmax = 124

cycles_involved_2 = ["SC2", "SC2R", "TC1", "TC1R", "TC2", "TC2R"]
colors_2 = [colors[cycle] for cycle in cycles_involved_2]
objective_2 = "Evap_MT_CDF_data.csv"
Nb_plates_2 = []
cdf_2 = []

for i in range(len(cycles_involved_2)):
    file_name = "code/monte_carlo/data/" + cycles_involved_2[i] + "/" + objective_2
    data = pd.read_csv(file_name, skiprows=1, header=None).values
    Nb_plates_2.append(data[:, 0])
    cdf_2.append(data[:, 1])


# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_2 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_2)):
    cdf_array_interpolated_2 = np.interp(nb_plates_interpolated, Nb_plates_2[i], cdf_2[i])

    if np.any(cdf_array_interpolated_2 >= target_CDF_2):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_2 >= target_CDF_2][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_2:.0%} is not reached for the cycle {cycles_involved_2[i]} in the Evaporator MT case.")
    min_Nb_plates_target_CDF_2.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_2_unrounded = np.nanmax(min_Nb_plates_target_CDF_2)
min_Nb_plates_target_CDF_2 = np.round(min_Nb_plates_target_CDF_2_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_2:.0%} for the Evaporator MT : {min_Nb_plates_target_CDF_2:.0f} plates")


plt.figure("Evaporator MT - CDF")

for i in range(len(cycles_involved_2)):
    plt.plot(Nb_plates_2[i], cdf_2[i],
            label=cycles_involved_2[i],
            clip_on=False, color=colors_2[i], zorder=2)
    
plt.axhline(y=target_CDF_2, color='k', linestyle=':', label=f'Target : {target_CDF_2:.0%}', zorder=1)

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Cumulated Distribution Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_2, xmax]
yticks = [0, 1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='lower right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs : plt.savefig("code/monte_carlo/figures/Evaporator_MT_CDFs.pdf")

plt.show()

######################################################
## 3. Condenser/Gas Cooler
######################################################

# Geometric parameters for the ACP70X heat exchanger
xmin = 4
xmax = 124

cycles_involved_3 = ["SC1", "SC1R", "SC2", "SC2R", "TC1", "TC1R", "TC2", "TC2R"]
colors_3 = [colors[cycle] for cycle in cycles_involved_3]
objectives_3 = ["Condenser_CDF_data.csv", "Condenser_CDF_data.csv", 
                "Condenser_CDF_data.csv", "Condenser_CDF_data.csv", 
                "GasCooler_CDF_data.csv", "GasCooler_CDF_data.csv",
                "GasCooler_CDF_data.csv", "GasCooler_CDF_data.csv"]
Nb_plates_3 = []
cdf_3 = []

for i in range(len(cycles_involved_3)):
    file_name = "code/monte_carlo/data/" + cycles_involved_3[i] + "/" + objectives_3[i]
    data = pd.read_csv(file_name, skiprows=1, header=None).values
    Nb_plates_3.append(data[:, 0])
    cdf_3.append(data[:, 1])

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_3 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_3)):
    cdf_array_interpolated_3 = np.interp(nb_plates_interpolated, Nb_plates_3[i], cdf_3[i])
    if np.any(cdf_array_interpolated_3 >= target_CDF_3):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_3 >= target_CDF_3][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_3:.0%} is not reached for the cycle {cycles_involved_3[i]} in the Condenser/Gas Cooler case.")
    min_Nb_plates_target_CDF_3.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_3_unrounded = np.nanmax(min_Nb_plates_target_CDF_3)
min_Nb_plates_target_CDF_3 = np.round(min_Nb_plates_target_CDF_3_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_3:.0%} for the Condenser/GasCooler : {min_Nb_plates_target_CDF_3:.0f} plates")

plt.figure("Condenser/GasCooler - CDF")

for i in range(len(cycles_involved_3)):
    plt.plot(Nb_plates_3[i], cdf_3[i],
            label=cycles_involved_3[i],
            clip_on=False, color=colors_3[i], zorder=2)
    
plt.axhline(y=target_CDF_3, color='k', linestyle=':', label=f'Target : {target_CDF_3:.0%}', zorder=1)

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Cumulated Distribution Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_3, xmax]
yticks = [0, 1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='lower right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs : plt.savefig("code/monte_carlo/figures/Condenser_GasCooler_CDFs.pdf")

plt.show()

######################################################
## 4. Recuperator LT
######################################################

# Geometric parameters for the ACK18 heat exchanger
xmin = 4
xmax = 52

cycles_involved_4 = ["SC1R", "SC2R", "TC2R"]
colors_4 = [colors[cycle] for cycle in cycles_involved_4]
objective_4 = "Recup_LT_CDF_data.csv"
Nb_plates_4 = []
cdf_4 = []

for i in range(len(cycles_involved_4)):
    file_name = "code/monte_carlo/data/" + cycles_involved_4[i] + "/" + objective_4
    data = pd.read_csv(file_name, skiprows=1, header=None).values
    Nb_plates_4.append(data[:, 0])
    cdf_4.append(data[:, 1])

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_4 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_4)):
    cdf_array_interpolated_4 = np.interp(nb_plates_interpolated, Nb_plates_4[i], cdf_4[i])
    if np.any(cdf_array_interpolated_4 >= target_CDF_4):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_4 >= target_CDF_4][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_4:.0%} is not reached for the cycle {cycles_involved_4[i]} in the Recuperator LT case.")
    min_Nb_plates_target_CDF_4.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_4_unrounded = np.nanmax(min_Nb_plates_target_CDF_4)
min_Nb_plates_target_CDF_4 = np.round(min_Nb_plates_target_CDF_4_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_4:.0%} for the Recuperator LT : {min_Nb_plates_target_CDF_4:.0f} plates")

plt.figure("Recuperator LT - CDF")

for i in range(len(cycles_involved_4)):
    plt.plot(Nb_plates_4[i], cdf_4[i],
            label=cycles_involved_4[i],
            clip_on=False, color=colors_4[i], zorder=2)
    
plt.axhline(y=target_CDF_4, color='k', linestyle=':', label=f'Target : {target_CDF_4:.0%}', zorder=1)

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Cumulated Distribution Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_4, xmax]
yticks = [0, 1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='lower right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs : plt.savefig("code/monte_carlo/figures/Recuperator_LT_CDFs.pdf")

plt.show()

######################################################
## 5. Recuperator MT
######################################################

# Geometric parameters for the ACK18 heat exchanger
xmin = 4
xmax = 52

cycles_involved_5 = ["SC2R", "TC1R", "TC2R"]
colors_5 = [colors[cycle] for cycle in cycles_involved_5]
objective_5 = "Recup_MT_CDF_data.csv"
Nb_plates_5 = []
cdf_5 = []

for i in range(len(cycles_involved_5)):
    file_name = "code/monte_carlo/data/" + cycles_involved_5[i] + "/" + objective_5
    data = pd.read_csv(file_name, skiprows=1, header=None).values
    Nb_plates_5.append(data[:, 0])
    cdf_5.append(data[:, 1])

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_5 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_5)):
    cdf_array_interpolated_5 = np.interp(nb_plates_interpolated, Nb_plates_5[i], cdf_5[i])
    if np.any(cdf_array_interpolated_5 >= target_CDF_5):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_5 >= target_CDF_5][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_5:.0%} is not reached for the cycle {cycles_involved_5[i]} in the Recuperator MT case.")
    min_Nb_plates_target_CDF_5.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_5_unrounded = np.nanmax(min_Nb_plates_target_CDF_5)
min_Nb_plates_target_CDF_5 = np.round(min_Nb_plates_target_CDF_5_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_5:.0%} for the Recuperator MT : {min_Nb_plates_target_CDF_5:.0f} plates")

plt.figure("Recuperator MT - CDF")

for i in range(len(cycles_involved_5)):
    plt.plot(Nb_plates_5[i], cdf_5[i],
            label=cycles_involved_5[i],
            clip_on=False, color=colors_5[i], zorder=2)
    
plt.axhline(y=target_CDF_5, color='k', linestyle=':', label=f'Target : {target_CDF_5:.0%}', zorder=1)

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Cumulated Distribution Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_5, xmax]
yticks = [0, 1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='lower right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs : plt.savefig("code/monte_carlo/figures/Recuperator_MT_CDFs.pdf")

plt.show()