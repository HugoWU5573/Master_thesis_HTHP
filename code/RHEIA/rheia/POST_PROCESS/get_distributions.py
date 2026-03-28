import numpy as np
import matplotlib.pyplot as plt
import post_process as rheia_pp

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

# Geometric parameters for the ACP70X heat exchanger
L = 0.526 
W = 0.111
phi = 1.2
xmin = 4
xmax = 124

cycles_involved_1 = ["SC1", "SC1R", "SC2", "SC2R", "TC2", "TC2R"]
colors_1 = [colors[cycle] for cycle in cycles_involved_1]
PCE_orders_1 = [4, 4, 3, 3, 3, 2]
objectives_1 = ["log_A_evap", "log_A_evap", "log_A_evap_LT", "log_A_evap_LT", "log_A_evap_LT", "log_A_evap_LT"]
Nb_plates_1_pdf = []
Nb_plates_1_cdf = []
pdf_1 = []
cdf_1 = []

for i in range(len(cycles_involved_1)):
    my_post_process_uq = rheia_pp.PostProcessUQ(cycles_involved_1[i], PCE_orders_1[i])
    log_A_pdf, pdf = my_post_process_uq.get_pdf("results", objectives_1[i])
    log_A_cdf, cdf = my_post_process_uq.get_cdf("results", objectives_1[i])

    A_geom_pdf = np.exp(log_A_pdf)
    Nb_plates_pdf = A_geom_pdf / (W*L*phi) + 2
    Nb_plates_array_pdf = np.array(Nb_plates_pdf)
    Nb_plates_1_pdf.append(Nb_plates_array_pdf)

    pdf_array = np.array(pdf)

    # Normalize the PDF to ensure that the area under the curve is equal to 1
    area = np.trapz(pdf_array, Nb_plates_array_pdf)
    pdf_array /= area
    pdf_array[Nb_plates_array_pdf > xmax] = np.nan
    pdf_array[Nb_plates_array_pdf < xmin] = np.nan
    pdf_1.append(pdf_array)

    A_geom_cdf = np.exp(log_A_cdf)
    Nb_plates_cdf = A_geom_cdf / (W*L*phi) + 2
    Nb_plates_array_cdf = np.array(Nb_plates_cdf)
    Nb_plates_1_cdf.append(Nb_plates_array_cdf)

    cdf_array = np.array(cdf)
    cdf_array[Nb_plates_array_cdf > xmax] = np.nan
    cdf_array[Nb_plates_array_cdf < xmin] = np.nan
    cdf_1.append(cdf_array)

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_1 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_1)):
    cdf_array_interpolated_1 = np.interp(nb_plates_interpolated, Nb_plates_1_cdf[i], cdf_1[i])
    if np.any(cdf_array_interpolated_1 >= target_CDF_1):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_1 >= target_CDF_1][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_1:.0%} is not reached for the cycle {cycles_involved_1[i]} in the Evaporator LT case.")
    min_Nb_plates_target_CDF_1.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_1_unrounded = np.nanmax(min_Nb_plates_target_CDF_1)
min_Nb_plates_target_CDF_1 = np.round(min_Nb_plates_target_CDF_1_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_1:.0%} for the Evaporator LT : {min_Nb_plates_target_CDF_1:.0f} plates")

plt.figure("Evaporator LT - PDF")

for i in range(len(cycles_involved_1)):
    plt.plot(Nb_plates_1_pdf[i], pdf_1[i],
            label=cycles_involved_1[i],
            clip_on=False, color=colors_1[i])

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 0.1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Probability Density Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_1, xmax]
yticks = [0, 0.1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '0.1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='upper right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs: plt.savefig("code/RHEIA/rheia/FIGURES/Evaporator_LT_PDFs.pdf")

plt.show()

plt.figure("Evaporator LT - CDF")

for i in range(len(cycles_involved_1)):
    plt.plot(Nb_plates_1_cdf[i], cdf_1[i],
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

if save_figs : plt.savefig("code/RHEIA/rheia/FIGURES/Evaporator_LT_CDFs.pdf")

plt.show()


######################################################
## 2. Evaporator MT
######################################################

# Geometric parameters for the ACP70X heat exchanger
L = 0.526 
W = 0.111
phi = 1.2
xmin = 4
xmax = 124

cycles_involved_2 = ["SC2", "SC2R", "TC1", "TC1R", "TC2", "TC2R"]
colors_2 = [colors[cycle] for cycle in cycles_involved_2]
PCE_orders_2 = [3, 3, 1, 3, 3, 2]
objectives_2 = ["log_A_evap_MT", "log_A_evap_MT", "log_A_evap", "log_A_evap", "log_A_evap_MT", "log_A_evap_MT"]
Nb_plates_2_pdf = []
Nb_plates_2_cdf = []
pdf_2 = []
cdf_2 = []

for i in range(len(cycles_involved_2)):
    my_post_process_uq = rheia_pp.PostProcessUQ(cycles_involved_2[i], PCE_orders_2[i])
    log_A_pdf, pdf = my_post_process_uq.get_pdf("results", objectives_2[i])
    log_A_cdf, cdf = my_post_process_uq.get_cdf("results", objectives_2[i])

    A_geom_pdf = np.exp(log_A_pdf)
    Nb_plates_pdf = A_geom_pdf / (W*L*phi) + 2
    Nb_plates_array_pdf = np.array(Nb_plates_pdf)
    Nb_plates_2_pdf.append(Nb_plates_array_pdf)

    pdf_array = np.array(pdf)

    # Normalize the PDF to ensure that the area under the curve is equal to 1
    area = np.trapz(pdf_array, Nb_plates_array_pdf)
    pdf_array /= area
    pdf_array[Nb_plates_array_pdf > xmax] = np.nan
    pdf_array[Nb_plates_array_pdf < xmin] = np.nan
    pdf_2.append(pdf_array)

    A_geom_cdf = np.exp(log_A_cdf)
    Nb_plates_cdf = A_geom_cdf / (W*L*phi) + 2
    Nb_plates_array_cdf = np.array(Nb_plates_cdf)
    Nb_plates_2_cdf.append(Nb_plates_array_cdf)

    cdf_array = np.array(cdf)
    cdf_array[Nb_plates_array_cdf > xmax] = np.nan
    cdf_array[Nb_plates_array_cdf < xmin] = np.nan
    cdf_2.append(cdf_array)

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_2 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_2)):
    cdf_array_interpolated_2 = np.interp(nb_plates_interpolated, Nb_plates_2_cdf[i], cdf_2[i])

    if np.any(cdf_array_interpolated_2 >= target_CDF_2):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_2 >= target_CDF_2][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_2:.0%} is not reached for the cycle {cycles_involved_2[i]} in the Evaporator MT case.")
    min_Nb_plates_target_CDF_2.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_2_unrounded = np.nanmax(min_Nb_plates_target_CDF_2)
min_Nb_plates_target_CDF_2 = np.round(min_Nb_plates_target_CDF_2_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_2:.0%} for the Evaporator MT : {min_Nb_plates_target_CDF_2:.0f} plates")


plt.figure("Evaporator MT - PDF")

for i in range(len(cycles_involved_2)):
    plt.plot(Nb_plates_2_pdf[i], pdf_2[i],
            label=cycles_involved_2[i],
            clip_on=False, color=colors_2[i])

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 0.1)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Probability Density Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_2, xmax]
yticks = [0, 0.1]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '0.1'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='upper right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs: plt.savefig("code/RHEIA/rheia/FIGURES/Evaporator_MT_PDFs.pdf")

plt.show()


plt.figure("Evaporator MT - CDF")

for i in range(len(cycles_involved_2)):
    plt.plot(Nb_plates_2_cdf[i], cdf_2[i],
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

if save_figs : plt.savefig("code/RHEIA/rheia/FIGURES/Evaporator_MT_CDFs.pdf")

plt.show()

######################################################
## 3. Condenser/Gas Cooler
######################################################

# Geometric parameters for the ACP70X heat exchanger
L = 0.526 
W = 0.111
phi = 1.2
xmin = 4
xmax = 124

cycles_involved_3 = ["SC1", "SC1R", "SC2", "SC2R", "TC1", "TC1R", "TC2", "TC2R"]
colors_3 = [colors[cycle] for cycle in cycles_involved_3]
PCE_orders_3 = [4, 4, 3, 3, 2, 3, 3, 2]
objectives_3 = ["log_A_cond", "log_A_cond", "log_A_cond", "log_A_cond", "log_A_GasCooler", "log_A_GasCooler", "log_A_GasCooler", "log_A_GasCooler"]
Nb_plates_3_pdf = []
Nb_plates_3_cdf = []
pdf_3 = []
cdf_3 = []

for i in range(len(cycles_involved_3)):
    my_post_process_uq = rheia_pp.PostProcessUQ(cycles_involved_3[i], PCE_orders_3[i])
    log_A_pdf, pdf = my_post_process_uq.get_pdf("results", objectives_3[i])
    log_A_cdf, cdf = my_post_process_uq.get_cdf("results", objectives_3[i])

    A_geom_pdf = np.exp(log_A_pdf)
    Nb_plates_pdf = A_geom_pdf / (W*L*phi) + 2
    Nb_plates_array_pdf = np.array(Nb_plates_pdf)
    Nb_plates_3_pdf.append(Nb_plates_array_pdf)

    pdf_array = np.array(pdf)

    # Normalize the PDF to ensure that the area under the curve is equal to 1
    area = np.trapz(pdf_array, Nb_plates_array_pdf)
    pdf_array /= area
    pdf_array[Nb_plates_array_pdf > xmax] = np.nan
    pdf_array[Nb_plates_array_pdf < xmin] = np.nan
    pdf_3.append(pdf_array)

    A_geom_cdf = np.exp(log_A_cdf)
    Nb_plates_cdf = A_geom_cdf / (W*L*phi) + 2
    Nb_plates_array_cdf = np.array(Nb_plates_cdf)
    Nb_plates_3_cdf.append(Nb_plates_array_cdf)

    cdf_array = np.array(cdf)
    cdf_array[Nb_plates_array_cdf > xmax] = np.nan
    cdf_array[Nb_plates_array_cdf < xmin] = np.nan
    cdf_3.append(cdf_array)

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_3 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_3)-1): # We exclude the last cycle (TC2R) for which the LOO is not good enough to be confident in the results
    cdf_array_interpolated_3 = np.interp(nb_plates_interpolated, Nb_plates_3_cdf[i], cdf_3[i])
    if np.any(cdf_array_interpolated_3 >= target_CDF_3):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_3 >= target_CDF_3][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_3:.0%} is not reached for the cycle {cycles_involved_3[i]} in the Condenser/Gas Cooler case.")
    min_Nb_plates_target_CDF_3.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_3_unrounded = np.nanmax(min_Nb_plates_target_CDF_3)
min_Nb_plates_target_CDF_3 = np.round(min_Nb_plates_target_CDF_3_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_3:.0%} for the Condenser/GasCooler : {min_Nb_plates_target_CDF_3:.0f} plates")


plt.figure("Condenser/GasCooler - PDF")

for i in range(len(cycles_involved_3)-1):
    plt.plot(Nb_plates_3_pdf[i], pdf_3[i],
            label=cycles_involved_3[i],
            clip_on=False, color=colors_3[i])
    
plt.plot(Nb_plates_3_pdf[-1], pdf_3[-1],
            label=cycles_involved_3[-1],
            clip_on=False, color=colors_3[-1], linestyle='--')
    

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 0.05)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Probability Density Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_3, xmax]
yticks = [0, 0.05]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '0.05'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='upper right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs: plt.savefig("code/RHEIA/rheia/FIGURES/Condenser_GasCooler_PDFs.pdf")

plt.show()


plt.figure("Condenser/GasCooler - CDF")

for i in range(len(cycles_involved_3)-1):
    plt.plot(Nb_plates_3_cdf[i], cdf_3[i],
            label=cycles_involved_3[i],
            clip_on=False, color=colors_3[i], zorder=2)
    
plt.plot(Nb_plates_3_cdf[-1], cdf_3[-1],
            label=cycles_involved_3[-1],
            clip_on=False, color=colors_3[-1], zorder=2, linestyle='--')
    
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

if save_figs : plt.savefig("code/RHEIA/rheia/FIGURES/Condenser_GasCooler_CDFs.pdf")

plt.show()

######################################################
## 4. Recuperator LT
######################################################

# Geometric parameters for the ACK18 heat exchanger
L = 0.315
W = 0.073
phi = 1.2
xmin = 4
xmax = 52

cycles_involved_4 = ["SC1R", "SC2R", "TC2R"]
colors_4 = [colors[cycle] for cycle in cycles_involved_4]
PCE_orders_4 = [4, 3, 2]
objectives_4 = ["log_A_recup", "log_A_recup_LT", "log_A_recup_LT"]
Nb_plates_4_pdf = []
Nb_plates_4_cdf = []
pdf_4 = []
cdf_4 = []

for i in range(len(cycles_involved_4)):
    my_post_process_uq = rheia_pp.PostProcessUQ(cycles_involved_4[i], PCE_orders_4[i])
    log_A_pdf, pdf = my_post_process_uq.get_pdf("results", objectives_4[i])
    log_A_cdf, cdf = my_post_process_uq.get_cdf("results", objectives_4[i])

    A_geom_pdf = np.exp(log_A_pdf)
    Nb_plates_pdf = A_geom_pdf / (W*L*phi) + 2
    Nb_plates_array_pdf = np.array(Nb_plates_pdf)
    Nb_plates_4_pdf.append(Nb_plates_array_pdf)

    pdf_array = np.array(pdf)

    # Normalize the PDF to ensure that the area under the curve is equal to 1
    area = np.trapz(pdf_array, Nb_plates_array_pdf)
    pdf_array /= area
    pdf_array[Nb_plates_array_pdf > xmax] = np.nan
    pdf_array[Nb_plates_array_pdf < xmin] = np.nan
    pdf_4.append(pdf_array)

    A_geom_cdf = np.exp(log_A_cdf)
    Nb_plates_cdf = A_geom_cdf / (W*L*phi) + 2
    Nb_plates_array_cdf = np.array(Nb_plates_cdf)
    Nb_plates_4_cdf.append(Nb_plates_array_cdf)

    cdf_array = np.array(cdf)
    cdf_array[Nb_plates_array_cdf > xmax] = np.nan
    cdf_array[Nb_plates_array_cdf < xmin] = np.nan
    cdf_4.append(cdf_array)

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_4 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_4)):
    cdf_array_interpolated_4 = np.interp(nb_plates_interpolated, Nb_plates_4_cdf[i], cdf_4[i])
    if np.any(cdf_array_interpolated_4 >= target_CDF_4):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_4 >= target_CDF_4][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_4:.0%} is not reached for the cycle {cycles_involved_4[i]} in the Recuperator LT case.")
    min_Nb_plates_target_CDF_4.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_4_unrounded = np.nanmax(min_Nb_plates_target_CDF_4)
min_Nb_plates_target_CDF_4 = np.round(min_Nb_plates_target_CDF_4_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_4:.0%} for the Recuperator LT : {min_Nb_plates_target_CDF_4:.0f} plates")


plt.figure("Recuperator LT - PDF")

for i in range(len(cycles_involved_4)):
    plt.plot(Nb_plates_4_pdf[i], pdf_4[i],
            label=cycles_involved_4[i],
            clip_on=False, color=colors_4[i])

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 0.10)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Probability Density Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_4, xmax]
yticks = [0, 0.10]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '0.10'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='upper right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs: plt.savefig("code/RHEIA/rheia/FIGURES/Recuperator_LT_PDFs.pdf")

plt.show()


plt.figure("Recuperator LT - CDF")

for i in range(len(cycles_involved_4)):
    plt.plot(Nb_plates_4_cdf[i], cdf_4[i],
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

if save_figs : plt.savefig("code/RHEIA/rheia/FIGURES/Recuperator_LT_CDFs.pdf")

plt.show()

######################################################
## 5. Recuperator MT
######################################################

# Geometric parameters for the ACK18 heat exchanger
L = 0.315
W = 0.073
phi = 1.2
xmin = 4
xmax = 52

cycles_involved_5 = ["SC2R", "TC1R", "TC2R"]
colors_5 = [colors[cycle] for cycle in cycles_involved_5]
PCE_orders_5 = [3, 3, 2]
objectives_5 = ["log_A_recup_MT", "log_A_recup", "log_A_recup_MT"]
Nb_plates_5_pdf = []
Nb_plates_5_cdf = []
pdf_5 = []
cdf_5 = []

for i in range(len(cycles_involved_5)):
    my_post_process_uq = rheia_pp.PostProcessUQ(cycles_involved_5[i], PCE_orders_5[i])
    log_A_pdf, pdf = my_post_process_uq.get_pdf("results", objectives_5[i])
    log_A_cdf, cdf = my_post_process_uq.get_cdf("results", objectives_5[i])

    A_geom_pdf = np.exp(log_A_pdf)
    Nb_plates_pdf = A_geom_pdf / (W*L*phi) + 2
    Nb_plates_array_pdf = np.array(Nb_plates_pdf)
    Nb_plates_5_pdf.append(Nb_plates_array_pdf)

    pdf_array = np.array(pdf)

    # Normalize the PDF to ensure that the area under the curve is equal to 1
    area = np.trapz(pdf_array, Nb_plates_array_pdf)
    pdf_array /= area
    pdf_array[Nb_plates_array_pdf > xmax] = np.nan
    pdf_array[Nb_plates_array_pdf < xmin] = np.nan
    pdf_5.append(pdf_array)

    A_geom_cdf = np.exp(log_A_cdf)
    Nb_plates_cdf = A_geom_cdf / (W*L*phi) + 2
    Nb_plates_array_cdf = np.array(Nb_plates_cdf)
    Nb_plates_5_cdf.append(Nb_plates_array_cdf)

    cdf_array = np.array(cdf)
    cdf_array[Nb_plates_array_cdf > xmax] = np.nan
    cdf_array[Nb_plates_array_cdf < xmin] = np.nan
    cdf_5.append(cdf_array)

# Compute the mininum number of plates required to reach the target CDF for each cycle
min_Nb_plates_target_CDF_5 = []
nb_plates_interpolated = np.linspace(xmin, xmax, 1000)

for i in range(len(cycles_involved_5)):
    cdf_array_interpolated_5 = np.interp(nb_plates_interpolated, Nb_plates_5_cdf[i], cdf_5[i])
    if np.any(cdf_array_interpolated_5 >= target_CDF_5):
        min_Nb_plates_target_CDF = nb_plates_interpolated[cdf_array_interpolated_5 >= target_CDF_5][0]
    else:
        min_Nb_plates_target_CDF = np.nan
        print(f"Warning : the target CDF of {target_CDF_5:.0%} is not reached for the cycle {cycles_involved_5[i]} in the Recuperator MT case.")
    min_Nb_plates_target_CDF_5.append(min_Nb_plates_target_CDF)

min_Nb_plates_target_CDF_5_unrounded = np.nanmax(min_Nb_plates_target_CDF_5)
min_Nb_plates_target_CDF_5 = np.round(min_Nb_plates_target_CDF_5_unrounded)
print(f"Minimum number of plates required to reach the target CDF of {target_CDF_5:.0%} for the Recuperator MT : {min_Nb_plates_target_CDF_5:.0f} plates")


plt.figure("Recuperator MT - PDF")

for i in range(len(cycles_involved_5)):
    plt.plot(Nb_plates_5_pdf[i], pdf_5[i],
            label=cycles_involved_5[i],
            clip_on=False, color=colors_5[i])

plt.xlabel('Number of plates [-]', fontsize=12)

plt.xlim(xmin, xmax)
plt.ylim(0, 0.10)

ax = plt.gca()

ax.tick_params(axis='both', which='major')
ax.set_title('Probability Density Function [-]', loc='left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 15))

xticks = [xmin, min_Nb_plates_target_CDF_5, xmax]
yticks = [0, 0.10]

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(['0', '0.10'])

plt.tick_params(axis='x', rotation=0)
plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
plt.legend(loc='upper right', fontsize=11, frameon=False)

plt.tight_layout()

if save_figs: plt.savefig("code/RHEIA/rheia/FIGURES/Recuperator_MT_PDFs.pdf")

plt.show()


plt.figure("Recuperator MT - CDF")

for i in range(len(cycles_involved_5)):
    plt.plot(Nb_plates_5_cdf[i], cdf_5[i],
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

if save_figs : plt.savefig("code/RHEIA/rheia/FIGURES/Recuperator_MT_CDFs.pdf")

plt.show()