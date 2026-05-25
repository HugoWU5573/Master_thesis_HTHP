import matplotlib.pyplot as plt
import numpy as np

###############################################################
## PART 1 : Change Q_MT with Q_HT = 25 kW
###############################################################

show_part_1 = False

Q_MT_values = np.array([4, 6, 8, 10, 12])
COP_values = np.array([2.696, 2.843, 2.997, 3.175, 3.365])
N_LP_values = np.array([53.11, 47.44, 42.84, 37.96, 30.44])  # 43% decrease from 4 kW to 12 kW
N_HP_values = np.array([35.56, 36.45, 37.49, 38.36, 39.39])  # 11% increase from 4 kW to 12 kW
z_LP_values = np.array([17.02, 15.25, 13.45, 11.68, 10.19])  # 40% decrease from 4 kW to 12 kW
z_HP_values = np.array([12.73, 16.36, 20.14, 23.59, 27.16])  # 113% increase from 4 kW to 12 kW
bottom_limit = np.ones(len(Q_MT_values)) * 25
upper_limit = np.ones(len(Q_MT_values)) * 70

if show_part_1:

    fig, (ax_N,ax_z, ax_cop) = plt.subplots(
        3, 1, sharex=True,
        gridspec_kw={'height_ratios': [2,2,1], 'hspace': 0.05},
        constrained_layout=True
    )

    ax_N.plot(Q_MT_values, N_LP_values, marker='x', color='red', clip_on=False, label=r"$N_\mathrm{{LP}}$")
    ax_N.text(5, N_LP_values[0]-1, r'$N_\mathrm{{LP}}$', fontsize=12, color="red", rotation=-7, ha='center')
    ax_N.plot(Q_MT_values, N_HP_values, marker='x', color='green', clip_on=False, label=r"$N_\mathrm{{HP}}$")
    ax_N.text(5, N_HP_values[0] + 2.5, r'$N_\mathrm{{HP}}$', fontsize=12, color="green", rotation=1, ha='center')
    ax_N.text(Q_MT_values[-1], 60, r'$\dot{Q}_{\mathrm{HT}} = 25$ kW$_{\mathrm{th}}$', fontsize=12, ha='right')

    ax_N.set_xlim(4, 12)
    ax_N.set_ylim(25, 70)
    ax_N.set_title('Compressor Frequencies [Hz]', loc='left', fontsize=12)
    ax_N.spines['top'].set_visible(False)
    ax_N.spines['right'].set_visible(False)
    ax_N.spines['bottom'].set_visible(False) 
    ax_N.spines['bottom'].set_position(('outward', 20))
    ax_N.spines['left'].set_position(('outward', 15))
    ax_N.set_yticks(np.arange(25, 71, 15))
    ax_N.tick_params(axis='y', labelsize=11, direction='in')
    ax_N.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_z.plot(Q_MT_values, z_LP_values, marker='x', color='blue', clip_on=False, label=r"$z_\mathrm{{LP}}$")
    ax_z.text(11, z_LP_values[-1] + 2, r'$z_\mathrm{{LP}}$', fontsize=12, color="blue", ha='center', rotation=-2)
    ax_z.plot(Q_MT_values, z_HP_values, marker='x', color='orange', clip_on=False, label=r"$z_\mathrm{{HP}}$")
    ax_z.text(11, z_HP_values[-1] -0.5, r'$z_\mathrm{{HP}}$', fontsize=12, color="orange", rotation=6, ha='center')

    ax_z.set_ylim(0, 30)
    ax_z.set_title('Valve Openings [%]', loc='left', fontsize=12)
    ax_z.spines['top'].set_visible(False)
    ax_z.spines['right'].set_visible(False)
    ax_z.spines['bottom'].set_visible(False)
    ax_z.spines['bottom'].set_position(('outward', 20))
    ax_z.spines['left'].set_position(('outward', 15))
    ax_z.set_yticks(np.arange(0, 31, 10))
    ax_z.tick_params(axis='y', labelsize=11, direction='in')
    ax_z.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_cop.plot(Q_MT_values, COP_values,marker='x', color='black', clip_on=False)

    ax_cop.set_ylim(2.5, 3.5)
    ax_cop.set_title('COP [-]', loc='left', fontsize=12)
    ax_cop.set_xlabel(r'$\dot{Q}_{\mathrm{MT}}$  [kW$_{\mathrm{th}}$]', fontsize=12)
    ax_cop.spines['top'].set_visible(False)
    ax_cop.spines['right'].set_visible(False)
    ax_cop.spines['bottom'].set_position(('outward', 20))
    ax_cop.spines['left'].set_position(('outward', 15))
    ax_cop.set_yticks(np.arange(2.5, 3.6, 1))
    ax_cop.set_xticks(np.arange(4, 13, 2))
    ax_cop.tick_params(axis='both', labelsize=11, direction='in')

    plt.savefig(f'code/Figures/Look_Up_Tables/LUT_Q_MT.pdf')
    plt.show()


###############################################################
## PART 2 : Change Q_HT with Q_MT = 10 kW
###############################################################

show_part_2 = True

Q_HT_values = np.array([21, 23, 25, 27,31])
COP_values = np.array([3.400, 3.278, 3.175, 3.087, 2.956])
N_LP_values = np.array([26.61, 32.69, 37.96,39.70, 47.26])
N_HP_values = np.array([33.46, 35.92, 38.36,40.86, 45.77])
z_LP_values = np.array([9.54, 10.59, 11.68,13.01, 15.52])
z_HP_values = np.array([23.19, 23.47, 23.59,23.71, 24.02])

if show_part_2:

    fig, (ax_N,ax_z, ax_cop) = plt.subplots(
        3, 1, sharex=True,
        gridspec_kw={'height_ratios': [2,2,1], 'hspace': 0.05},
        constrained_layout=True
    )

    ax_N.plot(Q_HT_values, N_LP_values, marker='x', color='red', clip_on=False, label=r"$N_\mathrm{{LP}}$")
    ax_N.text(26, N_LP_values[0]+5, r'$N_\mathrm{{LP}}$', fontsize=12, color="red", rotation=1, ha='center')
    ax_N.plot(Q_HT_values, N_HP_values, marker='x', color='green', clip_on=False, label=r"$N_\mathrm{{HP}}$")
    ax_N.text(22, N_HP_values[0] + 3.5, r'$N_\mathrm{{HP}}$', fontsize=12, color="green", rotation=1, ha='center')
    ax_N.text(Q_HT_values[-1], 60, r'$\dot{Q}_{\mathrm{MT}} = 10$ kW$_{\mathrm{th}}$', fontsize=12, ha='right')

    ax_N.set_xlim(21, 31)
    ax_N.set_ylim(25, 70)
    ax_N.set_title('Compressor Frequencies [Hz]', loc='left', fontsize=12)
    ax_N.spines['top'].set_visible(False)
    ax_N.spines['right'].set_visible(False)
    ax_N.spines['bottom'].set_visible(False) 
    ax_N.spines['bottom'].set_position(('outward', 20))
    ax_N.spines['left'].set_position(('outward', 15))
    ax_N.set_yticks(np.arange(25, 71, 15))
    ax_N.tick_params(axis='y', labelsize=11, direction='in')
    ax_N.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_z.plot(Q_HT_values, z_LP_values, marker='x', color='blue', clip_on=False, label=r"$z_\mathrm{{LP}}$")
    ax_z.text(29, z_LP_values[-1]+0.5, r'$z_\mathrm{{LP}}$', fontsize=12, color="blue", ha='center', rotation=2)
    ax_z.plot(Q_HT_values, z_HP_values, marker='x', color='orange', clip_on=False, label=r"$z_\mathrm{{HP}}$")
    ax_z.text(29, z_HP_values[-1]+1.5, r'$z_\mathrm{{HP}}$', fontsize=12, color="orange", ha='center')

    ax_z.set_ylim(0, 30)
    ax_z.set_title('Valve Openings [%]', loc='left', fontsize=12)
    ax_z.spines['top'].set_visible(False)
    ax_z.spines['right'].set_visible(False)
    ax_z.spines['bottom'].set_visible(False)
    ax_z.spines['bottom'].set_position(('outward', 20))
    ax_z.spines['left'].set_position(('outward', 15))
    ax_z.set_yticks(np.arange(0, 31, 10))
    ax_z.tick_params(axis='y', labelsize=11, direction='in')
    ax_z.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_cop.plot(Q_HT_values, COP_values,marker='x', color='black', clip_on=False)

    ax_cop.set_ylim(2.5, 3.5)
    ax_cop.set_title('COP [-]', loc='left', fontsize=12)
    ax_cop.set_xlabel(r'$\dot{Q}_{\mathrm{HT}}$  [kW$_{\mathrm{th}}$]', fontsize=12)
    ax_cop.spines['top'].set_visible(False)
    ax_cop.spines['right'].set_visible(False)
    ax_cop.spines['bottom'].set_position(('outward', 20))
    ax_cop.spines['left'].set_position(('outward', 15))
    ax_cop.set_yticks(np.arange(2.5, 3.6, 1))
    ax_cop.set_xticks(Q_HT_values)
    ax_cop.tick_params(axis='both', labelsize=11, direction='in')

    plt.savefig(f'code/Figures/Look_Up_Tables/LUT_Q_HT.pdf')
    plt.show()