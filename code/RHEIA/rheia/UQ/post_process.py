import rheia.POST_PROCESS.post_process as rheia_pp
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(dict_uq, save_figs=False, cycle_name=None):
    case = dict_uq.get('case')
    pol_order = dict_uq.get('pol order')
    result_dir = dict_uq.get('results dir')
    objective = dict_uq.get('objective of interest')
    my_post_process_uq = rheia_pp.PostProcessUQ(case, pol_order)

    ###################################
    ### 1. LOO error
    ###################################

    print("The LOO error for objective ", objective, " is: ", np.round(my_post_process_uq.get_loo(result_dir, objective), 4))
    
    ###################################
    ### 2. PDF
    ###################################
    log_A, pdf = my_post_process_uq.get_pdf(result_dir, objective)
    A_geom = np.exp(log_A)

    if "recup" in objective:
        L = 0.315
        W = 0.073
        phi = 1.2
        xmin = 4
        xmax = 52
    else :
        L = 0.526 
        W = 0.111
        phi = 1.2
        xmin = 4
        xmax = 124

    Nb_plates = A_geom / (W*L*phi) + 2

    max_pdf_index = np.argmax(pdf)
    most_probable_Nb_plates = np.round(Nb_plates[max_pdf_index])

    if "evap" in objective:
        objective_name = "Evaporator_UQ"
        if "LT" in objective:
            objective_name += "_LT"
        elif "MT" in objective:
            objective_name += "_MT"
    elif "cond" in objective:
        objective_name = "Condenser_UQ"
    elif "recup" in objective:
        objective_name = "Recuperator_UQ"
        if "LT" in objective:
            objective_name += "_LT"
        elif "MT" in objective:
            objective_name += "_MT"
    elif "Gas" in objective:
        objective_name = "GasCooler_UQ"

        # Convert to numpy arrays
    Nb_plates_array = np.array(Nb_plates)
    pdf_array = np.array(pdf)
    pdf_array[Nb_plates_array > xmax] = np.nan

        # X ticks
    xticks = np.round([xmin, most_probable_Nb_plates, xmax])

        # Y ticks
    ymax = np.ceil(np.nanmax(pdf_array))
    yticks = np.round([0, ymax])

    plt.figure(objective_name)

    plt.plot(Nb_plates_array, pdf_array,
             color="black",
             clip_on=False)

    plt.xlabel('Number of plates [-]', fontsize=12)

    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)

    ax = plt.gca()

    ax.tick_params(axis='both', which='major')
    ax.set_title('Probability Density Function [-]',
                 loc='left',
                 fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['0', ''])

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both',
                    which='major',
                    labelsize=11,
                    direction='in')

    plt.tight_layout()

    if cycle_name is not None and save_figs:
        fig_dir = f'code/Figures/{cycle_name}'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}/{objective_name + "_PDF"}.pdf')

    plt.show()


    ###################################
    ### 3. Sobol indices
    ###################################
    names, sobol = my_post_process_uq.get_sobol(result_dir, objective)

    plt.figure(objective_name)

    plt.barh(names, sobol,
            color="black",
            edgecolor="black")

    plt.xlabel('Sobol Index [-]', fontsize=12)

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 0))

    ax.tick_params(axis='both',
                which='major',
                labelsize=11,
                direction='in')

    xmin = 0
    xmax = np.ceil(np.nanmax(sobol))
    xticks = [xmin, xmax]
    ax.set_xticks(xticks)
    ax.set_yticks(range(len(names)))

    ax.set_xticklabels([f"{xmin}", f"{xmax:.0f}"])

    ax.set_title('Sobol Sensitivity Indices [-]',
                loc='left',
                fontsize=12)

    plt.tight_layout()

    if cycle_name is not None and save_figs:
        fig_dir = f'code/Figures/{cycle_name}'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}/{objective_name + "_Sobol"}.pdf')

    plt.show()
