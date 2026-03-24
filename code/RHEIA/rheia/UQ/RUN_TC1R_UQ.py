import rheia.UQ.uncertainty_quantification as rheia_uq
import multiprocessing as mp
from post_process import plot_results

run_training = False
run_post_process = True
save_figs = False

# Create the dictionaries with the parameters for the uncertainty quantification process

    # 0. Common parameters for all objectives of interest
COMMON = {'case': 'TC1R',
          'objective names': ['log_A_evap', 'log_A_GasCooler', 'log_A_recup'],
          'results dir': 'results',
          'sampling method': 'SOBOL',
          'create only samples': False,
          'draw pdf cdf': [True, 1e6],
          'n jobs': int(mp.cpu_count()),
          }

    # 1. For the Evaporator
dict_uq_evap = COMMON.copy()
dict_uq_evap['pol order'] = 3                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap['objective of interest'] = 'log_A_evap'

    # 2. For the Gas Cooler
dict_uq_GasCooler = COMMON.copy()
dict_uq_GasCooler['pol order'] = 3                        # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_GasCooler['objective of interest'] = 'log_A_GasCooler'

    # 3. For the Recuperator
dict_uq_recup = COMMON.copy()
dict_uq_recup['pol order'] = 3                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_recup['objective of interest'] = 'log_A_recup'

if __name__ == '__main__':

    # Run the uncertainty quantification process

    if run_training:
        rheia_uq.run_uq(dict_uq_evap)
        rheia_uq.run_uq(dict_uq_GasCooler)
        rheia_uq.run_uq(dict_uq_recup)

    # Post-process the results from the uncertainty quantification

    if run_post_process:
        plot_results(dict_uq_evap, cycle_name="TC1R", save_figs=save_figs)
        plot_results(dict_uq_GasCooler, cycle_name="TC1R", save_figs=save_figs)
        plot_results(dict_uq_recup, cycle_name="TC1R", save_figs=save_figs)


"""
    CONCLUSIONS :

    |------------------|------------------|-----------------------|-------------------|----------------|
    | Order of the PCE | LOO : log_A_evap | LOO : log_A_GasCooler | LOO : log_A_recup | Nb. of samples |
    |------------------|------------------|-----------------------|-------------------|----------------|
    |        1         |      0.10        |         0.74          |       0.12        |       32       |
    |        2         |      0.04        |         0.28          |       0.23        |      272       |
    |        3         |      0.02        |         0.19          |       0.07        |     1632       |
    |------------------|------------------|-----------------------|-------------------|----------------|

"""