import rheia.UQ.uncertainty_quantification as rheia_uq
import multiprocessing as mp
from post_process import plot_results

run_training = False
run_post_process = True
save_figs = False

# Create the dictionaries with the parameters for the uncertainty quantification process

    # 0. Common parameters for all objectives of interest
COMMON = {'case': 'TC1',
          'objective names': ['log_A_evap', 'log_A_GasCooler'],
          'results dir': 'results',
          'sampling method': 'SOBOL',
          'create only samples': False,
          'draw pdf cdf': [True, 1e6],
          'n jobs': int(mp.cpu_count()),
          }

    # 1. For the Evaporator
dict_uq_evap = COMMON.copy()
dict_uq_evap['pol order'] = 2                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap['objective of interest'] = 'log_A_evap'

    # 2. For the Gas Cooler
dict_uq_GasCooler = COMMON.copy()
dict_uq_GasCooler['pol order'] = 1                       # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_GasCooler['objective of interest'] = 'log_A_GasCooler'

if __name__ == '__main__':

    # Run the uncertainty quantification process

    if run_training:
        rheia_uq.run_uq(dict_uq_evap)
        rheia_uq.run_uq(dict_uq_GasCooler)

    # Post-process the results from the uncertainty quantification

    if run_post_process:
        plot_results(dict_uq_evap, cycle_name="TC1", save_figs=save_figs)
        plot_results(dict_uq_GasCooler, cycle_name="TC1", save_figs=save_figs)


"""
    CONCLUSIONS :

    |------------------|------------------|-----------------------|----------------|
    | Order of the PCE | LOO : log_A_evap | LOO : log_A_GasCooler | Nb. of samples |
    |------------------|------------------|-----------------------|----------------|
    |        1         |      0.005       |         0.08          |       32       |
    |        2         |      0.0003      |         0.09          |      272       |
    |        3         |      0.0004      |         0.09          |     1632       |
    |------------------|------------------|-----------------------|----------------|

"""