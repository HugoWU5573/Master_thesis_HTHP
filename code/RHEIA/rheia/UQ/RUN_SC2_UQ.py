import rheia.UQ.uncertainty_quantification as rheia_uq
import multiprocessing as mp
from post_process import plot_results

run_training = False
run_post_process = True
save_figs = False

# Create the dictionaries with the parameters for the uncertainty quantification process

    # 0. Common parameters for all objectives of interest
COMMON = {'case': 'SC2',
          'objective names': ['log_A_evap_LT', 'log_A_evap_MT', 'log_A_cond'],
          'results dir': 'results',
          'sampling method': 'SOBOL',
          'create only samples': False,
          'draw pdf cdf': [True, 1e6],
          'n jobs': int(mp.cpu_count()),
          }

    # 1. For the Evaporator_LT
dict_uq_evap_LT = COMMON.copy()
dict_uq_evap_LT['pol order'] = 3                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap_LT['objective of interest'] = 'log_A_evap_LT'

    # 2. For the Evaporator_MT
dict_uq_evap_MT = COMMON.copy()
dict_uq_evap_MT['pol order'] = 2                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap_MT['objective of interest'] = 'log_A_evap_MT'

    # 3. For the Condenser
dict_uq_cond = COMMON.copy()
dict_uq_cond['pol order'] = 3                               # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_cond['objective of interest'] = 'log_A_cond'

if __name__ == '__main__':

    # Run the uncertainty quantification process

    if run_training:
        rheia_uq.run_uq(dict_uq_evap_LT)
        rheia_uq.run_uq(dict_uq_evap_MT)
        rheia_uq.run_uq(dict_uq_cond)

    # Post-process the results from the uncertainty quantification

    if run_post_process:
        plot_results(dict_uq_evap_LT, cycle_name="SC2", save_figs=save_figs)
        plot_results(dict_uq_evap_MT, cycle_name="SC2", save_figs=save_figs)
        plot_results(dict_uq_cond, cycle_name="SC2", save_figs=save_figs)


"""
    CONCLUSIONS :

    |------------------|---------------------|---------------------|------------------|----------------|
    | Order of the PCE | LOO : log_A_evap_LT | LOO : log_A_evap_MT | LOO : log_A_cond | Nb. of samples |
    |------------------|---------------------|---------------------|------------------|----------------|
    |        1         |        0.18         |         0.01        |       0.65       |       32       |
    |        2         |        0.10         |        0.0010       |       0.13       |      272       |
    |        3         |        0.09         |        0.0011       |       0.03       |     1632       |
    |------------------|---------------------|---------------------|------------------|----------------|

"""