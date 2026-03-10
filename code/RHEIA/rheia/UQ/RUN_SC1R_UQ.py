import rheia.UQ.uncertainty_quantification as rheia_uq
import multiprocessing as mp
from post_process import plot_results

run_training = True
run_post_process = True
save_figs = False

# Create the dictionaries with the parameters for the uncertainty quantification process

    # 0. Common parameters for all objectives of interest
COMMON = {'case': 'SC1R',
          'objective names': ['log_A_evap', 'log_A_cond', 'log_A_recup'],
          'results dir': 'results',
          'sampling method': 'SOBOL',
          'create only samples': False,
          'draw pdf cdf': [True, 1e6],
          'n jobs': int(mp.cpu_count()),
          }

    # 1. For the Evaporator
dict_uq_evap = COMMON.copy()
dict_uq_evap['pol order'] = 1                           # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap['objective of interest'] = 'log_A_evap'

    # 2. For the Condenser
dict_uq_cond = COMMON.copy()
dict_uq_cond['pol order'] = 1                           # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_cond['objective of interest'] = 'log_A_cond'

    # 3. For the Recuperator
dict_uq_recup = COMMON.copy()
dict_uq_recup['pol order'] = 1                           # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_recup['objective of interest'] = 'log_A_recup'

if __name__ == '__main__':

    # Run the uncertainty quantification process

    if run_training:
        rheia_uq.run_uq(dict_uq_evap)
        rheia_uq.run_uq(dict_uq_cond)
        rheia_uq.run_uq(dict_uq_recup)

    # Post-process the results from the uncertainty quantification

    if run_post_process:
        plot_results(dict_uq_evap, cycle_name="SC1R", save_figs=save_figs)
        plot_results(dict_uq_cond, cycle_name="SC1R", save_figs=save_figs)
        plot_results(dict_uq_recup, cycle_name="SC1R", save_figs=save_figs)


"""
    CONCLUSIONS :

    |------------------|------------------|------------------|-------------------|
    | Order of the PCE | LOO : log_A_evap | LOO : log_A_cond | LOO : log_A_recup |
    |------------------|------------------|------------------|-------------------|
    |        1         |      ????        |       ????       |        ????       |
    |        2         |      ????        |       ????       |        ????       |
    |        3         |      ????        |       ????       |        ????       | 
    |------------------|------------------|------------------|-------------------|

"""