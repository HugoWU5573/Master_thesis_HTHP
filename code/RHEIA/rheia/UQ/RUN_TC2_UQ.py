import rheia.UQ.uncertainty_quantification as rheia_uq
import multiprocessing as mp
from post_process import plot_results

run_training = True
run_post_process = True
save_figs = False

# Create the dictionaries with the parameters for the uncertainty quantification process

    # 0. Common parameters for all objectives of interest
COMMON = {'case': 'TC2',
          'objective names': ['log_A_evap_LT', 'log_A_evap_MT', 'log_A_GasCooler'],
          'results dir': 'results',
          'sampling method': 'SOBOL',
          'create only samples': False,
          'draw pdf cdf': [True, 1e6],
          'n jobs': int(mp.cpu_count()),
          }

    # 1. For the Evaporator LT
dict_uq_evap_LT = COMMON.copy()
dict_uq_evap_LT['pol order'] = 1                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap_LT['objective of interest'] = 'log_A_evap_LT'

    # 2. For the Evaporator MT
dict_uq_evap_MT = COMMON.copy()
dict_uq_evap_MT['pol order'] = 1                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap_MT['objective of interest'] = 'log_A_evap_MT'

    # 3. For the Gas Cooler
dict_uq_GasCooler = COMMON.copy()
dict_uq_GasCooler['pol order'] = 1                              # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_GasCooler['objective of interest'] = 'log_A_GasCooler'

if __name__ == '__main__':

    # Run the uncertainty quantification process

    if run_training:
        rheia_uq.run_uq(dict_uq_evap_LT)
        rheia_uq.run_uq(dict_uq_evap_MT)
        rheia_uq.run_uq(dict_uq_GasCooler)

    # Post-process the results from the uncertainty quantification

    if run_post_process:
        plot_results(dict_uq_evap_LT, cycle_name="TC2", save_figs=save_figs)
        plot_results(dict_uq_evap_MT, cycle_name="TC2", save_figs=save_figs)
        plot_results(dict_uq_GasCooler, cycle_name="TC2", save_figs=save_figs)


"""
    CONCLUSIONS :

    |------------------|---------------------|---------------------|-----------------------|----------------|
    | Order of the PCE | LOO : log_A_evap_LT | LOO : log_A_evap_MT | LOO : log_A_GasCooler | Nb. of samples |
    |------------------|---------------------|---------------------|-----------------------|----------------|
    |        1         |        ????         |        ????         |          ????         |       ??       |
    |        2         |        ????         |        ????         |          ????         |      ???       |
    |        3         |        ????         |        ????         |          ????         |     ????       |
    |------------------|---------------------|---------------------|-----------------------|----------------|

"""