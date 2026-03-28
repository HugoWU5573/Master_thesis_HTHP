import rheia.UQ.uncertainty_quantification as rheia_uq
import multiprocessing as mp
from post_process import plot_results

run_training = True
run_post_process = True
save_figs = True

# Create the dictionaries with the parameters for the uncertainty quantification process

    # 0. Common parameters for all objectives of interest
COMMON = {'case': 'TC2R',
          'objective names': ['log_A_evap_LT', 'log_A_evap_MT', 'log_A_GasCooler', 'log_A_recup_LT', 'log_A_recup_MT'],
          'results dir': 'results',
          'sampling method': 'SOBOL',
          'create only samples': False,
          'draw pdf cdf': [True, 1e6],
          'n jobs': int(mp.cpu_count()),
          }

    # 1. For the Evaporator LT
dict_uq_evap_LT = COMMON.copy()
dict_uq_evap_LT['pol order'] = 2                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap_LT['objective of interest'] = 'log_A_evap_LT'

    # 2. For the Evaporator MT
dict_uq_evap_MT = COMMON.copy()
dict_uq_evap_MT['pol order'] = 2                            # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_evap_MT['objective of interest'] = 'log_A_evap_MT'

    # 3. For the Gas Cooler
dict_uq_GasCooler = COMMON.copy()
dict_uq_GasCooler['pol order'] = 2                          # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_GasCooler['objective of interest'] = 'log_A_GasCooler'

    # 4. For the Recuperator LT
dict_uq_recup_LT = COMMON.copy()
dict_uq_recup_LT['pol order'] = 2                           # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_recup_LT['objective of interest'] = 'log_A_recup_LT'

    # 5. For the Recuperator MT
dict_uq_recup_MT = COMMON.copy()
dict_uq_recup_MT['pol order'] = 2                           # See the conclusions at the end of this file for the choice of the order of the PCE.
dict_uq_recup_MT['objective of interest'] = 'log_A_recup_MT'

if __name__ == '__main__':

    # Run the uncertainty quantification process

    if run_training:
        rheia_uq.run_uq(dict_uq_evap_LT)
        rheia_uq.run_uq(dict_uq_evap_MT)
        rheia_uq.run_uq(dict_uq_GasCooler)
        rheia_uq.run_uq(dict_uq_recup_LT)
        rheia_uq.run_uq(dict_uq_recup_MT)

    # Post-process the results from the uncertainty quantification

    if run_post_process:
        plot_results(dict_uq_evap_LT, cycle_name="TC2R", save_figs=save_figs)
        plot_results(dict_uq_evap_MT, cycle_name="TC2R", save_figs=save_figs)
        plot_results(dict_uq_GasCooler, cycle_name="TC2R", save_figs=save_figs)
        plot_results(dict_uq_recup_LT, cycle_name="TC2R", save_figs=save_figs)
        plot_results(dict_uq_recup_MT, cycle_name="TC2R", save_figs=save_figs)


"""
    CONCLUSIONS : With the old samples (max nb. plates set to 1000)

    |------------------|---------------------|---------------------|-----------------------|----------------------|----------------------|----------------|
    | Order of the PCE | LOO : log_A_evap_LT | LOO : log_A_evap_MT | LOO : log_A_GasCooler | LOO : log_A_recup_LT | LOO : log_A_recup_MT | Nb. of samples |
    |------------------|---------------------|---------------------|-----------------------|----------------------|----------------------|----------------|
    |        1         |        0.04         |        0.82         |          28.0         |         1.17         |         7.37         |       36       |
    |        2         |        0.30         |        0.004        |          0.95         |         0.06         |         0.29         |      342       |
    |        3         |        1.07         |        0.02         |          1.25         |         0.12         |         0.38         |     2280       |
    |        4         |        0.88         |        0.01         |          2.78         |         0.08         |         0.37         |    11970       |
    |------------------|---------------------|---------------------|-----------------------|----------------------|----------------------|----------------|

    CONCLUSIONS : With the new samples (max nb. plates set to 150)

    |------------------|---------------------|---------------------|-----------------------|----------------------|----------------------|----------------|
    | Order of the PCE | LOO : log_A_evap_LT | LOO : log_A_evap_MT | LOO : log_A_GasCooler | LOO : log_A_recup_LT | LOO : log_A_recup_MT | Nb. of samples |
    |------------------|---------------------|---------------------|-----------------------|----------------------|----------------------|----------------|
    |        1         |        0.57         |        0.03         |          0.85         |         0.09         |         0.34         |       36       |
    |        2         |        0.21         |        0.003        |          0.50         |         0.04         |         0.19         |      342       |
    |        3         |        1.13         |        0.02         |          1.40         |         0.11         |         0.40         |     2280       |
    |------------------|---------------------|---------------------|-----------------------|----------------------|----------------------|----------------|


"""