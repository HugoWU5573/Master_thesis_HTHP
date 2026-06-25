import sys
from pathlib import Path
import numpy as np

code_dir = Path(__file__).parent.parent.parent.parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from code.cycles.thermodynamic.SC1R import run_SC1R_cycle


def set_params():

    return []

def evaluate(x_in, params = []):

    idx, sample = x_in
    print("Evaluating sample ", idx)

    try :
        log_A_evap, log_A_cond, log_A_recup = run_SC1R_cycle(sample, verbose=False, print_results=False, plot_results=False, save_results=False)
        
        return float(log_A_evap), float(log_A_cond), float(log_A_recup)

    except Exception as e:
        print("The exception is: ", e)
        print("Error in the cycle calculation for sample: ", sample)

        # We return a nan for the HEX area if an error occurs during the cycle calculation.
        return np.nan, np.nan, np.nan