import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys
import matplotlib.pyplot as plt

code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from cycles.SC1 import run_SC1_cycle


"""
    This function generates N input parameters sets for the cycle.

"""
def generate_input_parameters(N, seed=42):
    np.random.seed(seed)
    input_parameters = {}

    # Create the Gaussian variables
    gaussian_vars = [
        "h_single_phase", "h_evaporation", "h_condensation",
        "f_single_phase", "f_evaporation", "f_condensation"
    ]
    for var in gaussian_vars:
        mu = 1.0
        sigma = 0.15
        input_parameters[var] = np.random.normal(mu, sigma, N)

    # Create the uniform variables
    uniform_vars_mu = {
        "beta": 45,
        "phi": 1.2,
        "gamma": 0.55,
        "eta_is": 0.65,
        "Q": 25000,
        "T1_prime": 288.75,
        "T4_prime": 313.15,
    }

    uniform_vars_delta = {
        "beta": 15,
        "phi": 0.1,
        "gamma": 0.1,
        "eta_is": 0.1,
        "Q": 5000,
        "T1_prime": 10,
        "T4_prime": 10,
    }

    for var, delta in uniform_vars_delta.items():
        mu = uniform_vars_mu[var]
        input_parameters[var] = np.random.uniform(mu - delta, mu + delta, N)

    return pd.DataFrame(input_parameters)


"""
    This function runs the cycle for one set of input parameters and returns the results.

"""
def run_sample(row):

    print(f"Running sample number {row.name + 1}")  # Print progress

    sample = row.to_dict()

    try:
        log_A_evap, log_A_cond = run_SC1_cycle(sample)
    except Exception as e:
        print(f"Simulation failed: {e}")
        log_A_evap, log_A_cond = np.nan, np.nan
    
    return log_A_evap, log_A_cond

"""
    This function manages the parallel execution of the calls to run_sample.

"""
def run_parallel(df, n_workers=None):

    if n_workers=="normal":
        n_workers = int(cpu_count() * 0.75)  # We use 75% of available cores by default
    elif n_workers=="max":
        n_workers = int(cpu_count())  # We use all available cores
    with Pool(n_workers) as pool:
        results = pool.map(run_sample, [row for _, row in df.iterrows()])

    return results

"""
    This function performs the Monte Carlo analysis by generating input samples, running the simulations in parallel and saving the 
    results in a CSV file.

"""
def generate_samples(cycle_name, n_samples, seed=42, n_workers="normal"):

    # Make the path relative to the current file
    current_file_dir = Path(__file__).parent
    data_dir = current_file_dir / "data"
    data_dir.mkdir(exist_ok=True)
    output_csv = data_dir / cycle_name / "samples.csv"

    output_path = Path(output_csv)

    # If the csv file already exists, we load it to check how many samples it contains.
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        n_done = len(df_existing)
        print(f"Found existing {n_done} simulations. Reusing them.")
    else:
        df_existing = pd.DataFrame()
        n_done = 0

    if n_samples <= n_done:
        print(f"Already have {n_done} samples >= requested {n_samples}. Nothing to do.")
        return None

    n_new = n_samples - n_done
    print(f"Generating {n_new} new samples.")

    # We generate N new input parameters sets, with a different seed to ensure different samples from the previous ones.
    df_new = generate_input_parameters(n_new, seed=seed + n_done)

    # We run the simulations in parallel and get the results.
    results = run_parallel(df_new, n_workers=n_workers)

    df_new["log_A_evap"], df_new["log_A_cond"] = zip(*results)
    df_new.dropna(subset=["log_A_evap", "log_A_cond"], inplace=True)

    # We concatenate the new results with the existing ones (if any) and save everything in the csv file.
    if n_done > 0:
        df_full = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_full = df_new

    # We save the results in a CSV file.
    df_full.to_csv(output_csv, index=False)
    print(f"Saved {len(df_full)} samples to {output_csv}")

    return None

"""
    This function plots the empirical CDFs of the results.

"""
def plot_cdf(Nb_plates, samples_sizes, colors, xmin, xmax, cycle_name, name, target_CDF=0.85, save_data=False, save_figure=False):

    plt.figure(name)

    for i, size in enumerate(samples_sizes):

        data = Nb_plates[:size]
        data = data[(data >= xmin) & (data <= xmax)]

        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

        if save_data and i == len(samples_sizes) - 1:
            df_plot = pd.DataFrame({"Nb_plates": sorted_data, "CDF": cdf})
            csv_file = Path(__file__).parent / "data" / cycle_name / f"{name}_data.csv"
            df_plot.to_csv(csv_file, index=False)

        if i == len(samples_sizes) - 1:
            min_nb_plates_target = np.round(np.percentile(data, 85))

        plt.plot(sorted_data, cdf, label=f"{size} samples", color=colors[i], clip_on=False, zorder=2)

    plt.axhline(y=target_CDF, color='k', linestyle=':', label=f'Target : {target_CDF:.0%}', zorder=1)

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

    xticks = [xmin, min_nb_plates_target, xmax]
    yticks = [0, 1]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['0', '1'])

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(loc='lower right', fontsize=11, frameon=False)
    plt.tight_layout()
    if save_figure: plt.savefig(Path(__file__).parent / "figures" / cycle_name / f"{name}.pdf")
    plt.show()

    return min_nb_plates_target

"""
    This function plots the empirical PDFs of the results.

"""
def plot_pdf(Nb_plates, samples_sizes, colors, xmin, xmax, cycle_name, name, min_nb_plates_target, save_figure=False) :

    plt.figure(name)

    for i, size in enumerate(samples_sizes):

        data = Nb_plates[:size]
        data = data[(data >= xmin) & (data <= xmax)]

        # Histogram
        plt.hist(data, bins=100, density=True,alpha=0.5, color=colors[i],label=f"{size} samples")

    plt.xlabel('Number of plates [-]', fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(0, 0.15)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('Probability Density Function [-]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))

    xticks = [xmin, min_nb_plates_target, xmax]
    yticks = [0, 0.15]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['0', '0.15'])

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(loc='upper right', fontsize=11, frameon=False)
    plt.tight_layout()
    if save_figure: plt.savefig(Path(__file__).parent / "figures" / cycle_name / f"{name}.pdf")
    plt.show()


if __name__ == "__main__":

    save_data = False
    save_figures = False
    cycle_name = "SC1"

    get_samples = False
    post_process = True

    if get_samples:

        ## STEP 1 : Generate the samples and save the results in a CSV file

        N = 5000   # Number of samples to generate
        generate_samples(cycle_name, N)
    
    if post_process:

        ## STEP 2 : Analyse the results

        # Load the results from the CSV file
        csv_file = Path(__file__).parent / "data" / cycle_name / "samples.csv"
        df = pd.read_csv(csv_file)
        log_A_evap = df["log_A_evap"].values
        log_A_cond = df["log_A_cond"].values
        phi = df["phi"].values
        samples_sizes = [100, 1000, 5000]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        # Function to convert log_A to number of plates
        def logA_to_Nplates(log_A, phi, model):

            if model=="ACP70X":
                L = 0.526 
                W = 0.111
            elif model=="ACK18":
                L = 0.315
                W = 0.073

            A = np.exp(log_A)
            N = A / (W*L*phi) + 2
            return N
        
        # Compute the number of plates for each sample and for each HEX 
        Nb_plates_evap = np.zeros(len(log_A_evap))
        Nb_plates_cond = np.zeros(len(log_A_cond))

        for i in range(len(log_A_evap)):
            Nb_plates_evap[i] = logA_to_Nplates(log_A_evap[i], phi[i], model="ACP70X")
            Nb_plates_cond[i] = logA_to_Nplates(log_A_cond[i], phi[i], model="ACP70X")

        mean_nb_plates_evap = np.zeros(len(samples_sizes))
        std_nb_plates_evap = np.zeros(len(samples_sizes))
        mean_nb_plates_cond = np.zeros(len(samples_sizes))
        std_nb_plates_cond = np.zeros(len(samples_sizes))

        for i, size in enumerate(samples_sizes):
            mean_nb_plates_evap[i] = np.mean(Nb_plates_evap[:size])
            std_nb_plates_evap[i] = np.std(Nb_plates_evap[:size])
            mean_nb_plates_cond[i] = np.mean(Nb_plates_cond[:size])
            std_nb_plates_cond[i] = np.std(Nb_plates_cond[:size])

        # Print a table with the different statistics depending on the sample size
        print("-" * 128)
        print(f"| {'Sample Size':<12} | {'Mean Nb plates Evap':<25} | {'Std Nb plates Evap':<25} | {'Mean Nb plates Cond':<25} | {'Std Nb plates Cond':<25} |")
        print("-" * 128)
        for i, size in enumerate(samples_sizes):
            print(f"| {size:<12} | {mean_nb_plates_evap[i]:<25.2f} | {std_nb_plates_evap[i]:<25.2f} | {mean_nb_plates_cond[i]:<25.2f} | {std_nb_plates_cond[i]:<25.2f} |")
        print("-" * 128)

        ### EVAPORATOR LT
        xmin = 4 ; xmax = 124

        # Plot the empirical CDFs for the evaporator
        name = "Evap_LT_CDF"
        Nb_plates = Nb_plates_evap
        min_nb_plates_evap_LT = plot_cdf(Nb_plates, samples_sizes, colors, xmin, xmax, cycle_name, name, save_data=save_data, save_figure=save_figures)

        # Plot the histograms of the PDfs for the evaporator
        name = "Evap_LT_PDF"
        plot_pdf(Nb_plates, samples_sizes, colors, xmin, xmax, cycle_name, name, min_nb_plates_evap_LT, save_figure=save_figures)

        ### CONDENSER
        xmin = 4 ; xmax = 124

        # Plot the empirical CDFs for the condenser
        name = "Condenser_CDF"
        Nb_plates = Nb_plates_cond
        min_nb_plates_cond = plot_cdf(Nb_plates, samples_sizes, colors, xmin, xmax,cycle_name, name, save_data=save_data, save_figure=save_figures)

        # Plot the histograms of the PDfs for the condenser
        name = "Condenser_PDF"
        plot_pdf(Nb_plates, samples_sizes, colors, xmin, xmax, cycle_name, name, min_nb_plates_cond, save_figure=save_figures)