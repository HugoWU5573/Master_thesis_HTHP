import matplotlib.pyplot as plt
import math
import numpy as np

def number_of_samples(p, n):
    """
    Compute the number of samples required for a given order of the PCE and number of uncertain parameters.

    Parameters :
        - p : int --> order of the PCE
        - n : int --> number of uncertain parameters

    Returns :
        - N : int --> number of samples required
    
    """

    N = 2 * math.factorial(p + n) / (math.factorial(p) * math.factorial(n))
    
    return N


if __name__ == '__main__':

    save_fig = True

    labels = np.array([r"$SC1(R)$", r"$SC2(R)$ $or$ $TC1(R)$", r"$TC2(R)$"])
    p = np.array([13, 15, 17])
    n = np.array([1, 2, 3, 4])

    N = np.zeros((len(labels), len(n)))
    for i in range(len(labels)):
        for j in range(len(n)):
            N[i, j] = number_of_samples(p[i], n[j])

    plt.figure()

    for i in range(len(labels)):
        plt.plot(n, N[i, :], label=labels[i], marker=".", clip_on=False)
    
    plt.yscale('log')

    plt.xlabel('PCE order [-]', fontsize=12)

    ymax = np.max(N) ; ymin = np.min(N)

    plt.xlim(1, 4)
    plt.ylim(ymin, ymax)

    xticks = np.array([1, 4])
    yticks = np.round([10, 100, 1000, 10000])

    ax = plt.gca()

    ax.tick_params(axis='both', which='major')
    ax.set_title('Number of samples [-]', loc='left',fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

    plt.legend(loc='lower right', fontsize=11, frameon=False)
    plt.tight_layout()

    if save_fig: 
        plt.savefig('code/RHEIA/rheia/FIGURES/Number_of_samples.pdf')

    plt.show()