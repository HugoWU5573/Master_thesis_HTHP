import numpy as np 
from matplotlib import pyplot as plt

freq = np.arange(25, 61, 5)
data = np.zeros((len(freq), 10))

for i,f in enumerate(freq):
    data_str = np.loadtxt(f'code/fitting/compressor/mass_flow_poly/{f}Hz.csv', delimiter=';', skiprows = 30, max_rows = 1, usecols = np.arange(1, 11), dtype = str)
    for j in range(len(data_str)):
        data_str[j] = data_str[j].replace(',', '.')
    data[i] = data_str.astype(float)

'''
for i in range(10) : 
    plt.figure()
    plt.plot(freq, data[:, i])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(f'coefficient {i+1}')
plt.show()
'''

plt.figure()
for i in range(10) :
    coeff_max = np.max(abs(data[:, i]))
    plt.plot(freq, data[:, i]/coeff_max, label = f'coefficient {i+1}')
    
plt.title('Normalized coefficients of the polynomial fit for mass flow rate')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized coefficient')
plt.legend()
plt.show()