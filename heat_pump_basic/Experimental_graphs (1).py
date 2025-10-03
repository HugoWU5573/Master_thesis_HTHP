import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

fluid = 'R410A'

# Définir les cycles avec 4 vrais états thermodynamiques
cycles = {
    '10°C': {
        'color': 'tab:blue',
        'points': {
            1: {'T': 7.7 + 273.15, 'P': 10.7e5},
            2: {'T': 65.2 + 273.15, 'P': 24.2e5},
            3: {'T': 39.9 + 273.15, 'P': 24.2e5},
            4: {'T': 35.5 + 273.15, 'P': 10.7e5},
        }
    },
    '5°C': {
        'color': 'tab:orange',
        'points': {
            1: {'T': 6.1 + 273.15, 'P': 9.3e5},
            2: {'T': 90.0 + 273.15, 'P': 33.0e5},
            3: {'T': 53.5 + 273.15, 'P': 33.0e5},
            4: {'T': 47.2 + 273.15, 'P': 9.3e5},
        }
    },
    '0°C': {
        'color': 'tab:green',
        'points': {
            1: {'T': 5.5 + 273.15, 'P': 9.0e5},
            2: {'T': 100.7 + 273.15, 'P': 37.1e5},
            3: {'T': 59.1 + 273.15, 'P': 37.1e5},
            4: {'T': 51.7 + 273.15, 'P': 9.0e5},
        }
    }
}

# Calcul des entropies
for cycle in cycles.values():
    for i in range(1, 5):
        T = cycle['points'][i]['T']
        P = cycle['points'][i]['P']
        try:
            S = PropsSI('S', 'T', T, 'P', P, fluid) / 1000  # kJ/kg·K
            cycle['points'][i]['S'] = S
        except:
            cycle['points'][i]['S'] = None

# Courbe de saturation
s_liq, s_vap, T_range = [], [], []
for T_C in range(-60, 65):
    T_K = T_C + 273.15
    try:
        s_l = PropsSI('S', 'T', T_K, 'Q', 0, fluid) / 1000
        s_v = PropsSI('S', 'T', T_K, 'Q', 1, fluid) / 1000
        s_liq.append(s_l)
        s_vap.append(s_v)
        T_range.append(T_K)
    except:
        continue

# Tracé
plt.figure(figsize=(10,6))
plt.plot(s_liq, T_range, 'gray', linestyle='--', label='Saturation curve')
plt.plot(s_vap, T_range, 'gray', linestyle='--')

for label, data in cycles.items():
    pts = data['points']
    s_vals = [pts[i]['S'] for i in [1, 2, 3, 4, 1]]
    T_vals = [pts[i]['T'] for i in [1, 2, 3, 4, 1]]
    plt.plot(s_vals, T_vals, marker='o', color=data['color'], label=f'T_ext = {label}')
    for i in range(1, 5):
        plt.text(pts[i]['S'], pts[i]['T'], f'{i}', fontsize=10, ha='center', va='bottom')

plt.xlabel('Entropy [kJ/kg·K]')
plt.ylabel('Temperature [K]')
plt.title('Experimental T-s Diagram – R410A (4-point cycle)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
