import pandas as pd

# Charger les données
df = pd.read_csv('test_24_03_24.csv', sep=';', decimal=',', encoding='latin1')
df.columns = df.columns.str.strip()

# Constantes physiques (eau)
cp_water = 4180  # J/kg·K
rho_water = 1000  # kg/m³

# Plages stables pour chaque T_ext
stable_ranges = {
    '10°C': (2160, 2260),
    '5°C': (3748, 3848),
    '0°C': (5140, 5240),
    '-5°C': (5613, 5713),
}

# Colonne de débit (supposée L/min)
debit_col = 'AI5 Flowmeter C2'
pelec_col = 'Com. PowerMeter Total Puissance Active'
T_in_col = 'Com. Pac Mesure Temp. eau glycolée source chaude(condenseur in)'
T_out_col = 'Com. Pac Mesure Temp. eau glycolée source chaude(condenseur out)'

# Calcul du COP
cop_results = {}

for T_ext, (start, end) in stable_ranges.items():
    subset = df.iloc[start:end].copy()

    # Débit massique (L/min → m³/s → kg/s)
    V_dot = subset[debit_col] / 1000 / 60  # m³/s
    m_dot = V_dot * rho_water  # kg/s

    # Températures entrée/sortie du condenseur
    T_in = subset[T_in_col]
    T_out = subset[T_out_col]

    # Puissance électrique
    P_elec = subset[pelec_col]

    # Puissance thermique → Q_dot négatif, on corrige avec un signe -
    Q_dot = m_dot * cp_water * (T_out - T_in)
    COP = - Q_dot / P_elec.replace(0, pd.NA)  # Corriger le signe

    # Moyenne sur l’intervalle stable
    COP_mean = COP.mean(skipna=True)
    cop_results[T_ext] = round(COP_mean, 3)

# Affichage
for T_ext, cop in cop_results.items():
    print(f"COP moyen à T_ext = {T_ext} : {cop}")
