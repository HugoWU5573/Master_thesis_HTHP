import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

###############################################################
## PART 1 : Retrieve the results
###############################################################

csv_path = Path(__file__).resolve().parent / "Case_Study_all_results.csv"
data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

alpha = np.asarray(data["alpha"])
glide_HT = np.asarray(data["glide_HT"])
COP = np.asarray(data["COP"])
COP_Carnot = np.asarray(data["COP_Carnot"])
COP_Lorenz = np.asarray(data["COP_Lorenz"])
eta_2 = np.asarray(data["eta_2"])
eta_2_Lorenz = np.asarray(data["eta_2_Lorenz"])

# Split data into 40K and 60K groups
mask_40K = glide_HT == 40
mask_60K = glide_HT == 60

alpha_40K = alpha[mask_40K]
COP_40K = COP[mask_40K]
COP_Carnot_40K = COP_Carnot[mask_40K]
COP_Lorenz_40K = COP_Lorenz[mask_40K]
eta_2_40K = eta_2[mask_40K]
eta_2_Lorenz_40K = eta_2_Lorenz[mask_40K]

alpha_60K = alpha[mask_60K]
COP_60K = COP[mask_60K]
COP_Carnot_60K = COP_Carnot[mask_60K]
COP_Lorenz_60K = COP_Lorenz[mask_60K]
eta_2_60K = eta_2[mask_60K]
eta_2_Lorenz_60K = eta_2_Lorenz[mask_60K]


###############################################################
## PART 2 : Plot the results
###############################################################

show_part_2 = True

# Plot COP vs alpha for glide_HT = 40K and glide_HT = 60K (and comparison with Carnot)

upper_bound = 0.6 * COP_Carnot_40K
lower_bound = 0.4 * COP_Carnot_40K

if show_part_2 : 

    plt.figure()
    plt.plot(alpha_40K, COP_40K, zorder=3, color="black")
    plt.text(0.75, COP_40K[alpha_40K == 0.75][0] + 0.1, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40K$", fontsize=12, color="black", zorder=3, rotation=25)
    plt.plot(alpha_60K, COP_60K, zorder=3, color="black", linestyle="--")
    plt.text(0.1, COP_60K[alpha_60K == 0.1][0] + 0.1, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60K$", fontsize=12, color="black", zorder=3, rotation=10)
    plt.plot(alpha_40K, COP_Carnot_40K, color="grey", zorder=2)
    plt.text(0.1, COP_Carnot_40K[alpha_40K == 0.1][0] + 0.1, r"$\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="grey", zorder=2, rotation=15)
    #plt.plot(alpha_40K, upper_bound, linestyle=":", color="grey", zorder=1)
    #plt.plot(alpha_40K, lower_bound, linestyle=":", color="grey", zorder=1)
    plt.fill_between(alpha_40K, lower_bound, upper_bound, color="grey", alpha=0.2, zorder=1)
    plt.text(0.75, upper_bound[alpha_40K == 0.75][0] - 0.2, r"$60\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="grey", zorder=1, rotation=20)
    plt.text(0.75, lower_bound[alpha_40K == 0.75][0] + 0.1, r"$40\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="grey", zorder=1, rotation=15)
    plt.text(0.30, (upper_bound[alpha_40K == 0.30][0] + lower_bound[alpha_40K == 0.30][0])/2, r"$\mathrm{State}$-$\mathrm{of}$-$\mathrm{the}$-$\mathrm{art}$ $\mathrm{range}$", fontsize=12, color="grey", zorder=1, va="center", ha="center", rotation=8)
    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(1, 5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('COP  [-]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(1, 5.1, 1))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/COP_vs_alpha.pdf')
    plt.show()

    # Plot the second law efficiency vs alpha for glide_HT = 40K and glide_HT = 60K (Carnot and Lorenz)
    plt.figure()
    plt.plot(alpha_40K, eta_2_40K*100, zorder=3, color="black")
    plt.text(0.1, eta_2_40K[alpha_40K == 0.1][0]*100 + 1, r"$\mathrm{\eta}_{\mathrm{2nd,Carnot}}$ $(40K)$", fontsize=12, color="black", zorder=3, rotation=-1)
    plt.plot(alpha_60K, eta_2_60K*100, zorder=3, color="black", linestyle="--")
    plt.text(0.1, eta_2_60K[alpha_60K == 0.1][0]*100 + 1, r"$\mathrm{\eta}_{\mathrm{2nd,Carnot}}$ $(60K)$", fontsize=12, color="black", zorder=3, rotation=2)
    plt.plot(alpha_40K, eta_2_Lorenz_40K*100, color="grey", zorder=2)
    plt.text(0.75, eta_2_Lorenz_40K[alpha_40K == 0.75][0]*100, r"$\mathrm{\eta}_{\mathrm{2nd,Lorenz}}$ $(40K)$", fontsize=12, color="grey", zorder=2, rotation=-5)
    plt.plot(alpha_60K, eta_2_Lorenz_60K*100, color="grey", zorder=2, linestyle="--")
    plt.text(0.75, eta_2_Lorenz_60K[alpha_60K == 0.75][0]*100 - 6, r"$\mathrm{\eta}_{\mathrm{2nd,Lorenz}}$ $(60K)$", fontsize=12, color="grey", zorder=2, rotation=-10)
    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(40, 80)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'$\eta_{ex}$  [%]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(40, 81, 5))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/Second_Law_Efficiency_vs_alpha.pdf')
    plt.show()


###############################################################
## PART 3 : Compute the carbon footprint
###############################################################

show_part_3 = True

low_carbon_grid_emission = 150          # [gCO2/kWh]  (e.g. EU in 2025 is 172)
high_carbon_grid_emission = 500         # [gCO2/kWh]  (e.g. China in 2025 is 532)
gas_boiler_emission = 205               # [gCO2/kWh]  (Mateu-Royo et al. 2021)

intensity_gas_boiler = gas_boiler_emission * np.ones(len(alpha_40K))
intensity_low_carbon_grid_40K = low_carbon_grid_emission/COP_40K
intensity_low_carbon_grid_60K = low_carbon_grid_emission/COP_60K
intensity_high_carbon_grid_40K = high_carbon_grid_emission/COP_40K
intensity_high_carbon_grid_60K = high_carbon_grid_emission/COP_60K

if show_part_3 :

    plt.figure()
    plt.plot(alpha_40K, intensity_gas_boiler, linestyle=":", color="black", zorder=3)
    plt.text(0.8, intensity_gas_boiler[0] + 5, r"$\mathrm{Gas}$ $\mathrm{boiler}$", fontsize=12, color="black", zorder=3)
    plt.plot(alpha_40K, intensity_low_carbon_grid_40K, color="black", zorder=2)
    plt.text(0.6, intensity_low_carbon_grid_40K[alpha_40K == 0.6][0]-2, r"$\mathrm{Low}$ $\mathrm{carbon}$ $\mathrm{grid}$", fontsize=12, color="grey", zorder=2, rotation=-5)
    plt.plot(alpha_60K, intensity_low_carbon_grid_60K, color="black", zorder=3, linestyle="--")
    plt.text(0.1, intensity_low_carbon_grid_60K[alpha_60K == 0.1][0]-20, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60K$", fontsize=12, color="black", zorder=3, rotation=-3)
    plt.text(0.1, intensity_low_carbon_grid_40K[alpha_40K == 0.1][0]+2, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40K$", fontsize=12, color="black", zorder=2, rotation=-3)
    plt.plot(alpha_40K, intensity_high_carbon_grid_40K, color="black", zorder=1)
    plt.text(0.6, intensity_high_carbon_grid_40K[alpha_40K == 0.6][0]-20, r"$\mathrm{High}$ $\mathrm{carbon}$ $\mathrm{grid}$", fontsize=12, color="grey", zorder=1, rotation=-15)
    plt.plot(alpha_60K, intensity_high_carbon_grid_60K, color="black", zorder=3, linestyle="--")
    plt.fill_between(alpha_40K, intensity_low_carbon_grid_40K, intensity_low_carbon_grid_60K, color="grey", alpha=0.2, zorder=1)
    plt.fill_between(alpha_40K, intensity_high_carbon_grid_40K, intensity_high_carbon_grid_60K, color="grey", alpha=0.2, zorder=1)
    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 250)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'Carbon Footprint  [gCO$_2$/kWh]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 251, 50))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/Carbon_Footprint_vs_alpha.pdf')
    plt.show()


###############################################################
## PART 4 : Compute the LCOH
###############################################################

show_part_4 = True

CAPEX_heat_pump = 1000                  # [€/kW] (from Arpagaus et al. 2018 (800€) corrected for inflation to 2026 https://fxtop.com/fr/calculateur-inflation-entre-deux-dates.php)
CAPEX_gas_boiler = 70                   # [€/kW] (from Bamigbetan et al. 2019 (50€ in 2009) corrected for inflation to 2026 https://fxtop.com/fr/calculateur-inflation-entre-deux-dates.php)
natural_gas_boiler_efficiency = 0.95    # [-] (Mateu-Royo et al. 2021)
operating_hours_per_year = 8000         # [h/year]  (from Jouhara et al. 2024 + Bamigbetan et al. 2019)
lifetime_years = 25

electricity_price_high = 90/1000  # [€/kWh] (e.g. EU in 2025 was 107 USD/MWh, https://www.iea.org/data-and-statistics/charts/estimated-final-electricity-price-for-large-industrial-customers-in-energy-intensive-industries-2019-2025)
electricity_price_low = 40/1000   # [€/kWh] (e.g. USA in 2025 was 50 USD/MWh, https://www.iea.org/data-and-statistics/charts/estimated-final-electricity-price-for-large-industrial-customers-in-energy-intensive-industries-2019-2025)

gas_price_high = 35/1000          # [€/kWh] (e.g. China and EU in 2025 were around 35-40 USD/MWh)
gas_price_low = 13/1000           # [€/kWh] (e.g. USA in 2025 was around 14USD/MWh)

interest_rate = 0.05              # (assumed)

discount_factor = interest_rate / (1 - (1 + interest_rate)**(-lifetime_years))

LCOH_heat_pump_low_40K = np.zeros(len(alpha_40K))
LCOH_heat_pump_high_40K = np.zeros(len(alpha_40K))
LCOH_heat_pump_low_60K = np.zeros(len(alpha_60K))
LCOH_heat_pump_high_60K = np.zeros(len(alpha_60K))
LCOH_gas_boiler_low = np.zeros(len(alpha_40K))
LCOH_gas_boiler_high = np.zeros(len(alpha_40K))

for i in range(len(alpha_40K)) :
    LCOH_heat_pump_low_40K[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_low / COP_40K[i]
    LCOH_heat_pump_high_40K[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_high / COP_40K[i]
    LCOH_heat_pump_low_60K[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_low / COP_60K[i]
    LCOH_heat_pump_high_60K[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_high / COP_60K[i]
    LCOH_gas_boiler_low[i] = CAPEX_gas_boiler * discount_factor / operating_hours_per_year + gas_price_low / natural_gas_boiler_efficiency
    LCOH_gas_boiler_high[i] = CAPEX_gas_boiler * discount_factor / operating_hours_per_year + gas_price_high / natural_gas_boiler_efficiency

if show_part_4 :

    plt.figure()
    plt.plot(alpha_40K, LCOH_gas_boiler_high*100, linestyle=":", color="black", zorder=3)
    plt.text(0.75, LCOH_gas_boiler_high[0]*100 +0.1, r"$\mathrm{Gas}$ $\mathrm{boiler}$ $\mathrm{high}$", fontsize=12, color="black", zorder=3)
    plt.plot(alpha_40K, LCOH_gas_boiler_low*100, linestyle=":", color="black", zorder=3)
    plt.text(0.01, LCOH_gas_boiler_low[0]*100 +0.1, r"$\mathrm{Gas}$ $\mathrm{boiler}$ $\mathrm{low}$", fontsize=12, color="black", zorder=3)
    plt.plot(alpha_40K, LCOH_heat_pump_low_40K*100, color="black", zorder=2)
    plt.plot(alpha_40K, LCOH_heat_pump_high_40K*100, color="black", zorder=2)
    plt.text(0.3, LCOH_heat_pump_high_40K[alpha_40K == 0.3]*100 -0.2, r"$\mathrm{High}$ $\mathrm{cost}$ $\mathrm{grid}$", fontsize=12, color="grey", zorder=2, rotation=-15)
    plt.text(0.7, LCOH_heat_pump_low_40K[alpha_40K == 0.7]*100 -0.1, r"$\mathrm{Low}$ $\mathrm{cost}$ $\mathrm{grid}$", fontsize=12, color="grey", zorder=2, rotation=-10)
    plt.plot(alpha_60K, LCOH_heat_pump_low_60K*100, color="black", zorder=3, linestyle="--")
    plt.plot(alpha_60K, LCOH_heat_pump_high_60K*100, color="black", zorder=3, linestyle="--")
    plt.text(0.1, LCOH_heat_pump_low_40K[alpha_40K == 0.1]*100, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40K$", fontsize=12, color="black", zorder=2, rotation=-6)
    plt.text(0.1, LCOH_heat_pump_low_60K[alpha_60K == 0.1]*100-0.35, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60K$", fontsize=12, color="black", zorder=3, rotation=-6)
    plt.fill_between(alpha_40K, LCOH_heat_pump_low_40K*100, LCOH_heat_pump_low_60K*100, color="grey", alpha=0.2, zorder=2)
    plt.fill_between(alpha_40K, LCOH_heat_pump_high_40K*100, LCOH_heat_pump_high_60K*100, color="grey", alpha=0.2, zorder=3)
    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(1, 5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'LCOH  [c€/kWh]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(1, 5.5, 1))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/LCOH_vs_alpha.pdf')
    plt.show()