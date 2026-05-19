import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

###############################################################
## PART 1 : Retrieve the results
###############################################################

csv_path = Path(__file__).resolve().parent / "Case_Study_dual_results.csv"
data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

alpha_raw = np.asarray(data["alpha"])
COP_dual = np.asarray(data["COP"])
COP_Carnot = np.asarray(data["COP_Carnot"])
COP_Lorenz = np.asarray(data["COP_Lorenz"])
eta_2 = np.asarray(data["eta_2"])
eta_2_Lorenz = np.asarray(data["eta_2_Lorenz"])

# Split data into four groups: glide_HT = 40K or 60K, and T_MT = 40C or 60C
mask_40K_40C = (data["glide_HT"] == 40) & (data["T_MT"] == 40) & (data["alpha"] != 0.90)
mask_40K_60C = (data["glide_HT"] == 40) & (data["T_MT"] == 60) & (data["alpha"] != 0.90)
mask_60K_40C = (data["glide_HT"] == 60) & (data["T_MT"] == 40) & (data["alpha"] != 0.90)
mask_60K_60C = (data["glide_HT"] == 60) & (data["T_MT"] == 60) & (data["alpha"] != 0.90)

alpha = alpha_raw[mask_40K_60C]
COP_Carnot_40C = COP_Carnot[mask_40K_40C]
COP_Carnot_60C = COP_Carnot[mask_40K_60C]

COP_40K_40C = COP_dual[mask_40K_40C]
COP_Lorenz_40K_40C = COP_Lorenz[mask_40K_40C]
eta_2_40K_40C = eta_2[mask_40K_40C]
eta_2_Lorenz_40K_40C = eta_2_Lorenz[mask_40K_40C]

COP_40K_60C = COP_dual[mask_40K_60C]
COP_Lorenz_40K_60C = COP_Lorenz[mask_40K_60C]
eta_2_40K_60C = eta_2[mask_40K_60C]
eta_2_Lorenz_40K_60C = eta_2_Lorenz[mask_40K_60C]

COP_60K_40C = COP_dual[mask_60K_40C]
COP_Lorenz_60K_40C = COP_Lorenz[mask_60K_40C]
eta_2_60K_40C = eta_2[mask_60K_40C]
eta_2_Lorenz_60K_40C = eta_2_Lorenz[mask_60K_40C]

COP_60K_60C = COP_dual[mask_60K_60C]
COP_Lorenz_60K_60C = COP_Lorenz[mask_60K_60C]
eta_2_60K_60C = eta_2[mask_60K_60C]
eta_2_Lorenz_60K_60C = eta_2_Lorenz[mask_60K_60C]


###############################################################
## PART 2 : Plot the results
###############################################################

show_part_2_single_curve = False
show_part_2_dual_curve = False

upper_bound_40C = 0.6 * COP_Carnot_40C
lower_bound_40C = 0.4 * COP_Carnot_40C
upper_bound_60C = 0.6 * COP_Carnot_60C
lower_bound_60C = 0.4 * COP_Carnot_60C

if show_part_2_single_curve :

    # Plot COP vs alpha (and comparison with Carnot)
    plt.figure()

        # Plot for the T_MT = 60C case
    plt.plot(alpha, COP_40K_60C, zorder=3, color="black")
    plt.text(0.75, COP_40K_60C[alpha == 0.75][0] + 0.1, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", fontsize=12, color="black", zorder=3, rotation=25)
    plt.plot(alpha, COP_60K_60C, zorder=3, color="black", linestyle="--")
    plt.text(0.1, COP_60K_60C[alpha == 0.1][0] + 0.1, r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$", fontsize=12, color="black", zorder=3, rotation=10)
    plt.plot(alpha, COP_Carnot_60C, color="black", zorder=2, alpha=0.5)
    plt.text(0.1, COP_Carnot_60C[alpha == 0.1][0] + 0.1, r"$\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="black", alpha=0.5, zorder=2, rotation=15)
    plt.fill_between(alpha, lower_bound_60C, upper_bound_60C, color="black", alpha=0.2, zorder=1)
    plt.text(0.75, upper_bound_60C[alpha == 0.75][0] - 0.2, r"$60\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="black", alpha=0.5, zorder=1, rotation=20)
    plt.text(0.75, lower_bound_60C[alpha == 0.75][0] + 0.1, r"$40\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="black", alpha=0.5, zorder=1, rotation=15)
    plt.text(0.30, (upper_bound_60C[alpha == 0.30][0] + lower_bound_60C[alpha == 0.30][0])/2, r"$\mathrm{State}$-$\mathrm{of}$-$\mathrm{the}$-$\mathrm{art}$ $\mathrm{range}$", fontsize=12, color="black", alpha=0.5, zorder=1, va="center", ha="center", rotation=8)

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

    # Plot the second law efficiency vs alpha
    plt.figure()

        # Plot for the T_MT = 60C case
    plt.plot(alpha, eta_2_40K_60C*100, zorder=3, color="black", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$")
    plt.text(0.8, eta_2_40K_60C[alpha == 0.8][0]*100+1, r"$\mathrm{\eta}_{\mathrm{2nd,Carnot}}$", fontsize=14, color="black", zorder=3, rotation=10)
    plt.plot(alpha, eta_2_60K_60C*100, zorder=3, color="black", linestyle="--", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$")
    plt.plot(alpha, eta_2_Lorenz_40K_60C*100, color="black", alpha=0.5, zorder=2)
    plt.text(0.8, eta_2_Lorenz_40K_60C[alpha == 0.8][0]*100+0.2, r"$\mathrm{\eta}_{\mathrm{2nd,Lorenz}}$", fontsize=14, color="black", alpha=0.5, zorder=2, rotation=-5)
    plt.plot(alpha, eta_2_Lorenz_60K_60C*100, color="black", alpha=0.5, zorder=2, linestyle="--")

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
    ax.set_yticks(np.arange(40, 81, 10))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12, loc="lower left")
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/Second_Law_Efficiency_vs_alpha.pdf')
    plt.show()


if show_part_2_dual_curve : 

    # Plot COP vs alpha (and comparison with Carnot)
    plt.figure()

        # Plot for the T_MT = 60C case
    plt.plot(alpha, COP_40K_60C, zorder=3, color="black", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$")
    plt.plot(alpha, COP_60K_60C, zorder=3, color="black", linestyle="--", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$")
    plt.fill_between(alpha, lower_bound_60C, upper_bound_60C, color="black", alpha=0.2, zorder=1)
    plt.text(0.75, upper_bound_60C[alpha == 0.75][0] - 0.2, r"$60\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="black", alpha=0.5, zorder=1, rotation=20)
    plt.text(0.75, lower_bound_60C[alpha == 0.75][0] + 0.1, r"$40\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="black", alpha=0.5, zorder=1, rotation=15)
    plt.text(0.30, (upper_bound_60C[alpha == 0.30][0] + lower_bound_60C[alpha == 0.30][0])/2, r"$\mathrm{State}$-$\mathrm{of}$-$\mathrm{the}$-$\mathrm{art}$ $\mathrm{range}$", fontsize=12, color="black", alpha=0.5, zorder=1, va="center", ha="center", rotation=8)

        # Plot for the T_MT = 40C case
    plt.plot(alpha, COP_40K_40C, zorder=3, color="red")
    plt.plot(alpha, COP_60K_40C, zorder=3, color="red", linestyle="--")
    plt.fill_between(alpha, lower_bound_40C, upper_bound_40C, color="red", alpha=0.2, zorder=1)
    plt.text(0.75, upper_bound_40C[alpha == 0.75][0] - 0.22, r"$60\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="red", alpha=0.5, zorder=1, rotation=8)
    plt.text(0.75, lower_bound_40C[alpha == 0.75][0] + 0.08, r"$40\%$ $\mathrm{COP}_{\mathrm{Carnot}}$", fontsize=12, color="red", alpha=0.5, zorder=1, rotation=6)
    plt.text(0.30, (upper_bound_40C[alpha == 0.30][0] + lower_bound_40C[alpha == 0.30][0])/2, r"$\mathrm{State}$-$\mathrm{of}$-$\mathrm{the}$-$\mathrm{art}$ $\mathrm{range}$", fontsize=12, color="red", alpha=0.5, zorder=1, va="center", ha="center", rotation=4)

    plt.scatter(-1, -1, color="red", label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$", marker="s")
    plt.scatter(-1, -1, color="black", label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$", marker="s")

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
    plt.legend(frameon=False, fontsize=12, ncols=2)
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/COP_vs_alpha_2.pdf')
    plt.show()

    # Plot the second law efficiency vs alpha
    plt.figure()

        # Plot for the T_MT = 60C case
    plt.plot(alpha, eta_2_40K_60C*100, zorder=3, color="black", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$")
    plt.text(0.01, eta_2_60K_40C[alpha == 0.01][0]*100 + 1.25, r"$\mathrm{\eta}_{\mathrm{2nd,Carnot}}$", fontsize=14, color="red", zorder=3, rotation=1.5)
    plt.plot(alpha, eta_2_60K_60C*100, zorder=3, color="black", linestyle="--", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$")
    plt.plot(alpha, eta_2_Lorenz_40K_60C*100, color="black", alpha=0.5, zorder=2)
    plt.text(0.8, eta_2_Lorenz_40K_40C[alpha == 0.8][0]*100+1.25, r"$\mathrm{\eta}_{\mathrm{2nd,Lorenz}}$", fontsize=14, color="red", alpha=0.5, zorder=2, rotation=-3)
    plt.plot(alpha, eta_2_Lorenz_60K_60C*100, color="black", alpha=0.5, zorder=2, linestyle="--")

        # Plot for the T_MT = 40C case
    plt.plot(alpha, eta_2_40K_40C*100, zorder=3, color="red")
    plt.plot(alpha, eta_2_60K_40C*100, zorder=3, color="red", linestyle="--")
    plt.plot(alpha, eta_2_Lorenz_40K_40C*100, color="red", alpha=0.5, zorder=2)
    plt.plot(alpha, eta_2_Lorenz_60K_40C*100, color="red", alpha=0.5, zorder=2, linestyle="--")

    plt.scatter(-1, -1, color="red", label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$", marker="s")
    plt.scatter(-1, -1, color="black", label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$", marker="s")

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
    ax.set_yticks(np.arange(40, 81, 10))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12, ncols=2, loc='upper right', bbox_to_anchor=(1, 1.1))
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/Second_Law_Efficiency_vs_alpha_2.pdf')
    plt.show()


###############################################################
## PART 3 : Compute the carbon footprint
###############################################################

show_part_3 = False

low_carbon_grid_emission = 150          # [gCO2/kWh]  (e.g. EU in 2025 is 172)
high_carbon_grid_emission = 500         # [gCO2/kWh]  (e.g. China in 2025 is 532)
gas_boiler_emission = 205               # [gCO2/kWh]  (Mateu-Royo et al. 2021)

# Results for the gas boiler
intensity_gas_boiler = gas_boiler_emission * np.ones(len(alpha))

# Results for T_MT = 60C
intensity_low_carbon_grid_40K_60C = low_carbon_grid_emission/COP_40K_60C
intensity_low_carbon_grid_60K_60C = low_carbon_grid_emission/COP_60K_60C
intensity_high_carbon_grid_40K_60C = high_carbon_grid_emission/COP_40K_60C
intensity_high_carbon_grid_60K_60C = high_carbon_grid_emission/COP_60K_60C

# Results for T_MT = 40C
intensity_low_carbon_grid_40K_40C = low_carbon_grid_emission/COP_40K_40C
intensity_low_carbon_grid_60K_40C = low_carbon_grid_emission/COP_60K_40C
intensity_high_carbon_grid_40K_40C = high_carbon_grid_emission/COP_40K_40C
intensity_high_carbon_grid_60K_40C = high_carbon_grid_emission/COP_60K_40C

if show_part_3 :

    plt.figure()

    # Plot for the gas boiler
    plt.plot(alpha, intensity_gas_boiler, linestyle=":", color="black", zorder=3)
    plt.text(0.8, intensity_gas_boiler[0] + 5, r"$\mathrm{Gas}$ $\mathrm{boiler}$", fontsize=12, color="black", zorder=3)

    # Plot for the T_MT = 60C case
    plt.plot(alpha, intensity_low_carbon_grid_40K_60C, color="black", zorder=2, label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$")
    plt.text(0.7, intensity_low_carbon_grid_40K_60C[alpha == 0.7][0]+8, r"$150$ [gCO$_2$/$\mathrm{kWh}_{\mathrm{el}}$]", fontsize=12, color="black", zorder=2, rotation=-2)
    plt.plot(alpha, intensity_low_carbon_grid_60K_60C, color="black", zorder=3, linestyle="--", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$")
    plt.plot(alpha, intensity_high_carbon_grid_40K_60C, color="black", zorder=1)
    plt.text(0.7, intensity_high_carbon_grid_40K_60C[alpha == 0.7][0]+15, r"$500$ [gCO$_2$/$\mathrm{kWh}_{\mathrm{el}}$]", fontsize=12, color="black", zorder=1, rotation=-8)
    plt.plot(alpha, intensity_high_carbon_grid_60K_60C, color="black", zorder=3, linestyle="--")
    plt.fill_between(alpha, intensity_low_carbon_grid_40K_60C, intensity_low_carbon_grid_60K_60C, color="black", alpha=0.2, zorder=1, label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$")
    plt.fill_between(alpha, intensity_high_carbon_grid_40K_60C, intensity_high_carbon_grid_60K_60C, color="black", alpha=0.2, zorder=1)

    # Plot for the T_MT = 40C case
    plt.plot(alpha, intensity_low_carbon_grid_40K_40C, color="red", zorder=2)
    plt.plot(alpha, intensity_low_carbon_grid_60K_40C, color="red", zorder=3, linestyle="--")
    plt.plot(alpha, intensity_high_carbon_grid_40K_40C, color="red", zorder=1)
    plt.plot(alpha, intensity_high_carbon_grid_60K_40C, color="red", zorder=3, linestyle="--")
    plt.fill_between(alpha, intensity_low_carbon_grid_40K_40C, intensity_low_carbon_grid_60K_40C, color="red", alpha=0.2, zorder=1, label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$")
    plt.fill_between(alpha, intensity_high_carbon_grid_40K_40C, intensity_high_carbon_grid_60K_40C, color="red", alpha=0.2, zorder=1)

    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 250)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'Carbon Footprint  [gCO$_2$/$\mathrm{kWh}_{\mathrm{th}}$]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 251, 50))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12, ncols=2, loc="lower left", bbox_to_anchor=(0, -0.05))
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/Carbon_Footprint_vs_alpha.pdf')
    plt.show()


###############################################################
## PART 4 : Compute the LCOH
###############################################################

show_part_4 = False

CAPEX_heat_pump = 1600                  # [€/kW] (from compute_cost.py, method based on Ommen et al. 2015)
CAPEX_gas_boiler = 70                   # [€/kW] (from Bamigbetan et al. 2019 (50€ in 2009) corrected for inflation to 2026 https://fxtop.com/fr/calculateur-inflation-entre-deux-dates.php)
natural_gas_boiler_efficiency = 0.95    # [-] (Mateu-Royo et al. 2021)
operating_hours_per_year = 8000         # [h/year]  (from Jouhara et al. 2024 + Bamigbetan et al. 2019)
lifetime_years = 20                     # Ommen et al. 

electricity_price_high = 90/1000  # [€/kWh] (e.g. EU in 2025 was 107 USD/MWh, https://www.iea.org/data-and-statistics/charts/estimated-final-electricity-price-for-large-industrial-customers-in-energy-intensive-industries-2019-2025)
electricity_price_low = 40/1000   # [€/kWh] (e.g. USA in 2025 was 50 USD/MWh, https://www.iea.org/data-and-statistics/charts/estimated-final-electricity-price-for-large-industrial-customers-in-energy-intensive-industries-2019-2025)

gas_price_high = 35/1000          # [€/kWh] (e.g. China and EU in 2025 were around 35-40 USD/MWh)
gas_price_low = 13/1000           # [€/kWh] (e.g. USA in 2025 was around 14USD/MWh)

carbon_tax = 80/1e6               # [€/gCO2] (e.g. ETS price in EU is around 80€/tonne CO2 in 2026 for the direct emissions -> only applies to the gas boiler https://tradingeconomics.com/commodity/carbon)

interest_rate = 0.05              # (assumed)

discount_factor = interest_rate / (1 - (1 + interest_rate)**(-lifetime_years))

# Results for the gas boiler
LCOH_gas_boiler_low = np.zeros(len(alpha))
LCOH_gas_boiler_high = np.zeros(len(alpha))
LCOH_gas_boiler_low_no_tax = np.zeros(len(alpha))

# Results for the T_MT = 60C case
LCOH_heat_pump_low_40K_60C = np.zeros(len(alpha))
LCOH_heat_pump_high_40K_60C = np.zeros(len(alpha))
LCOH_heat_pump_low_60K_60C = np.zeros(len(alpha))
LCOH_heat_pump_high_60K_60C = np.zeros(len(alpha))

# Results for the T_MT = 40C case
LCOH_heat_pump_low_40K_40C = np.zeros(len(alpha))
LCOH_heat_pump_high_40K_40C = np.zeros(len(alpha))
LCOH_heat_pump_low_60K_40C = np.zeros(len(alpha))
LCOH_heat_pump_high_60K_40C = np.zeros(len(alpha))

for i in range(len(alpha)) :

    # Gas boiler
    LCOH_gas_boiler_low[i] = CAPEX_gas_boiler * discount_factor / operating_hours_per_year + gas_price_low / natural_gas_boiler_efficiency + gas_boiler_emission * carbon_tax
    LCOH_gas_boiler_high[i] = CAPEX_gas_boiler * discount_factor / operating_hours_per_year + gas_price_high / natural_gas_boiler_efficiency + gas_boiler_emission * carbon_tax
    LCOH_gas_boiler_low_no_tax[i] = CAPEX_gas_boiler * discount_factor / operating_hours_per_year + gas_price_low / natural_gas_boiler_efficiency

    # T_MT = 60C case
    LCOH_heat_pump_low_40K_60C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_low / COP_40K_60C[i]
    LCOH_heat_pump_high_40K_60C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_high / COP_40K_60C[i]
    LCOH_heat_pump_low_60K_60C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_low / COP_60K_60C[i]
    LCOH_heat_pump_high_60K_60C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_high / COP_60K_60C[i]

    # T_MT = 40C case
    LCOH_heat_pump_low_40K_40C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_low / COP_40K_40C[i]
    LCOH_heat_pump_high_40K_40C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_high / COP_40K_40C[i]
    LCOH_heat_pump_low_60K_40C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_low / COP_60K_40C[i]
    LCOH_heat_pump_high_60K_40C[i] = CAPEX_heat_pump * discount_factor / operating_hours_per_year + electricity_price_high / COP_60K_40C[i]

if show_part_4 :

    plt.figure()

    # Plot for the gas boiler
    plt.plot(alpha, LCOH_gas_boiler_high*100, linestyle=":", color="black", zorder=3)
    plt.text(0.75, LCOH_gas_boiler_high[0]*100 + 0.1, r"$\mathrm{Gas}$ $\mathrm{boiler}$ $\mathrm{high}$", fontsize=12, color="black", zorder=3)
    plt.plot(alpha, LCOH_gas_boiler_low*100, linestyle=":", color="black", zorder=3)
    plt.text(0.75, LCOH_gas_boiler_low[0]*100 +0.1, r"$\mathrm{Gas}$ $\mathrm{boiler}$ $\mathrm{low}$", fontsize=12, color="black", zorder=3)
    #plt.plot(alpha, LCOH_gas_boiler_low_no_tax*100, linestyle=":", color="black", zorder=2)
    #plt.text(0.01, LCOH_gas_boiler_low_no_tax[0]*100 - 0.3, r"$\mathrm{Gas}$ $\mathrm{boiler}$ $\mathrm{low}$ $\mathrm{(no}$ $\mathrm{tax)}$", fontsize=12, color="black", zorder=2)

    # Plot for the T_MT = 60C case
    plt.plot(alpha, LCOH_heat_pump_low_40K_60C*100, color="black", zorder=2, label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$")
    plt.plot(alpha, LCOH_heat_pump_high_40K_60C*100, color="black", zorder=2)
    plt.plot(alpha, LCOH_heat_pump_low_60K_60C*100, color="black", zorder=3, linestyle="--", label=r"$\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$")
    plt.plot(alpha, LCOH_heat_pump_high_60K_60C*100, color="black", zorder=3, linestyle="--")
    plt.fill_between(alpha, LCOH_heat_pump_low_40K_60C*100, LCOH_heat_pump_low_60K_60C*100, color="black", alpha=0.2, zorder=2, label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$")
    plt.fill_between(alpha, LCOH_heat_pump_high_40K_60C*100, LCOH_heat_pump_high_60K_60C*100, color="black", alpha=0.2, zorder=3)

    # Plot for the T_MT = 40C case
    plt.plot(alpha, LCOH_heat_pump_low_40K_40C*100, color="red", zorder=2)
    plt.plot(alpha, LCOH_heat_pump_high_40K_40C*100, color="red", zorder=2)
    plt.text(0.75, LCOH_heat_pump_high_40K_40C[alpha == 0.75]*100-0.08, r"$9$ [c€/$\mathrm{kWh}_{\mathrm{el}}$]", fontsize=12, color="black", zorder=2, rotation=-8)
    plt.text(0.1, LCOH_heat_pump_low_40K_40C[alpha == 0.1]*100+0.02, r"$4$ [c€/$\mathrm{kWh}_{\mathrm{el}}$]", fontsize=12, color="black", zorder=2, rotation=-3)
    plt.plot(alpha, LCOH_heat_pump_low_60K_40C*100, color="red", zorder=3, linestyle="--")
    plt.plot(alpha, LCOH_heat_pump_high_60K_40C*100, color="red", zorder=3, linestyle="--")
    plt.fill_between(alpha, LCOH_heat_pump_low_40K_40C*100, LCOH_heat_pump_low_60K_40C*100, color="red", alpha=0.2, zorder=2, label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$")
    plt.fill_between(alpha, LCOH_heat_pump_high_40K_40C*100, LCOH_heat_pump_high_60K_40C*100, color="red", alpha=0.2, zorder=3)

    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(2, 6)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'LCOH  [c€/$\mathrm{kWh}_{\mathrm{th}}$]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(2, 6.5, 1))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12, ncols=2, loc="lower left",bbox_to_anchor=(0, -0.05))    
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/LCOH_vs_alpha.pdf')
    plt.show()


###############################################################
## PART 5 : Compare single evaporator and dual evaporator
###############################################################

show_part_5 = False

csv_path = Path(__file__).resolve().parent / "Case_Study_single_results.csv"
data_single = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

COP_single = data_single["COP"]

mask_40K_40C = (data["glide_HT"] == 40) & (data["T_MT"] == 40) & (data["alpha"] != 0.90)
mask_40K_60C = (data["glide_HT"] == 40) & (data["T_MT"] == 60) & (data["alpha"] != 0.90)
mask_60K_40C = (data["glide_HT"] == 60) & (data["T_MT"] == 40) & (data["alpha"] != 0.90)
mask_60K_60C = (data["glide_HT"] == 60) & (data["T_MT"] == 60) & (data["alpha"] != 0.90)

alpha_single = alpha_raw[mask_40K_40C]
COP_single_40K_40C = COP_single[mask_40K_40C]
COP_single_40K_60C = COP_single[mask_40K_60C]
COP_single_60K_40C = COP_single[mask_60K_40C]
COP_single_60K_60C = COP_single[mask_60K_60C]

delta_COP_40K_40C = (COP_single_40K_40C - COP_40K_40C) / COP_single_40K_40C

delta_COP_40K_60C = (COP_single_40K_60C - COP_40K_60C) / COP_single_40K_60C

delta_COP_60K_40C = (COP_single_60K_40C - COP_60K_40C) / COP_single_60K_40C

delta_COP_60K_60C = (COP_single_60K_60C - COP_60K_60C) / COP_single_60K_60C
    
delta_COP_40K_40C_mean = np.mean(delta_COP_40K_40C)
print(f"On average, the single evaporator configuration has a COP that is {delta_COP_40K_40C_mean*100:.3f}% higher than the dual evaporator configuration for the T_MT = 40C and delta_T_HT = 40K case.")
delta_COP_40K_60C_mean = np.mean(delta_COP_40K_60C)
print(f"On average, the single evaporator configuration has a COP that is {delta_COP_40K_60C_mean*100:.3f}% higher than the dual evaporator configuration for the T_MT = 60C and delta_T_HT = 40K case.")
delta_COP_60K_40C_mean = np.mean(delta_COP_60K_40C)
print(f"On average, the single evaporator configuration has a COP that is {delta_COP_60K_40C_mean*100:.3f}% higher than the dual evaporator configuration for the T_MT = 40C and delta_T_HT = 60K case.")
delta_COP_60K_60C_mean = np.mean(delta_COP_60K_60C)
print(f"On average, the single evaporator configuration has a COP that is {delta_COP_60K_60C_mean*100:.3f}% higher than the dual evaporator configuration for the T_MT = 60C and delta_T_HT = 60K case.")

delta_COP_dual_single = np.mean([delta_COP_40K_40C_mean, delta_COP_40K_60C_mean, delta_COP_60K_40C_mean, delta_COP_60K_60C_mean])
print(f"On average, the single evaporator configuration has a COP that is {delta_COP_dual_single*100:.3f}% higher than the dual evaporator configuration.")


if show_part_5 :

    plt.figure()
    plt.plot([2, 5], [2, 5], color="black", linestyle="--")
    plt.scatter(COP_single_40K_40C, COP_40K_40C, label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", color="black")
    plt.scatter(COP_single_40K_60C, COP_40K_60C, label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", color="red")
    plt.scatter(COP_single_60K_40C, COP_60K_40C, label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$", color="green")
    plt.scatter(COP_single_60K_60C, COP_60K_60C, label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$", color="blue")

    plt.xlabel(r'COP Parallel HTHPs  [-]', fontsize=12)
    plt.xlim(2, 5)
    plt.ylim(2, 5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r"COP Integrated HTHP  [-]", loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(2, 5.1, 1))
    ax.set_yticks(np.arange(2, 5.1, 1))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/COP_single_vs_dual_scatter.pdf')
    plt.show()

    plt.figure()
    plt.plot(alpha_single, COP_single_40K_40C, color="black", label=r"Parallel HTHPs", linestyle="--")
    plt.plot(alpha, COP_40K_40C, color="black", label=r"Integrated HTHP", linestyle="-")
    plt.plot(alpha_single, COP_single_40K_60C, color="red", linestyle="--")
    plt.plot(alpha, COP_40K_60C, color="red")
    plt.plot(alpha_single, COP_single_60K_40C, color="green", linestyle="--")
    plt.plot(alpha, COP_60K_40C, color="green", linestyle="-.")
    plt.plot(alpha_single, COP_single_60K_60C, color="blue", linestyle="--")
    plt.plot(alpha, COP_60K_60C, color="blue")

    plt.scatter(-1, -1, color="black", label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", marker="s")
    plt.scatter(-1, -1, color="red", label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", marker="s")
    plt.scatter(-1, -1, color="green", label=r"$\mathrm{T}_{\mathrm{MT}} = 40\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$", marker="s")
    plt.scatter(-1, -1, color="blue", label=r"$\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 60\mathrm{K}$", marker="s")

    plt.xlabel(r'$\alpha$  [-]', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(2, 5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title('COP  [-]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(2, 5.1, 1))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/COP_single_vs_dual.pdf')
    plt.show()


###############################################################
## PART 6 : Create bar charts to decompose the LCOH
###############################################################

show_part_6 = False

CAPEX_heat_pump_single = 1690              # [€/kW]  (from compute_cost.py, method based on Ommen et al.)

    # Remark : We use the high cost (with carbon tax) case for illustration

CAPEX_PART_gas_boiler = CAPEX_gas_boiler * discount_factor / operating_hours_per_year * 100
OPEX_PART_gas_boiler = gas_price_high / natural_gas_boiler_efficiency * 100
TAX_PART_gas_boiler = gas_boiler_emission * carbon_tax * 100


    # Case 1 : T_MT = 60°C, glide_HT = 40K (demonstrator case), alpha = 0.5
alpha_case_1 = 0.5

COP_dual_CASE_1 = COP_40K_60C[alpha == alpha_case_1]
COP_single_CASE_1 = COP_single_40K_60C[alpha_single == alpha_case_1]

CAPEX_PART_heat_pump_dual_CASE_1 = CAPEX_heat_pump * discount_factor / operating_hours_per_year * 100
OPEX_PART_heat_pump_dual_CASE_1 = electricity_price_high / COP_dual_CASE_1 * 100
TAX_PART_heat_pump_dual_CASE_1 = 0

CAPEX_PART_heat_pump_single_CASE_1 = CAPEX_heat_pump_single * discount_factor / operating_hours_per_year * 100
OPEX_PART_heat_pump_single_CASE_1 = electricity_price_high / COP_single_CASE_1 * 100
TAX_PART_heat_pump_single_CASE_1 = 0


    # Case 2 : T_MT = 60°C, glide_HT = 40K (demonstrator case), alpha = 0.8
alpha_case_2 = 0.8

COP_dual_CASE_2 = COP_40K_60C[alpha == alpha_case_2]
COP_single_CASE_2 = COP_single_40K_60C[alpha_single == alpha_case_2]

CAPEX_PART_heat_pump_dual_CASE_2 = CAPEX_heat_pump * discount_factor / operating_hours_per_year * 100
OPEX_PART_heat_pump_dual_CASE_2 = electricity_price_high / COP_dual_CASE_2 * 100
TAX_PART_heat_pump_dual_CASE_2 = 0

CAPEX_PART_heat_pump_single_CASE_2 = CAPEX_heat_pump_single * discount_factor / operating_hours_per_year * 100
OPEX_PART_heat_pump_single_CASE_2 = electricity_price_high / COP_single_CASE_2 * 100
TAX_PART_heat_pump_single_CASE_2 = 0

if show_part_6 :

    # CASE 1 : T_MT = 60°C, glide_HT = 40K (demonstrator case), alpha = 0.5

    plt.figure()
    
    plt.barh(2, CAPEX_PART_gas_boiler, left=0, color="black")
    plt.barh(2, OPEX_PART_gas_boiler, left=CAPEX_PART_gas_boiler, color="black", alpha=0.5)
    plt.barh(2, TAX_PART_gas_boiler, left=CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler, color="red")

    plt.barh(0, CAPEX_PART_heat_pump_dual_CASE_1, left=0, color="black")
    plt.barh(0, OPEX_PART_heat_pump_dual_CASE_1, left=CAPEX_PART_heat_pump_dual_CASE_1, color="black", alpha=0.5)
    #plt.barh(0, TAX_PART_heat_pump_dual_CASE_1, left=CAPEX_PART_heat_pump_dual_CASE_1 + OPEX_PART_heat_pump_dual_CASE_1, color="red")

    plt.barh(1, CAPEX_PART_heat_pump_single_CASE_1, left=0, color="black")
    plt.barh(1, OPEX_PART_heat_pump_single_CASE_1, left=CAPEX_PART_heat_pump_single_CASE_1, color="black", alpha=0.5)
    #plt.barh(1, TAX_PART_heat_pump_single_CASE_1, left=CAPEX_PART_heat_pump_single_CASE_1 + OPEX_PART_heat_pump_single_CASE_1, color="red")

    total_heat_pump_single_CASE_1 = CAPEX_PART_heat_pump_single_CASE_1 + OPEX_PART_heat_pump_single_CASE_1 + TAX_PART_heat_pump_single_CASE_1
    total_heat_pump_dual_CASE_1 = CAPEX_PART_heat_pump_dual_CASE_1 + OPEX_PART_heat_pump_dual_CASE_1 + TAX_PART_heat_pump_dual_CASE_1
    total_gas_boiler = CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler + TAX_PART_gas_boiler

    gain = 1 - total_heat_pump_dual_CASE_1 / total_heat_pump_single_CASE_1
    print(f"By switching from the single evaporator configuration to the dual evaporator configuration, the LCOH is reduced by {gain[0]*100:.1f}% in the case 1.")

    percentage_CAPEX_single_CASE_1 = np.round(CAPEX_PART_heat_pump_single_CASE_1 / total_heat_pump_single_CASE_1 * 100,0)
    percentage_CAPEX_dual_CASE_1 = np.round(CAPEX_PART_heat_pump_dual_CASE_1 / total_heat_pump_dual_CASE_1 * 100,0)
    percentage_CAPEX_gas_boiler = np.round(CAPEX_PART_gas_boiler / total_gas_boiler * 100,0)

    percentage_OPEX_single_CASE_1 = np.round(OPEX_PART_heat_pump_single_CASE_1 / total_heat_pump_single_CASE_1 * 100,0)
    percentage_OPEX_dual_CASE_1 = np.round(OPEX_PART_heat_pump_dual_CASE_1 / total_heat_pump_dual_CASE_1 * 100,0)
    percentage_OPEX_gas_boiler = np.round(OPEX_PART_gas_boiler / total_gas_boiler * 100,0)

    percentage_TAX_single_CASE_1 = np.round(TAX_PART_heat_pump_single_CASE_1 / total_heat_pump_single_CASE_1 * 100,0)
    percentage_TAX_dual_CASE_1 = np.round(TAX_PART_heat_pump_dual_CASE_1 / total_heat_pump_dual_CASE_1 * 100,0)
    percentage_TAX_gas_boiler = np.round(TAX_PART_gas_boiler / total_gas_boiler * 100,0)

    plt.text((CAPEX_PART_heat_pump_dual_CASE_1)/2, 0, "CAPEX\n" + str(int(percentage_CAPEX_dual_CASE_1[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_heat_pump_single_CASE_1/2, 1, str(int(percentage_CAPEX_single_CASE_1[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_heat_pump_dual_CASE_1 + OPEX_PART_heat_pump_dual_CASE_1/2, 0, "OPEX\n" + str(int(percentage_OPEX_dual_CASE_1[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_heat_pump_single_CASE_1 + OPEX_PART_heat_pump_single_CASE_1/2, 1, str(int(percentage_OPEX_single_CASE_1[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler/2, 2, str(int(percentage_OPEX_gas_boiler)) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler + TAX_PART_gas_boiler/2, 2, "CARBON TAX\n" + str(int(percentage_TAX_gas_boiler)) + "%", fontsize=12, color="white", va="center", ha="center")

    plt.text(total_heat_pump_single_CASE_1+0.1, 1, "Parallel\nHTHPs", fontsize=12, color="black", va="center", ha="left")
    plt.text(total_heat_pump_dual_CASE_1+0.1, 0, "Integrated\nHTHP", fontsize=12, color="black", va="center", ha="left")
    plt.text(total_gas_boiler+0.1, 2, "Gas\nboiler", fontsize=12, color="black", va="center", ha="left")

    plt.xlabel(r"LCOH  [c€/$\mathrm{kWh}_{\mathrm{th}}$]", fontsize=12)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
    ax.set_yticks([])
    ax.set_xticks([0, np.round(total_gas_boiler,1)])
    ax.set_xlim(0,  np.round(total_gas_boiler,1))
    ax.set_xticklabels([0, str(np.round(total_gas_boiler,1))])
    #ax.set_yticklabels(["Single \n evaporator \n HTHPs", "Dual \n evaporator \n HTHP", "Gas \n boiler"])
    ax.tick_params(axis='y', which='both', length=0)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    plt.title(r"$\alpha = 0.5$, $\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", loc='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/LCOH_Breakdown_Case_1.pdf')
    plt.show()


    # CASE 2 : T_MT = 60°C, glide_HT = 40K (demonstrator case), alpha = 0.8

    plt.figure()
    
    plt.barh(2, CAPEX_PART_gas_boiler, left=0, color="black")
    plt.barh(2, OPEX_PART_gas_boiler, left=CAPEX_PART_gas_boiler, color="black", alpha=0.5)
    plt.barh(2, TAX_PART_gas_boiler, left=CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler, color="red")

    plt.barh(0, CAPEX_PART_heat_pump_dual_CASE_2, left=0, color="black")
    plt.barh(0, OPEX_PART_heat_pump_dual_CASE_2, left=CAPEX_PART_heat_pump_dual_CASE_2, color="black", alpha=0.5)
    plt.barh(0, TAX_PART_heat_pump_dual_CASE_2, left=CAPEX_PART_heat_pump_dual_CASE_2 + OPEX_PART_heat_pump_dual_CASE_2, color="red")

    plt.barh(1, CAPEX_PART_heat_pump_single_CASE_2, left=0, color="black")
    plt.barh(1, OPEX_PART_heat_pump_single_CASE_2, left=CAPEX_PART_heat_pump_single_CASE_2, color="black", alpha=0.5)
    plt.barh(1, TAX_PART_heat_pump_single_CASE_2, left=CAPEX_PART_heat_pump_single_CASE_2 + OPEX_PART_heat_pump_single_CASE_2, color="red")

    total_heat_pump_single_CASE_2 = CAPEX_PART_heat_pump_single_CASE_2 + OPEX_PART_heat_pump_single_CASE_2 + TAX_PART_heat_pump_single_CASE_2
    total_heat_pump_dual_CASE_2 = CAPEX_PART_heat_pump_dual_CASE_2 + OPEX_PART_heat_pump_dual_CASE_2 + TAX_PART_heat_pump_dual_CASE_2
    total_gas_boiler = CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler + TAX_PART_gas_boiler

    percentage_CAPEX_single_CASE_2 = np.round(CAPEX_PART_heat_pump_single_CASE_2 / total_heat_pump_single_CASE_2 * 100,0)
    percentage_CAPEX_dual_CASE_2 = np.round(CAPEX_PART_heat_pump_dual_CASE_2 / total_heat_pump_dual_CASE_2 * 100,0)
    percentage_CAPEX_gas_boiler = np.round(CAPEX_PART_gas_boiler / total_gas_boiler * 100,0)

    percentage_OPEX_single_CASE_2 = np.round(OPEX_PART_heat_pump_single_CASE_2 / total_heat_pump_single_CASE_2 * 100,0)
    percentage_OPEX_dual_CASE_2 = np.round(OPEX_PART_heat_pump_dual_CASE_2 / total_heat_pump_dual_CASE_2 * 100,0)
    percentage_OPEX_gas_boiler = np.round(OPEX_PART_gas_boiler / total_gas_boiler * 100,0)

    percentage_TAX_single_CASE_2 = np.round(TAX_PART_heat_pump_single_CASE_2 / total_heat_pump_single_CASE_2 * 100,0)
    percentage_TAX_dual_CASE_2 = np.round(TAX_PART_heat_pump_dual_CASE_2 / total_heat_pump_dual_CASE_2 * 100,0)
    percentage_TAX_gas_boiler = np.round(TAX_PART_gas_boiler / total_gas_boiler * 100,0)

    plt.text((CAPEX_PART_heat_pump_dual_CASE_2)/2, 0, "CAPEX\n" + str(int(percentage_CAPEX_dual_CASE_2[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_heat_pump_single_CASE_2/2, 1, str(int(percentage_CAPEX_single_CASE_2[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_heat_pump_dual_CASE_2 + OPEX_PART_heat_pump_dual_CASE_2/2, 0, "OPEX\n" + str(int(percentage_OPEX_dual_CASE_2[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_heat_pump_single_CASE_2 + OPEX_PART_heat_pump_single_CASE_2/2, 1, str(int(percentage_OPEX_single_CASE_2[0])) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler/2, 2, str(int(percentage_OPEX_gas_boiler)) + "%", fontsize=12, color="white", va="center", ha="center")
    plt.text(CAPEX_PART_gas_boiler + OPEX_PART_gas_boiler + TAX_PART_gas_boiler/2, 2, "CARBON TAX\n" + str(int(percentage_TAX_gas_boiler)) + "%", fontsize=12, color="white", va="center", ha="center")

    plt.text(total_heat_pump_single_CASE_2+0.1, 1, "Parallel HTHPs", fontsize=12, color="black", va="center", ha="left")
    plt.text(total_heat_pump_dual_CASE_2+0.1, 0, "Integrated HTHP", fontsize=12, color="black", va="center", ha="left")
    plt.text(total_gas_boiler+0.1, 2, "Gas\nboiler", fontsize=12, color="black", va="center", ha="left")

    plt.xlabel(r"LCOH  [c€/$\mathrm{kWh}_{\mathrm{th}}$]", fontsize=12)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
    ax.set_yticks([])
    ax.set_xticks([0, np.round(total_gas_boiler,1)])
    ax.set_xlim(0,  np.round(total_gas_boiler,1))
    ax.set_xticklabels([0, str(np.round(total_gas_boiler,1))])
    #ax.set_yticklabels(["Single \n evaporator \n HTHPs", "Dual \n evaporator \n HTHP", "Gas \n boiler"])
    ax.tick_params(axis='y', which='both', length=0)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    plt.title(r"$\alpha = 0.8$, $\mathrm{T}_{\mathrm{MT}} = 60\mathrm{°C}$, $\Delta \mathrm{T}_{\mathrm{HT}} = 40\mathrm{K}$", loc='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'code/Figures/Case_Studies/LCOH_Breakdown_Case_2.pdf')
    plt.show()


###############################################################
## PART 7 : Cost as a function of elec/gas price ratio
###############################################################

elec_gas_price_ratio_EU = electricity_price_high / gas_price_high
elec_gas_price_ratio_US = electricity_price_low / gas_price_low

f_gas = interp1d([elec_gas_price_ratio_EU, elec_gas_price_ratio_US], [gas_price_high, gas_price_low], fill_value="extrapolate")
f_elec = interp1d([elec_gas_price_ratio_EU, elec_gas_price_ratio_US], [electricity_price_high, electricity_price_low], fill_value="extrapolate")

elec_gas_price_ratio_range = np.linspace(1, 6, 100)
electricity_price_range = f_elec(elec_gas_price_ratio_range)
gas_price_range = f_gas(elec_gas_price_ratio_range)

show_part_7 = True

if show_part_7 :

    plt.figure()
    plt.scatter(elec_gas_price_ratio_EU, electricity_price_high, color="red", label="Electricity price EU")
    plt.scatter(elec_gas_price_ratio_EU, gas_price_high, color="green", label="Gas price EU")
    plt.scatter(elec_gas_price_ratio_US, electricity_price_low, color="blue", label="Electricity price US")
    plt.scatter(elec_gas_price_ratio_US, gas_price_low, color="orange", label="Gas price US")
    plt.plot(elec_gas_price_ratio_range, electricity_price_range, color="black", linestyle="--")
    plt.plot(elec_gas_price_ratio_range, gas_price_range, color="black", linestyle="--")
    plt.xlabel(r'$\mathrm{Electricity}$ $\mathrm{price}$ / $\mathrm{Gas}$ $\mathrm{price}$  [-]', fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.set_title(r'Energy prices  [€/kWh]', loc='left', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 20))
    ax.spines['left'].set_position(('outward', 15))
    plt.tick_params(axis='x', rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=11, direction='in')
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.show()




""" TO BE COMPLETED """