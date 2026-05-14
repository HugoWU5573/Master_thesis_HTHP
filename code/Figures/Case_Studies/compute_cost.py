
#########################################################
## PART 1 : Input data for the method
#########################################################

# Data from Ommen et al. (2015) Technical and economic working domains of industrial heat pumps

inflation_correction_factor = 1.3  # Correction factor to account for inflation from 2015 to 2026

PEC_Compressor = 19850*inflation_correction_factor
X_Compressor = 279.8
alpha_Compressor = 0.73

PEC_Heat_Exchanger = 15526*inflation_correction_factor
X_Heat_Exchanger = 42
alpha_Heat_Exchanger = 0.8


#########################################################
## PART 2 : Data for the dual-evaporator case
#########################################################

LP_comp_flow_rate_dual = 10.0326
HP_comp_flow_rate_dual = 8.4678

GasCooler_Area_dual = 3.306
Evaporator_LT_Area_dual = 0.968
Evaporator_MT_Area_dual = 0.768
Recuperator_LT_Area_dual = 0.272
Recuperator_MT_Area_dual = 0.441

PEC_total_dual = PEC_Compressor * (LP_comp_flow_rate_dual/X_Compressor)**alpha_Compressor \
                + PEC_Compressor * (HP_comp_flow_rate_dual/X_Compressor)**alpha_Compressor \
                + PEC_Heat_Exchanger * (GasCooler_Area_dual/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Evaporator_LT_Area_dual/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Evaporator_MT_Area_dual/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Recuperator_LT_Area_dual/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Recuperator_MT_Area_dual/X_Heat_Exchanger)**alpha_Heat_Exchanger

TCI_total_dual = PEC_total_dual * 4.16
TCI_dual_per_kW = TCI_total_dual / 25  # [€/kW]

print(f"Total Capital Investment for the dual-evaporator case: {TCI_dual_per_kW:.2f} €/kW")

#########################################################
## PART 3 : Data for the single-evaporator case
#########################################################

LP_comp_flow_rate_single_1 = 12.3278
HP_comp_flow_rate_single_1 = 6.5916
HP_comp_flow_rate_single_2 = 3.6114

GasCooler_Area_single_1 = 1.054
GasCooler_Area_single_2 = 1.413

Evaporator_Area_single_1 = 0.874
Evaporator_Area_single_2 = 0.761

Recuperator_Area_single_1 = 0.329
Recuperator_Area_single_2 = 0.163

PEC_total_single = PEC_Compressor * (LP_comp_flow_rate_single_1/X_Compressor)**alpha_Compressor \
                + PEC_Compressor * (HP_comp_flow_rate_single_1/X_Compressor)**alpha_Compressor \
                + PEC_Compressor * (HP_comp_flow_rate_single_2/X_Compressor)**alpha_Compressor \
                + PEC_Heat_Exchanger * (GasCooler_Area_single_1/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (GasCooler_Area_single_2/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Evaporator_Area_single_1/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Evaporator_Area_single_2/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Recuperator_Area_single_1/X_Heat_Exchanger)**alpha_Heat_Exchanger \
                + PEC_Heat_Exchanger * (Recuperator_Area_single_2/X_Heat_Exchanger)**alpha_Heat_Exchanger

TCI_total_single = PEC_total_single * 4.16
TCI_single_per_kW = TCI_total_single / 25  # [€/kW]

print(f"Total Capital Investment for the single-evaporator case: {TCI_single_per_kW:.2f} €/kW")

#########################################################
## PART 4 : Compute the ratio between the two cases
#########################################################

ratio_TCI_per_kW = TCI_dual_per_kW / TCI_single_per_kW
gain = (1 - ratio_TCI_per_kW) * 100

print(f"Cost reduction of the dual-evaporator case compared to the single-evaporator case: {gain:.2f} %")