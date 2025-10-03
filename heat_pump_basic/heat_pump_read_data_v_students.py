# -*- coding: utf-8 -*-
"""
===============================================================================
@ LELME2240 - Energy systems lab. 

Script for the processing of the data from the heat pump test bench.

-> Students version.
===============================================================================
- Date:     February 2022
- Author:   Antoine Laterre
===============================================================================
    Legend:
        HP  = Heat Pump
        EV  = Eau de Ville               (ENG: tap water)
        CS  = Cold Sink/Source/Side
        HS  = Hot  Sink/Source/Side
        SU  = Supply
        EX  = Exhaust
        BO  = Boiler
        ST  = Storage
        ECS = Eau Chaude Sanitaire       (ENG: DHW, Domestic Hot Water)
        EG  = Eau Glycolée               (ENG: glycol water)
        EC  = Eau de Chauffe
        LP  = Data from lab. probe
        TP  = Top
        MB  = Modbus
        SP  = Set Point
        AM  = Ambient
        EX  = External
    Colors in plots:
        # red   -> heat source
        # blue  -> heat sink
===============================================================================
"""
#%%
#===SPECIFY THE PATHS - TO BE MODIFIED=========================================
#

file        = 'test_24_03_24.xlsx'    # specify the results file name

#%%
#===IMPORT PACKAGES============================================================
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#===LOAD THE DATA==============================================================
#

# --- Extracting the data -----------------------------------------------------

data         = pd.read_excel(file)

# --- Time --------------------------------------------------------------------
time_old     = pd.DataFrame(data, columns=['Heure']).to_numpy()
time         = np.zeros(len(time_old))
for i, t in enumerate(time_old):    time[i] = (t[0].hour*60+t[0].minute)*60+t[0].second
time        -= time[0]                                                                    # [s]

# --- Mass flow rates ---------------------------------------------------------
m_eg_cs_su_lp   = pd.DataFrame(data, columns=['AI7 Flowmeter C1 ']).to_numpy()            # [l/min]
#   F301W	Flowmeter C1	             Flow sensor QVE2100.020 DN20	AI 7	l/s	4-20mA
m_ec_hs_su_lp   = pd.DataFrame(data, columns=['AI5 Flowmeter C2 ']).to_numpy()            # [l/min]
#   F302W	Flowmeter C2	             Flow sensor QVE2100.020 DN20	AI 5	l/s	4-20mA
m_ev_bo_ex_lp   = pd.DataFrame(data, columns=['AI4 Flowmeter ECS E.V. ']).to_numpy()      # [l/min]
#   F303W	Flowmeter ECS E.V.	       Flow sensor QVE2100.020 DN20	AI 4	l/s	4-20mA
m_ec_bo_su_lp   = pd.DataFrame(data, columns=['AI0 Flowmeter ECS PAC ']).to_numpy()       # [l/min]
#   F304W	Flowmeter ECS PAC	       Flow sensor QVE2100.020 DN20	AI 0	l/s	4-20mA
m_eg_st_su_lp   = pd.DataFrame(data, columns=['AI6 Flowmeter Stockage C1 ']).to_numpy()   # [l/min]
#   F305W	Flowmeter stockage C1	    Flow sensor QVE2100.020 DN20	AI 6	l/s	4-20mA

# --- Eau de ville temperatures -----------------------------------------------
T_ev_cs_su_lp   = pd.DataFrame(data, columns=['RTD1 AI0 Temp. IN EX1 E.V. ']).to_numpy()  # [°C]
#   T101	Temperature IN EX1 E.V.	PT100	RTD0	°C
T_ev_cs_ex_lp   = pd.DataFrame(data, columns=['RTD1 AI1 Temp. OUT EX1 E.V. ']).to_numpy() # [°C]
#   T102	Temperature OUT EX1 E.V.	PT100	RTD1	°C
T_ev_hs_su_lp   = pd.DataFrame(data, columns=['RTD1 AI7 Temp.IN EX2 E.V.']).to_numpy()    # [°C] 
#   T108	Temperature IN EX2 E.V.	PT100	RTD7	°C
T_ev_hs_ex_lp   = pd.DataFrame(data, columns=['RTD1 AI6 Temp. OUT EX2 E.V ']).to_numpy()  # [°C]
#   T107	Temperature OUT EX2 E.V.	PT100	RTD6	°C
T_ev_bo_su_lp   = pd.DataFrame(data, columns=['RTD2 AI1 Temp. IN ECS E.V. ']).to_numpy()  # [°C]
#   T110	Temperature IN ECS E.V. 	PT100	RTD1	°C
T_ev_bo_ex_lp   = pd.DataFrame(data, columns=['RTD2 AI0 Temp. OUT ECS E.V. ']).to_numpy() # [°C]
#   T109	Temperature OUT ECS E.V.	PT100	RTD0	°C

# --- Eau glycolée temperatures -----------------------------------------------
T_eg_cs_su_mb   = pd.DataFrame(data, columns=[' Com. Pac Mesure Temp. eau glycolée source froide(evaporateur in)']).to_numpy()      # [°C]
T_eg_cs_su_lp   = pd.DataFrame(data, columns=['RTD1 AI2 Temp. IN PAC C1 ']).to_numpy()                                              # [°C]
#   T103	Temperature IN PAC C1	    PT100	RTD2	°C
T_eg_cs_ex_mb   = pd.DataFrame(data, columns=['Com. Pac Mesure Temp. eau glycolée source froide(evaporateur out)']).to_numpy()      # [°C]
T_eg_cs_ex_lp   = pd.DataFrame(data, columns=['RTD1 AI3 Temp. OUT PAC C1 ']).to_numpy()                                             # [°C]
#   T104	Temperature OUT PAC C1	    PT100	RTD3	°C
T_ec_hs_su_mb   = pd.DataFrame(data, columns=['Com. Pac Mesure Temp. eau glycolée source chaude(condenseur out)']).to_numpy()       # [°C]
T_ec_hs_su_lp   = pd.DataFrame(data, columns=['RTD1 AI5 Temp. OUT EX2 PAC ']).to_numpy()                                            # [°C]
#   T105	Temperature IN EX2 PAC	    PT100	RTD4	°C
T_ec_hs_ex_mb   = pd.DataFrame(data, columns=['Com. Pac Mesure Temp. eau glycolée source chaude(condenseur in)']).to_numpy()        # [°C]
T_ec_hs_ex_lp   = pd.DataFrame(data, columns=['RTD1 AI4 Temp. IN EX2 PAC ']).to_numpy()                                             # [°C]
#   T106	Temperature OUT EX2 PAC	 PT100	RTD5	°C
T_ec_bo_su_mb   = pd.DataFrame(data, columns=[' Com. Pac Mesure Temp. eau glycolée boiler in']).to_numpy()                          # [°C]
T_ec_bo_su_lp   = pd.DataFrame(data, columns=['RTD2 AI3 Temp. OUT ECS PAC ']).to_numpy()                                            # [°C]
#   T112	Temperature OUT ECS PAC	 PT100	RTD3	°C
T_ec_bo_ex_lp   = pd.DataFrame(data, columns=['RTD2 AI2 Temp. IN ECS PAC ']).to_numpy()                                             # [°C]
#   T111	Temperature IN ECS PAC	    PT100	RTD2	°C
T_ec_bo_tp_mb   = pd.DataFrame(data, columns=['Com. Pac Mesure Temp. eau chaude boiler top']).to_numpy()                            # [°C]

T_sp_bo_eg_su  = pd.DataFrame(data, columns=[' Com. Pac Consigne Temp. eau glycolée boiler ']).to_numpy()                           # [°C]
T_sp_ht_eg_su  = T_sp_bo_eg_su
T_bo_eg_su_mb  = pd.DataFrame(data, columns=[' Com. Pac Mesure Temp. eau glycolée boiler in']).to_numpy()                           # [°C]
T_sp_bo_ecs_1  = pd.DataFrame(data, columns=[' Com. Pac Consigne Temp. ECS ']).to_numpy()                                           # [°C]
T_sp_bo_ecs_2  = pd.DataFrame(data, columns=[' Com. Pac Consigne ECS ']).to_numpy()                                                 # [°C]

T_sp_am        = pd.DataFrame(data, columns=[' Com. Pac Consigne Temp. Ambiante ']).to_numpy()                                      # [°C]
T_ex_mb        = pd.DataFrame(data, columns=[' Com. Pac Mesure Temp. Extérieur ']).to_numpy()                                       # [°C]

# --- Power ---------------------------------------------------------------

P = pd.DataFrame(data, columns=[' Com. PowerMeter Total Puissance Active']).to_numpy()  # [W]
Q = pd.DataFrame(data, columns=[' Com. PowerMeter Total Puissance Reactive']).to_numpy()  # [var]

 

#%%	
#===PLOT RAW DATA==============================================================
#
'''
# EX 1 - Eau de ville/Eau glycolée, cold side: temperatures -------------------
plt.figure('Temperature EX 1')
plt.plot(time/60,T_ev_cs_su_lp,color='crimson',linewidth=2)
plt.plot(time/60,T_ev_cs_ex_lp,color='orangered',linewidth=2)
plt.plot(time/60,T_eg_cs_ex_lp,color='navy',linewidth=2)
plt.plot(time/60,T_eg_cs_ex_mb,'--',color='navy',linewidth=1)
plt.plot(time/60,T_eg_cs_su_lp,color='dodgerblue',linewidth=2)
plt.plot(time/60,T_eg_cs_su_mb,'--',color='dodgerblue',linewidth=1)
plt.xlabel('time [min]')
plt.ylabel('temperature [°C]')
plt.legend(['Eau de ville IN (LP)','Eau de ville OUT (LP)','Eau glycolée IN (LP)','Eau glycolée IN (MB)','Eau glycolée OUT (LP)','Eau glycolée OUT (MB)'])
plt.title('EX 1: cold side heat exchanger')
plt.tight_layout()
'''

# HEAT PUMP - Eau glycolée/Eau chauffe, hot and cold sides: temperatures ------
plt.figure('Temperature HEAT PUMP')
plt.plot(time/60,T_eg_cs_su_lp,color='crimson',linewidth=2)
plt.plot(time/60,T_eg_cs_su_mb,'--',color='crimson',linewidth=1)
plt.plot(time/60,T_eg_cs_ex_lp,color='orangered',linewidth=2)
plt.plot(time/60,T_eg_cs_ex_mb,'--',color='orangered',linewidth=1)
plt.plot(time/60,T_ec_hs_su_lp,color='navy',linewidth=2)
plt.plot(time/60,T_ec_hs_su_mb,'--',color='navy',linewidth=1)
plt.plot(time/60,T_ec_hs_ex_lp,color='dodgerblue',linewidth=2)
plt.plot(time/60,T_ec_hs_ex_mb,'--',color='dodgerblue',linewidth=1)
plt.xlabel('time [min]')
plt.ylabel('temperature [°C]')
plt.legend(['Eau glycolée evap. IN (LP)','Eau glycolée evap. IN (MB)','Eau glycolée evap. OUT (LP)','Eau glycolée evap. OUT (MB)',
            'Eau chauffe cond. IN (LP)','Eau chauffe cond. IN (MB)','Eau chauffe cond. OUT (LP)','Eau chauffe cond. OUT (MB)'])
plt.title('HEAT PUMP: evaporator and condenser')
plt.tight_layout()

'''
# EX 2 - Eau de ville/Eau glycolée, hot side ----------------------------------
plt.figure('Temperature EX 2')
plt.plot(time/60,T_ec_hs_ex_lp,color='crimson',linewidth=2)
plt.plot(time/60,T_ec_hs_ex_mb,'--',color='crimson',linewidth=1)
plt.plot(time/60,T_ec_hs_su_lp,color='orangered',linewidth=2)
plt.plot(time/60,T_ec_hs_su_mb,'--',color='orangered',linewidth=1)
plt.plot(time/60,T_ev_hs_su_lp,color='navy',linewidth=2)
plt.plot(time/60,T_ev_hs_ex_lp,color='dodgerblue',linewidth=2)
plt.xlabel('time [min]')
plt.ylabel('temperature [°C]')
plt.legend(['Eau chauffe IN (LP)','Eau chauffe IN (MB)','Eau chauffe OUT (LP)','Eau chauffe OUT (MB)','Eau de ville IN (LP)','Eau de ville OUT (LP)'])
plt.title('EX 2: hot side heat exchanger')
plt.tight_layout()
'''

# CH - Control of the heaters pump --------------------------------------------
plt.figure('Temperature CH')
plt.plot(time/60,T_ec_hs_ex_lp,color='crimson',linewidth=2)
plt.plot(time/60,T_ec_hs_ex_mb,'--',color='crimson',linewidth=1)
plt.plot(time/60,T_sp_ht_eg_su,linestyle='dashdot',color='green',linewidth=2)
plt.plot(time/60,T_sp_am,'--',color='grey',linewidth=1.5)
plt.plot(time/60,T_ex_mb,'--',color='black',linewidth=1.5)
plt.xlabel('time [min]')
plt.ylabel('temperature [°C]')
plt.legend(['Eau chauffe départ (LP)','Eau chauffe départ (MB)','Consigne eau chauffe','Consigne température ambiante','Température extérieure'])
plt.title('CH: control of heaters')
plt.tight_layout()

# HEAT PUMP - Mass flow rates -------------------------------------------------
plt.figure('Mass flow rates HEAT PUMP')
plt.plot(time/60,m_eg_cs_su_lp,color='crimson',linewidth=2)
plt.plot(time/60,m_ec_hs_su_lp,color='navy',linewidth=2)
plt.xlabel('time [min]')
plt.ylabel('mass flow rate [l/min]')
plt.legend(['Eau glycolée evap. IN (LP)','Eau chauffe cond. OUT (LP)'])
plt.title('HEAT PUMP: mass flow rates')
plt.tight_layout()


# Power ---------------------------------------------------------------
plt.figure('Power')
plt.plot(time/60,P/1000,color='crimson',linewidth=2)
plt.plot(time/60,Q/1000,color='navy',linewidth=2)
plt.xlabel('time [min]')
plt.ylabel('power [kW]')
plt.legend(['Active power','Reactive power'])
plt.title('Power')
plt.tight_layout()

plt.show()