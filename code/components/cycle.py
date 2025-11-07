import CoolProp
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns

list_labels_exergy = [r'$P_{el,LT}$', r'$P_{el,MT}$',
                    r'$\dot{ \Delta E}_{ex,LT,wf}$', r'$\dot{ \Delta E}_{ex,MT,wf}$', r'$\dot{ \Delta E}_{ex,HT,wf}$',
                    r'$\dot{ \Delta E}_{ex,LT,water}$', r'$\dot{ \Delta E}_{ex,MT,water}$', r'$\dot{ \Delta E}_{ex,HT,water}$',
                    r'$P_{mec,comp,LT}$', r'$P_{mec,comp,MT}$',
                    r'$P_{irr,comp,LT}$', r'$P_{irr,comp,MT}$',
                    r'$P_{irr,adex,LT}$', r'$P_{irr,adex,MT}$',
                    r'$P_{irr,ex,LT}$', r'$P_{irr,ex,MT}$', r'$P_{irr,ex,HT}$',
                    r'$P_{irr,recup,LT}$', r'$P_{irr,recup,MT}$',
                    r'$P_{irr,mix}$']
list_colors_exergy_all = sns.color_palette("hls", len(list_labels_exergy))
dict_colors_exergy_all = {label: color for label, color in zip(list_labels_exergy, list_colors_exergy_all)}


class Cycle(): 

    def __init__(self, name):
        self.name = name

        # States 
        self.state_1 = None
        self.state_2 = None
        self.state_3 = None
        self.state_3_evap = None
        self.state_3_comp = None
        self.state_4 = None
        self.state_5 = None
        self.state_6 = None
        self.state_7 = None
        self.state_8 = None
        self.state_9 = None
        self.state_10 = None
        self.state_1_prime = None
        self.state_2_prime = None
        self.state_3_prime = None
        self.state_4_prime = None
        self.state_5_prime = None
        self.state_6_prime = None

        # Mass flow rates
        self.mdot_wf = None         # Working fluid mass flow rate [kg/s]
        self.mdot_wf_bottom = None  # Working fluid mass flow rate in bottom cycle [kg/s] (useful for dual-evaporator cycles)
        self.mdot_wf_top = None     # Working fluid mass flow rate in top cycle [kg/s] (useful for dual-evaporator cycles)
        self.mdot_LT = None         # Low temperature heat source mass flow rate [kg/s
        self.mdot_MT = None         # Medium temperature heat source mass flow rate [kg/s]
        self.mdot_HT = None         # High temperature heat source mass flow rate [kg/s]

        # Power of the compressor
        self.P_comp = None   # Compressor power [W]
        self.P_comp_bottom = None   # Compressor power in bottom cycle [W] (useful for dual-evaporator cycles)
        self.P_comp_top = None   # Compressor power in top cycle [W] (useful for dual-evaporator cycles)

        # List of transforms
        self.transforms = []

        # COP
        self.COP = None

        

    def __str__(self):
        # ──────────────────────────────────────────────
        # 1️⃣ Cycle title
        # ──────────────────────────────────────────────
        title = f"\n{'=' * 50}  Cycle: {self.name}  {'=' * 50}\n"

        # ──────────────────────────────────────────────
        # 2️⃣ Mass flow rates
        # ──────────────────────────────────────────────
        mass_flow_items = [
            ("mdot_wf", self.mdot_wf),
            ("mdot_LT", self.mdot_LT),
            ("mdot_MT", self.mdot_MT),
            ("mdot_HT", self.mdot_HT),
            ("mdot_wf_bottom", self.mdot_wf_bottom),
            ("mdot_wf_top", self.mdot_wf_top),
        ]
        mass_flow_filtered = [(n, v) for n, v in mass_flow_items if v is not None]

        if mass_flow_filtered:
            lines = [
                "+--------------------+--------------+",
                "| Mass Flow Rate     | Value [kg/s] |",
                "+--------------------+--------------+",
            ]
            for name, val in mass_flow_filtered:
                lines.append(f"| {name:<18} | {val:<12.4f} |")
            lines.append("+--------------------+--------------+")
            mass_flow_str = "\n".join(lines)
        else:
            mass_flow_str = "(No mass flow rates defined)"

        # ──────────────────────────────────────────────
        # 3️⃣ Formatting helper function
        # ──────────────────────────────────────────────
        def fmt(value, scale=1.0, unit_fmt="{:.2f}"):
            """Safely format numerical values with scaling and optional precision."""
            if value is None:
                return ""
            try:
                return unit_fmt.format(value / scale)
            except Exception:
                return ""

        # ──────────────────────────────────────────────
        # 4️⃣ Table of thermodynamic states
        # ──────────────────────────────────────────────
        state_header = (
            "+--------+--------------+--------------+-------------------+---------------------+----------------+-------------+"
        )
        header_rows = [
            state_header,
            "| State  | Temperature  | Pressure     | Enthalpy          | Entropy             | Exergy         | Quality     |",
            "|        | [K]          | [bar]        | [kJ/kg]           | [J/kg/K]            | [kJ/kg]        | [-]         |",
            state_header,
        ]

        rows = []

        # Helper to add a row if the state exists
        def add_state(label, state_obj):
            """Add a formatted row to the table for a given state object."""
            if state_obj is None:
                return
            T = getattr(state_obj, "T", None)
            if T is None:
                return

            p = getattr(state_obj, "p", None)
            h = getattr(state_obj, "h", None)
            s = getattr(state_obj, "s", None)
            e = getattr(state_obj, "e", None)
            Q = getattr(state_obj, "Q", None) or getattr(state_obj, "x", None)

            row = [
                label,
                fmt(T),
                fmt(p, scale=1e5),      # Convert Pa → bar
                fmt(h, scale=1000),     # Convert J/kg → kJ/kg
                fmt(s),
                fmt(e, scale=1000),     # Convert J/kg → kJ/kg
                fmt(Q, unit_fmt="{:.4f}") if Q is not None else "",
            ]
            rows.append(
                "| {0:<6} | {1:<12} | {2:<12} | {3:<17} | {4:<19} | {5:<14} | {6:<11} |".format(*row)
            )

        # Loop through normal states (1–10)
        for i in range(1, 11):
            state_obj = getattr(self, f"state_{i}", None)
            add_state(str(i), state_obj)

        # Loop through primed states (1'–6')
        for i in range(1, 7):
            state_obj = getattr(self, f"state_{i}_prime", None)
            add_state(f"{i}'", state_obj)

        rows.append(state_header)
        state_table_str = "\n".join(header_rows + rows)

        # ──────────────────────────────────────────────
        # 5️⃣ COP / Performance
        # ──────────────────────────────────────────────
        cop_val = self.COP
        if cop_val is None:
            cop_str = "(No COP defined)"
        else:
            cop_str = fmt(cop_val, unit_fmt="{:.3f}")

        performance_str = f"+----------------+-------+\n| Performance    | Value |\n+----------------+-------+\n| COP            | {cop_str:<5} |\n+----------------+-------+"

        # ──────────────────────────────────────────────
        # 6️⃣ Build final output string
        # ──────────────────────────────────────────────
        output = [
            title,
            "\nState Properties:\n",
            state_table_str,
            "\n\nMass Flow Rates:\n",
            mass_flow_str,
            "\n\nPerformance:\n",
            performance_str,
            f"\n{'=' * 118}\n",
        ]
        return "".join(output)
    
    def Ts_diagram(self, plot = True, n=100) : 
        # Generate saturation curve for working fluid

        T_points = []
        s_points = []
        states = {}
        labels_transform = []
        T_sat_state = {}
        
        for i, transform in enumerate(self.transforms):
            states[transform.label_in] = getattr(self, f"state_{transform.label_in}")
            states[transform.label_out] = getattr(self, f"state_{transform.label_out}")
            T_points_transform, s_points_transform = transform.get_points_between(states[transform.label_in], states[transform.label_out], n)[:2]
            T_points.append(T_points_transform)
            s_points.append(s_points_transform)
            labels_transform.append(transform.type)
            if transform.label_out_secondary == '7' : 
                T_points_transform, s_points_transform = transform.get_points_between(states[transform.label_in_secondary], states[transform.label_out_secondary], n)[:2]
                T_points.append(T_points_transform)
                s_points.append(s_points_transform)
                labels_transform.append(transform.type)
            if transform.label_out_secondary == '9' : 
                T_points_transform, s_points_transform = transform.get_points_between(states[transform.label_in_secondary], states[transform.label_out_secondary], n)[:2]
                T_points.append(T_points_transform)
                s_points.append(s_points_transform)
                labels_transform.append(transform.type)    

        T_points = np.array(T_points)
        s_points =  np.array(s_points)
        
        heos = CoolProp.AbstractState("HEOS", list(states.values())[0].fluid)
        T_crit = heos.T_critical()
        p_crit = heos.p_critical()
        for key, value in states.items() :
            if key in ['1', '3', '5'] :
                if p_crit >= value.p :
                    heos.update(CoolProp.PQ_INPUTS, value.p, 1)
                    T_sat_state[key] = heos.T()
        T_sat_state = list(T_sat_state.values())

        plt.figure(figsize=(8,6))

        s_states = np.zeros(len(states))
        T_states = np.zeros(len(states))
        for i, (label, state) in enumerate(states.items()):
            s_states[i] = state.s
            T_states[i] = state.T
            #plt.text(state.s/1e3, state.T-273.15, f"{label}", fontsize=9)

        s_max_state = max(s_states)
        s_min_state = min(s_states)
        T_max_state = max(T_states)
        T_min_state = min(T_states)

        heos.update(CoolProp.QT_INPUTS, 0, T_min_state)
        s_min_liq = heos.smass()
        heos.update(CoolProp.QT_INPUTS, 1, T_min_state)
        #s_max_vap = heos.smass()

        plt.xlim((min(s_min_state/1e3, s_min_liq/1e3), s_max_state/1e3))
        plt.ylim((T_min_state-273.15, max((T_max_state-273.15, T_crit-273.15))))

        T_sat = np.linspace(T_min_state, T_crit - 1, 100)
        s_liq = np.zeros(100)
        s_vap = np.zeros(100)
        for i, T in enumerate(T_sat):
            heos.update(CoolProp.QT_INPUTS, 0, T)
            s_liq[i] = heos.smass()
            heos.update(CoolProp.QT_INPUTS, 1,T)
            s_vap[i] = heos.smass()

        plt.plot(s_liq/1e3, T_sat-273.15, 'black', label='Saturated Liquid', clip_on = False)
        plt.plot(s_vap/1e3, T_sat-273.15, 'black', label='Saturated Vapor', clip_on = False)
        heos.update(CoolProp.QT_INPUTS, 0.5, T_crit)
        s_crit = heos.smass()
        plt.scatter(s_crit/1e3, T_crit-273.15, color='black', s=10, clip_on = False)  # Triple point

        plt.plot(s_points.T/1e3, T_points.T-273.15, '-', label=labels_transform, color = 'firebrick', clip_on = False)
        plt.scatter(s_states/1e3, T_states-273.15, color='firebrick', clip_on = False)

        plt.xlabel('Entropy [kJ/kg/K]', fontsize = 12)
        #plt.legend(frameon=False)


        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.set_title('Temperature [°C]', loc='left', fontsize=12)

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move bottom and left spines away
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))

        # Show only axis limit values for entropy (s). Keep T state ticks but ensure limits included.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # X ticks: only the axis limits (no per-state s values)
        xticks = np.array([xlim[0], xlim[1]])

        # Y ticks: use state T values if available, but ensure axis limits are included
        yticks = []
        if T_max_state - 273.15 < ylim[1]:
            yticks = np.concatenate([np.array(T_sat_state) - 273.15, np.array([T_max_state - 273.15, ylim[1]])])
        else : 
            yticks = np.concatenate([np.array(T_sat_state) - 273.15, np.array([ylim[1]])])


        # Apply ticks and formatted labels (s in kJ/kg/K, T in °C)
        xticks = np.sort(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v:.3f}" for v in xticks])

        
        yticks = np.sort(yticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.1f}" for v in yticks])

        # Improve readability
        plt.tick_params(axis='x', rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

        #plt.tight_layout()
        fig_dir = f'code/Figures/{self.name}'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}/Ts_diagram.png', dpi=600)
        if plot == True : plt.show()
        return
    
    def ph_diagram(self, plot = True, n=100) :
        # Generate saturation curve for working fluid

        p_points = []
        h_points = []
        states = {}
        labels_transform = []
        p_sat_state = {}
        
        for i, transform in enumerate(self.transforms):
            states[transform.label_in] = getattr(self, f"state_{transform.label_in}")
            states[transform.label_out] = getattr(self, f"state_{transform.label_out}")
            p_points_transform, h_points_transform = transform.get_points_between(states[transform.label_in], states[transform.label_out], n)[2:]
            p_points.append(p_points_transform)
            h_points.append(h_points_transform)
            labels_transform.append(transform.type)
            if transform.label_out_secondary == '7' : 
                p_points_transform, h_points_transform = transform.get_points_between(states[transform.label_in_secondary], states[transform.label_out_secondary], n)[2:]
                p_points.append(p_points_transform)
                h_points.append(h_points_transform)
                labels_transform.append(transform.type)
            if transform.label_out_secondary == '9' : 
                p_points_transform, h_points_transform = transform.get_points_between(states[transform.label_in_secondary], states[transform.label_out_secondary], n)[2:]
                p_points.append(p_points_transform)
                h_points.append(h_points_transform)
                labels_transform.append(transform.type)    

        p_points = np.array(p_points)
        h_points =  np.array(h_points)
        heos = CoolProp.AbstractState("HEOS", list(states.values())[0].fluid)

        p_states = np.zeros(len(states))
        h_states = np.zeros(len(states))
        for i, (label, state) in enumerate(states.items()):
            p_states[i] = state.p
            h_states[i] = state.h
            #plt.text(state.h/1e3, state.p/1e5, f"{label}", fontsize=9)
        

        h_max_state = max(h_states)
        h_min_state = min(h_states)
        p_max_state = max(p_states)
        p_min_state = min(p_states)

        heos.update(CoolProp.PQ_INPUTS, p_min_state, 0)
        h_min_liq = heos.hmass()
        heos.update(CoolProp.PQ_INPUTS, p_min_state, 1)

        plt.figure(figsize=(8,6))
        plt.plot(h_points.T/1e3, p_points.T/1e5, '-', label=labels_transform, color = 'firebrick', clip_on = False)
        plt.scatter(h_states/1e3, p_states/1e5, color = 'firebrick', clip_on = False)
        plt.yscale('log')

        p_crit = heos.p_critical()
        for key, value in states.items() :
            if key in ['1', '3', '5'] :
                if p_crit >= value.p :
                    p_sat_state[key] = value.p
        p_sat_state = list(p_sat_state.values())


        plt.xlim((min(h_min_state/1e3, h_min_liq/1e3), h_max_state/1e3))
        plt.ylim((p_min_state/1e5, max((p_max_state, p_crit))/1e5))


        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.set_title('Pressure [bar]', loc='left', fontsize=12)

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move bottom and left spines away
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))

        # Disable automatic tick locator
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_minor_locator(plt.NullLocator())

        # Show only axis limit values for entropy (s). Keep T state ticks but ensure limits included.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # X ticks: only the axis limits (no per-state s values)
        xticks = np.array([xlim[0], xlim[1]])

        # Y ticks: use state T values if available, but ensure axis limits are included
        yticks = []
        if p_max_state/1e5 < ylim[1]:
            yticks = np.concatenate([np.array(p_sat_state)/1e5, np.array([p_max_state/1e5, ylim[1]])])
        else : 
            yticks = np.concatenate([np.array(p_sat_state)/1e5, np.array([ylim[1]])])


        # Apply ticks and formatted labels (s in kJ/kg/K, T in °C)
        xticks = np.sort(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v:.1f}" for v in xticks])

        
        yticks = np.sort(yticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.2f}" for v in yticks])

        # Improve readability
        plt.tick_params(axis='x', rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

        p_sat = np.linspace(ylim[0]*1e5, p_crit - 1e4, 100)
        h_liq = np.zeros(100)
        h_vap = np.zeros(100)
        for i, p in enumerate(p_sat):
            heos.update(CoolProp.PQ_INPUTS, p, 0)
            h_liq[i] = heos.hmass()
            heos.update(CoolProp.PQ_INPUTS, p, 1)
            h_vap[i] = heos.hmass()

        plt.plot(h_liq/1e3, p_sat/1e5, 'black', label='Saturated Liquid', clip_on = False)
        plt.plot(h_vap/1e3, p_sat/1e5, 'black', label='Saturated Vapor', clip_on = False)
        heos.update(CoolProp.PQ_INPUTS, p_crit, 0.5)
        plt.scatter(heos.hmass()/1e3, p_crit/1e5, color='black', clip_on=False, s = 15)

        #plt.tight_layout()
        fig_dir = f'code/Figures/{self.name}'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}/ph_diagram.png', dpi=600)
        if plot == True : plt.show()
        return

    



        plt.show()


    def energy_chart(self, plot=True) :
        if self.mdot_wf_bottom is None : 
            self.mdot_wf_bottom = 0
        if self.mdot_wf_top is None :
            self.mdot_wf_top = 0

        dict_received = {}
        dict_delivered = {}
        
        for transform in self.transforms :
            state_in = getattr(self, f"state_{transform.label_in}")
            state_out = getattr(self, f"state_{transform.label_out}")
            if transform.type == 'comp' :
                if transform.label_in in ['1', '2'] :
                    mdot_wf = self.mdot_wf_bottom
                    P_comp = self.P_comp_bottom
                    args = {'P_el' : P_comp, 'mdot_wf' : mdot_wf}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_received[r'$P_{el,LT}$'] = energies['P_{el}']
                    dict_delivered[r'$P_{mec,comp,LT}$'] = energies['P_{loss}']
                else :
                    mdot_wf = self.mdot_wf_top
                    P_comp = self.P_comp_top
                    args = {'P_el' : P_comp, 'mdot_wf' : mdot_wf}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_received[r'$P_{el,MT}$'] = energies['P_{el}']
                    dict_delivered[r'$P_{mec,comp,MT}$'] = energies['P_{loss}']

                
            elif transform.type == 'hex' :
                state_in_secondary = getattr(self, f"state_{transform.label_in_secondary}")
                state_out_secondary = getattr(self, f"state_{transform.label_out_secondary}")
                
                if transform.label_in_secondary in ['1_prime', '2_prime'] : 
                    mdot_secondary = self.mdot_LT
                    mdot_wf = self.mdot_wf_bottom
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_received[r'$\dot Q_{LT}$'] = energies['P_{wf}']
                    dict_delivered[r'$P_{L,LT}$'] = energies['P_{loss}']

                elif transform.label_in_secondary in ['4_prime', '3_prime'] : 
                    mdot_secondary = self.mdot_MT
                    mdot_wf = abs(self.mdot_wf_top - self.mdot_wf_bottom)
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)

                    
                    if energies['P_{wf}'] >= 0 :
                        dict_received[r'$\dot Q_{MT}$'] = energies['P_{wf}']
                        dict_delivered[r'$P_{L,MT}$'] = energies['P_{loss}']
                    else : 
                        dict_delivered[r'$\dot Q_{MT}$'] = energies['P_{secondary}']
                        dict_delivered[r'$P_{L,MT}$'] = energies['P_{loss}']

                elif transform.label_in_secondary in ['5_prime', '6_prime'] :
                    mdot_secondary = self.mdot_HT
                    mdot_wf = self.mdot_wf_top
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_delivered[r'$\dot Q_{HT}$'] = energies['P_{secondary}']
                    dict_delivered[r'$P_{L,HT}$'] = energies['P_{loss}']

                elif transform.label_in_secondary in ['1', '2', '7', '8', '9'] :
                    mdot_secondary = self.mdot_wf_bottom
                    mdot_wf = self.mdot_wf_bottom
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_delivered[r'$P_{L,recup,LT}$'] = energies['P_{loss}']
                elif transform.label_in_secondary in ['3', '4', '6', '7'] :
                    mdot_secondary = abs(self.mdot_wf_top - self.mdot_wf_bottom)
                    mdot_wf = abs(self.mdot_wf_top - self.mdot_wf_bottom)
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_delivered[r'$P_{L,recup,MT}$'] = energies['P_{loss}']
                else : 
                    raise ValueError("Unknown secondary mass flow rate for energy analysis.")
        
        # Plot the energy chart

        plt.figure(figsize=(9,6))
        y_pos = np.arange(2)
        for dic, i in zip([dict_received, dict_delivered], y_pos):
            x_tot = 0
            # sort items by decreasing value (original units)
            for key, value in sorted(dic.items(), key=lambda kv: kv[1], reverse=True):
                label = key
                value = value / 1e3  # Convert to kW
                if value > 1e-3:
                    plt.barh(y_pos[i], value, label=label, left=x_tot)
                    x_tot += value
                else :
                    del dic[key]


        plt.xlabel('Power [kW]', fontsize = 12)

        length = len(dict_received) + len(dict_delivered)
        n_cols = length // 3 if length % 3 == 0 else (length // 3) + 1
        
        plt.legend(frameon=False,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.95),
           ncol=n_cols)  # Adjust ncol based on number of legend items



        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Energy \n Inputs', 'Energy \n Outputs'])
        # hide tick marks (the small lines)
        ax.tick_params(axis='y', which='both', length=0)
        ax.invert_yaxis()  # labels read top-to-bottom

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Move bottom and left spines away
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))

        #plt.tight_layout()
        plt.xlim((0, sum(dict_received.values())/1e3))
        plt.xticks((0, sum(dict_received.values())/1e3), [f"", f"{sum(dict_received.values())/1e3:.1f}"])
        fig_dir = f'code/Figures/{self.name}'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}/energy_chart.png', dpi=600)
        if plot == True : plt.show()

        return 
    
    def exergy_chart(self, T0, p0, plot = True, losses = 'per_type') :
        if self.mdot_wf_bottom is None : 
            self.mdot_wf_bottom = 0
        if self.mdot_wf_top is None :
            self.mdot_wf_top = 0
        dict_received = {}
        dict_delivered = {}
        for transform in self.transforms :
            state_in = getattr(self, f"state_{transform.label_in}")
            state_out = getattr(self, f"state_{transform.label_out}")
            if transform.type == 'comp' :
                if transform.label_in in ['1', '2'] :
                    mdot_wf = self.mdot_wf_bottom
                    P_comp = self.P_comp_bottom
                    args = {'P_el' : P_comp, 'mdot_wf' : mdot_wf}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    dict_received[r'$P_{el,LT}$'] = exergies['P_{el}']
                    dict_delivered[r'$P_{mec,comp,LT}$'] = exergies['P_{loss}']
                    dict_delivered[r'$P_{irr,comp,LT}$'] = exergies['P_{irr}']

                else :
                    mdot_wf = self.mdot_wf_top
                    P_comp = self.P_comp_top
                    args = {'P_el' : P_comp, 'mdot_wf' : mdot_wf}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    dict_received[r'$P_{el,MT}$'] = exergies['P_{el}']
                    dict_delivered[r'$P_{mec,comp,MT}$'] = exergies['P_{loss}']
                    dict_delivered[r'$P_{irr,comp,MT}$'] = exergies['P_{irr}']
            elif transform.type == 'isobaric_mixing' : 
                state_in = [state_in, state_out]
                state_out = getattr(self, f"state_3")
                if transform.label_in == '3_evap' :
                    mdot = [abs(self.mdot_wf_bottom - self.mdot_wf_top), self.mdot_wf_bottom]
                else : 
                    mdot = [self.mdot_wf_bottom, abs(self.mdot_wf_bottom - self.mdot_wf_top)]
                args = {'mdot_wf' : mdot}
                exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                dict_delivered[r'$P_{irr,mix}$'] = exergies['P_{irr}']

            elif transform.type == 'adex' :
                if transform.label_in == '9' :
                    mdot_wf = self.mdot_wf_bottom
                    args = {'mdot_wf' : mdot_wf}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    dict_delivered[r'$P_{irr,adex,MT}$'] = exergies['P_{irr}']
                elif transform.label_in == '7' :
                    mdot_wf = abs(self.mdot_wf_top - self.mdot_wf_bottom)
                    args = {'mdot_wf' : mdot_wf}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    dict_delivered[r'$P_{irr,adex,LT}$'] = exergies['P_{irr}']

            elif transform.type == 'hex' :
                state_in_secondary = getattr(self, f"state_{transform.label_in_secondary}")
                state_out_secondary = getattr(self, f"state_{transform.label_out_secondary}")
                
                if transform.label_in_secondary in ['1_prime', '2_prime'] : 
                    mdot_secondary = self.mdot_LT
                    mdot_wf = self.mdot_wf_bottom
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    if abs(exergies['P_{wf}']) > abs(exergies['P_{secondary}']) :
                        dict_delivered[r'$\dot{ \Delta E}_{ex,LT,water}$'] = abs(exergies['P_{secondary}'])
                    else : 
                        dict_received[r'$\dot{ \Delta E}_{ex,LT,wf}$'] = abs(exergies['P_{secondary}'])

                    dict_delivered[r'$P_{irr,ex,LT}$'] = exergies['P_{irr}']

                elif transform.label_in_secondary in ['4_prime', '3_prime'] : 
                    mdot_secondary = self.mdot_MT
                    mdot_wf = abs(self.mdot_wf_top - self.mdot_wf_bottom)
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)

                    if abs(exergies['P_{wf}']) > abs(exergies['P_{secondary}']) :
                        dict_delivered[r'$\dot{ \Delta E}_{ex,MT,water}$'] = abs(exergies['P_{secondary}'])
                    else :
                        dict_received[r'$\dot{ \Delta E}_{ex,MT,wf}$'] = abs(exergies['P_{secondary}'])
                    dict_delivered[r'$P_{irr,ex,MT}$'] = exergies['P_{irr}']

                elif transform.label_in_secondary in ['5_prime', '6_prime'] :
                    mdot_secondary = self.mdot_HT
                    mdot_wf = self.mdot_wf_top
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    if abs(exergies['P_{wf}']) > abs(exergies['P_{secondary}']) :
                        dict_delivered[r'$\dot{ \Delta E}_{ex,HT,water}$'] = abs(exergies['P_{secondary}'])
                    else :
                        dict_received[r'$\dot{ \Delta E}_{ex,HT, wf}$'] = abs(exergies['P_{secondary}'])
                    dict_delivered[r'$P_{irr,ex,HT}$'] = exergies['P_{irr}']

                elif transform.label_in_secondary in ['1', '2', '7', '8', '9'] :
                    mdot_secondary = self.mdot_wf_bottom
                    mdot_wf = self.mdot_wf_bottom
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    dict_delivered[r'$P_{irr,recup,LT}$'] = exergies['P_{irr}']

                elif transform.label_in_secondary in ['3', '4', '6', '7'] :
                    mdot_secondary = self.mdot_wf_top
                    mdot_wf = self.mdot_wf_top
                    args = {'mdot_wf' : mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    dict_delivered[r'$P_{irr,recup,MT}$'] = exergies['P_{irr}']
                else : 
                    raise ValueError("Unknown secondary mass flow rate for energy analysis.")

        # Plot the energy chart

        if losses == 'all' :
            pass
        elif losses == 'per_type' :
            losses_compressors = []
            losses_exchangers = []
            losses_adex = []
            losses_recup = []
            losses_mec = []
            to_delete = []
            for key in dict_delivered.keys() :
                if 'irr,comp' in key :
                    losses_compressors.append(dict_delivered[key])
                    to_delete.append(key)
                elif 'irr,ex' in key  :
                    losses_exchangers.append(dict_delivered[key])
                    to_delete.append(key)
                elif 'irr,adex' in key :
                    losses_adex.append(dict_delivered[key])
                    to_delete.append(key)
                elif 'irr,recup' in key :
                    losses_recup.append(dict_delivered[key])
                    to_delete.append(key)
                elif 'mec' in key :
                    losses_mec.append(dict_delivered[key])
                    to_delete.append(key)
            for key in to_delete :
                del dict_delivered[key]
            if len(losses_compressors) > 0 :
                dict_delivered[r'$P_{irr,comp,tot}$'] = sum(losses_compressors)
            if len(losses_exchangers) > 0 :
                dict_delivered[r'$P_{irr,ex,tot}$'] = sum(losses_exchangers)
            if len(losses_adex) > 0 :
                dict_delivered[r'$P_{irr,adex,tot}$'] = sum(losses_adex)
            if len(losses_recup) > 0 :
                dict_delivered[r'$P_{irr,recup,tot}$'] = sum(losses_recup)
            if len(losses_mec) > 0 :
                dict_delivered[r'$P_{mec,tot}$'] = sum(losses_mec)
        elif losses == 'biggest' :
            pass 

        plt.figure(figsize=(9,6))
        
        y_pos = np.arange(2)
        for dic, i in zip([dict_received, dict_delivered], y_pos):
            x_tot = 0
            # sort items by decreasing value (original units)
            for key, value in sorted(dic.items(), key=lambda kv: kv[1], reverse=True):
                label = key
                value = value / 1e3  # Convert to kW
                if value > 1e-3:
                    plt.barh(y_pos[i], value, label=label, left=x_tot)
                    x_tot += value


        plt.xlabel('Power [kW]', fontsize = 12)
        #plt.legend(frameon=False, loc = 'lower right')

        length = len(dict_received) + len(dict_delivered)
        n_cols = length // 3 if length % 3 == 0 else (length // 3) + 1

        plt.legend(frameon=False,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.95),
           ncol=n_cols)  # Adjust ncol based on number of legend items


        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Exergy \n Inputs', 'Exergy \n Outputs'])
        # hide tick marks (the small lines)
        ax.tick_params(axis='y', which='both', length=0)
        ax.invert_yaxis()  # labels read top-to-bottom

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Move bottom and left spines away
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))

        #plt.tight_layout()
        plt.xlim((0, sum(dict_received.values())/1e3))
        plt.xticks((0, sum(dict_received.values())/1e3), [f"", f"{sum(dict_received.values())/1e3:.1f}"])
        '''
        fig, (ax1, ax2) = plt.subplots(figsize=(10,6), nrows=1, ncols=2)
        total = sum(dict_received.values())

        # Create the second subplot
        dict_delivered = dict(sorted(dict_delivered.items(), key=lambda item: item[1], reverse=True))
        # Plot horizontal bars (convert to kW) and annotate values to the right of each bar
        vals_perc = [v/total * 100 for v in dict_delivered.values()]
        bars = ax2.barh(list(dict_delivered.keys()), vals_perc)

        # Padding to place the text slightly to the right of the bar
        pad = (max(vals_perc) * 0.01) if len(vals_perc) and max(vals_perc) > 0 else 0.01

        for bar in bars:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax2.text(width + pad, y, f"{width:.1f} %", va='center', fontsize=11)
        ax2.set_title('Exergy Outputs')
        

        # Hide top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        ax2.set_xticks([])  # Hide x-axis ticks
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_xlim(0, 100)
        ax2.tick_params(axis='y', which='both', length=0)


        # Create the first subplot
        dict_received = dict(sorted(dict_received.items(), key=lambda item: item[1], reverse=True))
        # Plot horizontal bars (convert to kW) and annotate values to the right of each bar
        vals_perc = [v/total * 100 for v in dict_received.values()]
        bars = ax1.barh(list(dict_received.keys()), vals_perc)

        # Padding to place the text slightly to the right of the bar
        pad = (max(vals_perc) * 0.01) if len(vals_perc) and max(vals_perc) > 0 else 0.01

        for bar in bars:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax1.text(width + pad, y, f"{width:.1f} %", va='center', fontsize=11)
        ax1.set_title('Exergy Inputs')
        # Hide top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        ax1.set_xticks([])  # Hide x-axis ticks
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlim(0, 100)
        ax1.tick_params(axis='y', which='both', length=0)
        '''

        fig_dir = f'code/Figures/{self.name}'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}/exergy_chart.png', dpi=600)
        if plot == True : plt.show()

        return 
            

        






'''
cycle = Cycle("TwoStage_R290")

# Example of state objects (each one could be an instance of your State class)
state1 = State(T=280, p=4e5, fluid='R290')
state2 = State(T=320, p=1e6, fluid='R290')

cycle.state_1 = state1
cycle.state_2 = state2
cycle.mdot_wf = 0.25
cycle.mdot_HT = 0.12

print(cycle)
'''
