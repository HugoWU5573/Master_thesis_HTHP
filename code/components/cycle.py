import CoolProp
from matplotlib import pyplot as plt
import numpy as np

class Cycle(): 

    def __init__(self, name):
        self.name = name

        # States 
        self.state_1 = None
        self.state_2 = None
        self.state_3 = None
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
        self.mdot_wf = None  # Working fluid mass flow rate [kg/s]
        self.mdot_LT = None  # Low temperature heat source mass flow rate [kg/s
        self.mdot_MT = None  # Medium temperature heat source mass flow rate [kg/s]
        self.mdot_HT = None  # High temperature heat source mass flow rate [kg/s]

        # Power of the compressor
        self.P_comp = None   # Compressor power [W]

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
        ]
        mass_flow_filtered = [(n, v) for n, v in mass_flow_items if v is not None]

        if mass_flow_filtered:
            lines = [
                "+----------------+--------------+",
                "| Mass Flow Rate | Value [kg/s] |",
                "+----------------+--------------+",
            ]
            for name, val in mass_flow_filtered:
                lines.append(f"| {name:<14} | {val:<12.4f} |")
            lines.append("+----------------+--------------+")
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
            "|        | [K]          | [bar]        | [kJ/kg]           | [J/kg/K]            | [J/kg]         | [-]         |",
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
                fmt(e),
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

        T_points = np.zeros((len(self.transforms), n))
        s_points = np.zeros((len(self.transforms), n))
        states = {}
        labels_transorm = []
        
        for i, transform in enumerate(self.transforms):
            states[transform.label_in] = getattr(self, f"state_{transform.label_in}")
            states[transform.label_out] = getattr(self, f"state_{transform.label_out}")
            T_points[i, :], s_points[i, :] = transform.get_points_between(states[transform.label_in], states[transform.label_out], n)
            labels_transorm.append(transform.type)
        
        heos = CoolProp.AbstractState("HEOS", list(states.values())[0].fluid)
        T_crit = heos.T_critical() - 1
        
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

        T_sat = np.linspace(T_min_state, T_crit, 100)
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

        plt.plot(s_points.T/1e3, T_points.T-273.15, '-', label=labels_transorm, color = 'firebrick', clip_on = False)
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
        yticks = np.unique(T_states - 273.15) if T_states.size else np.array([])
        if yticks.size:
            yticks = yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]
            tol = 1e-9
            to_add = []
            if yticks.size == 0 or (ylim[0] < yticks.min() - tol):
                to_add.append(ylim[0])
            if yticks.size == 0 or (ylim[1] > yticks.max() + tol):
                to_add.append(ylim[1])
            if to_add:
                yticks = np.unique(np.concatenate([yticks, np.array(to_add)]))
        else:
            yticks = np.array([ylim[0], ylim[1]])

        # Apply ticks and formatted labels (s in kJ/kg/K, T in °C)
        xticks = np.sort(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v:.3f}" for v in xticks])

        if yticks.size:
            yticks = np.sort(yticks)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.1f}" for v in yticks])

        # Improve readability
        plt.tick_params(axis='x', rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=11, direction='in')

        #plt.tight_layout()
        plt.savefig('code/Figures/' + self.name + '_Ts_diagram.png', dpi=300)
        if plot == True : plt.show()
        return


    def energy_chart(self, plot=True) :

        dict_received = {}
        dict_delivered = {}
        
        for transform in self.transforms :
            state_in = getattr(self, f"state_{transform.label_in}")
            state_out = getattr(self, f"state_{transform.label_out}")
            if transform.type == 'comp' :
                args = {'P_el' : self.P_comp, 'mdot_wf' : self.mdot_wf}
                energies = transform.energy_analysis(state_in, state_out, args)
                dict_received[r'$P_{el}$'] = energies['P_{el}']
                dict_delivered[r'$P_{L,comp}$'] = energies['P_{loss}']

            elif transform.type == 'evap' or transform.type == 'cond' :
                state_in_secondary = getattr(self, f"state_{transform.label_in_secondary}")
                state_out_secondary = getattr(self, f"state_{transform.label_out_secondary}")
                
                if transform.label_in_secondary in ['1_prime', '2_prime'] : 
                    mdot_secondary = self.mdot_LT
                    args = {'mdot_wf' : self.mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_received[r'$\dot Q_{LT}$'] = energies['P_{wf}']
                    dict_delivered[r'$P_{L,LT}$'] = energies['P_{loss}']

                elif transform.label_in_secondary in ['4_prime', '3_prime'] : 
                    mdot_secondary = self.mdot_MT
                    args = {'mdot_wf' : self.mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    if transform.type == 'cond' :
                        dict_delivered[r'$\dot Q_{MT}$'] = energies['P_{secondary}']
                        dict_delivered[r'$P_{L,MT}$'] = energies['P_{loss}']
                    else :
                        dict_received[r'$\dot Q_{MT}$'] = energies['P_{wf}']
                        dict_delivered[r'$P_{L,MT}$'] = energies['P_{loss}']

                elif transform.label_in_secondary in ['5_prime', '6_prime'] :
                    mdot_secondary = self.mdot_HT
                    args = {'mdot_wf' : self.mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    energies = transform.energy_analysis(state_in, state_out, args)
                    dict_delivered[r'$\dot Q_{HT}$'] = energies['P_{secondary}']
                    dict_delivered[r'$P_{L,HT}$'] = energies['P_{loss}']
                else : 
                    raise ValueError("Unknown secondary mass flow rate for energy analysis.")

        # Plot the energy chart

        plt.figure(figsize=(8,6))
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
        n_cols = length // 2 if length % 2 == 0 else (length // 2) + 1
        
        plt.legend(frameon=False,
           loc='lower center',
           bbox_to_anchor=(0.5, 1),
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
        plt.savefig('code/Figures/' + self.name + '_energy_chart.png', dpi=300)
        if plot == True : plt.show()

        return 
    
    def exergy_chart(self, T0, p0, plot = True) :
        dict_received = {}
        dict_delivered = {}
        for transform in self.transforms :
            state_in = getattr(self, f"state_{transform.label_in}")
            state_out = getattr(self, f"state_{transform.label_out}")
            if transform.type == 'comp' :
                args = {'P_el' : self.P_comp, 'mdot_wf' : self.mdot_wf}
                exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                dict_received[r'$P_{el}$'] = exergies['P_{el}']
                dict_delivered[r'$P_{mec,comp}$'] = exergies['P_{loss}']
                dict_delivered[r'$P_{irr,comp}$'] = exergies['P_{irr}']


            elif transform.type == 'adex' :
                mdot_wf = self.mdot_wf
                args = {'mdot_wf' : mdot_wf}
                exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                dict_delivered[r'$P_{irr,adex}$'] = exergies['P_{irr}']

            elif transform.type == 'evap' or transform.type == 'cond' :
                state_in_secondary = getattr(self, f"state_{transform.label_in_secondary}")
                state_out_secondary = getattr(self, f"state_{transform.label_out_secondary}")
                
                if transform.label_in_secondary in ['1_prime', '2_prime'] : 
                    mdot_secondary = self.mdot_LT
                    args = {'mdot_wf' : self.mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    if exergies['P_{wf}'] > exergies['P_{secondary}'] :
                        dict_delivered[r'$\dot{ \Delta E}_{LT, water}$'] = exergies['P_{secondary}']
                    else : 
                        dict_received[r'$\dot{ \DeltaE}_{LT, wf}$'] = exergies['P_{secondary}']

                    dict_delivered[r'$P_{irr,LT}$'] = exergies['P_{irr}']

                elif transform.label_in_secondary in ['4_prime', '3_prime'] : 
                    mdot_secondary = self.mdot_MT
                    args = {'mdot_wf' : self.mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)

                    if exergies['P_{wf}'] > exergies['P_{secondary}'] :
                        dict_delivered[r'$\dot{ \Delta E}_{MT, water}$'] = exergies['P_{secondary}']
                    else :
                        dict_received[r'$\dot{ \Delta E}_{MT, wf}$'] = exergies['P_{secondary}']
                    dict_delivered[r'$P_{irr,MT}$'] = exergies['P_{irr}']

                elif transform.label_in_secondary in ['5_prime', '6_prime'] :
                    mdot_secondary = self.mdot_HT
                    args = {'mdot_wf' : self.mdot_wf, 'mdot_secondary' : mdot_secondary, 
                            'state_in_secondary' : state_in_secondary, 'state_out_secondary' : state_out_secondary}
                    exergies = transform.exergy_analysis(T0, p0, state_in, state_out, args)
                    if exergies['P_{wf}'] > exergies['P_{secondary}'] :
                        dict_delivered[r'$\dot{ \Delta E}_{HT, water}$'] = exergies['P_{secondary}']
                    else :
                        dict_received[r'$\dot{ \Delta E}_{HT, wf}$'] = exergies['P_{secondary}']
                    dict_delivered[r'$P_{irr,HT}$'] = exergies['P_{irr}']
                else : 
                    raise ValueError("Unknown secondary mass flow rate for energy analysis.")
        

        # Plot the energy chart

        plt.figure(figsize=(8,6))
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
        n_cols = length // 2 if length % 2 == 0 else (length // 2) + 1

        plt.legend(frameon=False,
           loc='lower center',
           bbox_to_anchor=(0.5, 1),
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
        plt.savefig('code/Figures/' + self.name + '_exergy_chart.png', dpi=300)
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
