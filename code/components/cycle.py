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

        # List of transforms
        self.transforms = []

        

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
        # 5️⃣ Build final output string
        # ──────────────────────────────────────────────
        output = [
            title,
            "\nState Properties:\n",
            state_table_str,
            "\n\nMass Flow Rates:\n",
            mass_flow_str,
            f"\n{'=' * 118}\n",
        ]
        return "".join(output)
    
    def Ts_diagram(self, n=100) : 
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
        plt.show()

        






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
