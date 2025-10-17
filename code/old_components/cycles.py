import pandas as pd
from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt
import numpy as np


class Cycle():

    """
    Creates a cycle object based on the name provided.
        - Example : name = "SC1", "SC1R", "SC2", "SC2R", ...

    """

    def __init__(self, name):
        self.name = name

        # Working fluid
        self.working_fluid = None

        # Temperatures
        self.T1 = None          # K
        self.T2 = None          # K
        self.T3 = None          # K
        self.T4 = None          # K
        self.T5 = None          # K
        self.T6 = None          # K
        self.T7 = None          # K
        self.T8 = None          # K
        self.T9 = None          # K
        self.T10 = None         # K
        self.T1_prime = None    # K
        self.T2_prime = None    # K
        self.T3_prime = None    # K
        self.T4_prime = None    # K
        self.T5_prime = None    # K
        self.T6_prime = None    # K

        # Pressures
        self.p1 = None          # Pa
        self.p2 = None          # Pa
        self.p3 = None          # Pa
        self.p4 = None          # Pa
        self.p5 = None          # Pa
        self.p6 = None          # Pa
        self.p7 = None          # Pa
        self.p8 = None          # Pa
        self.p9 = None          # Pa
        self.p10 = None         # Pa
        self.p1_prime = None    # Pa
        self.p2_prime = None    # Pa
        self.p3_prime = None    # Pa
        self.p4_prime = None    # Pa
        self.p5_prime = None    # Pa
        self.p6_prime = None    # Pa

        # Enthalpies
        self.h1 = None          # J/kg
        self.h2 = None          # J/kg
        self.h3 = None          # J/kg
        self.h4 = None          # J/kg
        self.h5 = None          # J/kg
        self.h6 = None          # J/kg
        self.h7 = None          # J/kg
        self.h8 = None          # J/kg
        self.h9 = None          # J/kg
        self.h10 = None         # J/kg
        self.h1_prime = None    # J/kg
        self.h2_prime = None    # J/kg
        self.h3_prime = None    # J/kg
        self.h4_prime = None    # J/kg
        self.h5_prime = None    # J/kg
        self.h6_prime = None    # J/kg

        # Entropies
        self.s1 = None          # J/(kg K)
        self.s2 = None          # J/(kg K)
        self.s3 = None          # J/(kg K)
        self.s4 = None          # J/(kg K)
        self.s5 = None          # J/(kg K)
        self.s6 = None          # J/(kg K)
        self.s7 = None          # J/(kg K)
        self.s8 = None          # J/(kg K)
        self.s9 = None          # J/(kg K)
        self.s10 = None         # J/(kg K)
        self.s1_prime = None    # J/(kg K)
        self.s2_prime = None    # J/(kg K)
        self.s3_prime = None    # J/(kg K)
        self.s4_prime = None    # J/(kg K)
        self.s5_prime = None    # J/(kg K)
        self.s6_prime = None    # J/(kg K)

        # Exergies
        self.e1 = None          # J/kg
        self.e2 = None          # J/kg
        self.e3 = None          # J/kg
        self.e4 = None          # J/kg
        self.e5 = None          # J/kg
        self.e6 = None          # J/kg
        self.e7 = None          # J/kg
        self.e8 = None          # J/kg
        self.e9 = None          # J/kg
        self.e10 = None         # J/kg
        self.e1_prime = None    # J/kg
        self.e2_prime = None    # J/kg
        self.e3_prime = None    # J/kg
        self.e4_prime = None    # J/kg
        self.e5_prime = None    # J/kg
        self.e6_prime = None    # J/kg

        # Qualities
        self.x1 = None          # -
        self.x2 = None          # -
        self.x3 = None          # -
        self.x4 = None          # -
        self.x5 = None          # -
        self.x6 = None          # -
        self.x7 = None          # -
        self.x8 = None          # -
        self.x9 = None          # -
        self.x10 = None         # -
        self.x1_prime = None    # -
        self.x2_prime = None    # -
        self.x3_prime = None    # -
        self.x4_prime = None    # -
        self.x5_prime = None    # -
        self.x6_prime = None    # -

        # Mass flow rates
        self.mdot_wf = None     # kg/s
        self.mdot_LT = None     # kg/s
        self.mdot_MT = None     # kg/s
        self.mdot_HT = None     # kg/s


    """
    This method provides a string representation of the cycle.
    The output includes the cycle name, mass flow rates, and a table of state properties.

    Remark : This method is called when printing any instance of the Cycle class -> print(cycle_instance)

    """
    def __str__(self):

        # Title
        title = f"\n================================================== Cycle: {self.name} ==================================================\n"

        # Mass flow rates table (only non-None)
        mass_flow_items = [
            ("mdot_wf", self.mdot_wf),
            ("mdot_LT", self.mdot_LT),
            ("mdot_MT", self.mdot_MT),
            ("mdot_HT", self.mdot_HT),
        ]
        mass_flow_filtered = [(name, value) for name, value in mass_flow_items if value is not None]
        if mass_flow_filtered:
            mass_flow_header = "+----------------+--------------+"
            mass_flow_table = [mass_flow_header,
                               "| Mass Flow Rate | Value [kg/s] |",
                               mass_flow_header]
            for name, value in mass_flow_filtered:
                mass_flow_table.append(f"| {name:<14} | {value:<12.2f} |")
            mass_flow_table.append(mass_flow_header)
            mass_flow_str = "\n".join(mass_flow_table)
        else:
            mass_flow_str = "(No mass flow rates set)"

        # State properties table (only states with temperature)
        state_rows = []
        state_header = "+-------+--------------+--------------+-------------------+---------------------+----------------+-------------+"
        state_rows.append(state_header)
        state_rows.append("| State | Temperature  | Pressure     | Enthalpy          | Entropy             | Exergy         | Quality     |")
        state_rows.append("|       | [K]          | [bar]        | [kJ/kg]           | [J/kg/K]            | [J/kg]         | [-]         |")
        state_rows.append(state_header)

        def format_function(val, fmtstr):
            return fmtstr.format(val) if val is not None else ""

        # States 1-10
        for i in range(1, 11):
            T = getattr(self, f"T{i}")
            if T is not None:
                row = [
                    f"{i}",
                    f"{T:.2f}",
                    format_function(getattr(self, f"p{i}"), "{:.2f}") if getattr(self, f"p{i}") is not None else "",
                    format_function(getattr(self, f"h{i}"), "{:.2f}") if getattr(self, f"h{i}") is not None else "",
                    format_function(getattr(self, f"s{i}"), "{:.2f}") if getattr(self, f"s{i}") is not None else "",
                    format_function(getattr(self, f"e{i}"), "{:.2f}") if getattr(self, f"e{i}") is not None else "",
                    format_function(getattr(self, f"x{i}"), "{:.4f}") if getattr(self, f"x{i}") is not None else "",
                ]
                # Pressure in bar, enthalpy in kJ/kg
                if row[2]:
                    row[2] = f"{float(row[2])/1e5:.2f}"
                if row[3]:
                    row[3] = f"{float(row[3])/1000:.2f}"
                state_rows.append("| {0:<5} | {1:<12} | {2:<12} | {3:<17} | {4:<19} | {5:<14} | {6:<11} |".format(*row))
        # States 1'-6'
        for i in range(1, 7):
            T = getattr(self, f"T{i}_prime")
            if T is not None:
                row = [
                    f"{i}'",
                    f"{T:.2f}",
                    format_function(getattr(self, f"p{i}_prime"), "{:.2f}") if getattr(self, f"p{i}_prime") is not None else "",
                    format_function(getattr(self, f"h{i}_prime"), "{:.2f}") if getattr(self, f"h{i}_prime") is not None else "",
                    format_function(getattr(self, f"s{i}_prime"), "{:.2f}") if getattr(self, f"s{i}_prime") is not None else "",
                    format_function(getattr(self, f"e{i}_prime"), "{:.2f}") if getattr(self, f"e{i}_prime") is not None else "",
                    format_function(getattr(self, f"x{i}_prime"), "{:.4f}") if getattr(self, f"x{i}_prime") is not None else "",
                ]
                # Pressure in bar, enthalpy in kJ/kg
                if row[2]:
                    row[2] = f"{float(row[2])/1e5:.2f}"
                if row[3]:
                    row[3] = f"{float(row[3])/1000:.2f}"
                state_rows.append("| {0:<5} | {1:<12} | {2:<12} | {3:<17} | {4:<19} | {5:<14} | {6:<11} |".format(*row))
        state_rows.append(state_header)
        state_table_str = "\n".join(state_rows)

        # Build output string
        output = title
        output += "\nState Properties:\n"
        output += state_table_str
        output += "\n\nMass Flow Rates:\n"
        output += mass_flow_str
        output += "\n=================================================================================================================\n"
        return output
    
    def ts_diagram(self):
        """
        Plots the T-s diagram of the cycle using matplotlib.
        Requires the states to be defined in the cycle.
        """

        # Generate saturation curve for working fluid
        T_min = PropsSI('Tmin', self.working_fluid) + 1
        T_crit = PropsSI('Tcrit', self.working_fluid) - 1
        print(PropsSI('Tcrit', self.working_fluid))
        T_sat = np.linspace(T_min, T_crit, 500)
        s_liq = [PropsSI('S', 'T', T, 'Q', 0, self.working_fluid) for T in T_sat]
        s_vap = [PropsSI('S', 'T', T, 'Q', 1, self.working_fluid) for T in T_sat]

        plt.figure(figsize=(8,6))
        plt.plot(s_liq, T_sat, 'black')
        plt.plot(s_vap, T_sat, 'black')
        plt.scatter(PropsSI('S', 'T', T_crit, 'Q', 0.5, self.working_fluid), T_crit, color='black', s=10)  # Triple point

        plt.xlabel('Entropy [J/kg-K]')
        plt.legend(frameon=False)


        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.set_title('Temperature [°C]', loc='left')

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move bottom and left spines away
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        plt.show()


# Example of usage
SC1 = Cycle("SC1")
SC1.T1_prime = 10 + 273.15
SC1.T4_prime = 35 + 273.15
SC1.p1_prime = 1e5
SC1.p4_prime = 1e5
SC1.mdot_LT = 0.4
SC1.mdot_MT = 0.5
SC1.working_fluid = 'R290'
print(SC1)

SC1.ts_diagram()