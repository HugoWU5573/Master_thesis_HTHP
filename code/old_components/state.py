from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt

class State:
    """
    Class representing a thermodynamic state of a fluid.
    You can progressively define the attributes (T, p, Q, h, s),
    but once the state is completely determined, it becomes immutable.
    """

    _keys = ('T', 'p', 'Q', 'h', 's')

    def __init__(self, T=None, p=None, Q=None, h=None, s=None, fluid='R290'):
        super().__setattr__('_updating', False)
        super().__setattr__('_locked', False)
        super().__setattr__('fluid', fluid)

        # Initial assignment without immediate calculation
        super().__setattr__('T', T)
        super().__setattr__('p', p)
        super().__setattr__('Q', Q)
        super().__setattr__('h', h)
        super().__setattr__('s', s)
        super().__setattr__('valid', False)
        super().__setattr__('_initialized', True)

        # Try to calculate immediately if two properties are already known
        self._try_update()

    # --------------------------------------------------------------
    # ATTRIBUTE ASSIGNMENT OVERRIDE
    # --------------------------------------------------------------
    def __setattr__(self, name, value):
        # Intercept changes to thermodynamic properties
        if getattr(self, '_initialized', False) and name in self._keys:
            # If the state is already complete, modification is forbidden
            if self._locked:
                raise RuntimeError("You tried to modify an already existing state")

        super().__setattr__(name, value)

        # Recalculate automatically if a state variable changes
        if getattr(self, '_initialized', False) and name in self._keys:
            self._try_update()

    # --------------------------------------------------------------
    # AUTOMATIC STATE UPDATE
    # --------------------------------------------------------------
    def _try_update(self):
        """Attempt a recalculation if two independent properties are defined."""
        if self._updating or self._locked:
            return
        self._updating = True

        defined = [k for k in self._keys if getattr(self, k) is not None]

        # Wait until at least two values are defined
        if len(defined) < 2:
            self.valid = False
            self._updating = False
            return

        try:
            self._update_state()
            self.valid = True
            self._locked = True  # Once the state is complete, lock it
        except Exception as e:
            raise e
            self.valid = False
        finally:
            self._updating = False

    # --------------------------------------------------------------
    # MAIN STATE CALCULATION
    # --------------------------------------------------------------
    def _update_state(self):
        """Compute missing properties based on valid input pairs."""
        T, p, Q, h, s, fluid = self.T, self.p, self.Q, self.h, self.s, self.fluid

        # --- Case under the saturation curve ---
        if Q is not None:
            if Q == -1:
                raise ValueError("Point not under the saturation curve")

            if T is not None:
                self.p = PropsSI('P', 'T', T, 'Q', Q, fluid)
                self.h = PropsSI('H', 'T', T, 'Q', Q, fluid)
                self.s = PropsSI('S', 'T', T, 'Q', Q, fluid)

            elif p is not None:
                self.T = PropsSI('T', 'P', p, 'Q', Q, fluid)
                self.h = PropsSI('H', 'P', p, 'Q', Q, fluid)
                self.s = PropsSI('S', 'P', p, 'Q', Q, fluid)
            else:
                raise ValueError("Need T or p when Q is specified")

        # --- Case outside the saturation curve ---
        elif T is not None and p is not None:
            self.h = PropsSI('H', 'T', T, 'P', p, fluid)
            self.s = PropsSI('S', 'T', T, 'P', p, fluid)
            self.Q = PropsSI('Q', 'T', T, 'P', p, fluid)

        elif T is not None and s is not None:
            self.p = PropsSI('P', 'T', T, 'S', s, fluid)
            self.h = PropsSI('H', 'T', T, 'S', s, fluid)
            self.Q = PropsSI('Q', 'T', T, 'S', s, fluid)

        elif p is not None and h is not None:
            self.T = PropsSI('T', 'P', p, 'H', h, fluid)
            self.s = PropsSI('S', 'P', p, 'H', h, fluid)
            self.Q = PropsSI('Q', 'P', p, 'H', h, fluid)

        elif p is not None and s is not None:
            self.T = PropsSI('T', 'P', p, 'S', s, fluid)
            self.h = PropsSI('H', 'P', p, 'S', s, fluid)
            self.Q = PropsSI('Q', 'P', p, 'S', s, fluid)

        elif h is not None and s is not None:
            self.T = PropsSI('T', 'H', h, 'S', s, fluid)
            self.p = PropsSI('P', 'H', h, 'S', s, fluid)
            self.Q = PropsSI('Q', 'H', h, 'S', s, fluid)

        else:
            raise ValueError("Invalid or insufficient property combination")

    # --------------------------------------------------------------
    # STRING REPRESENTATION
    # --------------------------------------------------------------
    def __str__(self):
        return (
            f"State("
            f"T={self.T}, p={self.p}, h={self.h}, s={self.s}, Q={self.Q}, "
            f"valid={self.valid}, locked={self._locked})"
        )

    # --------------------------------------------------------------
    # EXERGY CALCULATION
    # --------------------------------------------------------------
    def exergy(self, T0):
        """Compute the specific exergy of the state."""
        if self.h is not None and self.s is not None:
            return self.h - T0 * self.s
        raise ValueError("Need h and s defined to compute exergy.")





''' 
state_in = State(p=4e5, Q = 0.5, fluid='R290') #inside point of curve
state_out = State(p=4e5, s = 2500, fluid='R290') #outside point of curve
state_out_2 = State(p=4e5, s = 500, fluid='R290') #outside point of curve

T_in, p_in, Q_in, h_in, s_in = state_in.T, state_in.p, state_in.Q, state_in.h, state_in.s
T_out, p_out, Q_out, h_out, s_out = state_out.T, state_out.p, state_out.Q, state_out.h, state_out.s
T_out_2, p_out_2, Q_out_2, h_out_2, s_out_2 = state_out_2.T, state_out_2.p, state_out_2.Q, state_out_2.h, state_out_2.s



# Test cases for point below saturation curve

state_test = State(T=T_in, Q = Q_in, fluid='R290')
state_test_2 = State(p=p_in, Q = Q_in, fluid='R290')
#state_test_3 = State(h=h_in, Q = Q_in, fluid='R290')
#state_test_4 = State(s=s_in, Q = Q_in, fluid='R290')

#state_test_5 = State(T=T_in, p = p_in, fluid='R290')
#state_test_6 = State(T=T_in, h = h_in, fluid='R290')
state_test_7 = State(T=T_in, s = s_in, fluid='R290')
state_test_8 = State(p=p_in, h = h_in, fluid='R290')
state_test_9 = State(p=p_in, s = s_in, fluid='R290')
state_test_10 = State(h=h_in, s = s_in, fluid='R290')

# Test cases for point beside saturation curve
state_test = State(T=T_out, s = s_out, fluid='R290')
state_test_2 = State(p=p_out, s = s_out, fluid='R290')
state_test_3 = State(h=h_out, s = s_out, fluid='R290')
state_test_4 = State(T=T_out, p = p_out, fluid='R290')
#state_test_5 = State(T=T_out, h = h_out, fluid='R290')
state_test_6 = State(p=p_out, h = h_out, fluid='R290')
state_test_7 = State(T=T_out_2, s = s_out_2, fluid='R290')
state_test_8 = State(p=p_out_2, s = s_out_2, fluid='R290')
state_test_9 = State(h=h_out_2, s = s_out_2, fluid='R290')
state_test_10 = State(T=T_out_2, p = p_out_2, fluid='R290')
#state_test_11 = State(T=T_out_2, h = h_out_2, fluid='R290')
state_test_12 = State(p=p_out_2, h = h_out_2, fluid='R290')


fluid = 'R290'
T_min = PropsSI('TMIN', fluid)
T_max = PropsSI('TCRIT', fluid) - 1  # Avoid critical point

T_range = np.linspace(T_min, T_max, 300)
s_liq = [PropsSI('S', 'T', T, 'Q', 0, fluid) for T in T_range]
s_vap = [PropsSI('S', 'T', T, 'Q', 1, fluid) for T in T_range]

plt.figure(figsize=(8,6))
plt.plot(s_liq, T_range, label='Saturated Liquid')
plt.plot(s_vap, T_range, label='Saturated Vapor')
plt.scatter([state_in.s], [state_in.T], color='red', label='State In', zorder=5)
plt.annotate('In', (state_in.s, state_in.T), textcoords="offset points", xytext=(10,10), ha='center', color='red')
plt.scatter([state_out.s], [state_out.T], color='blue', label='State Out', zorder=5)
plt.annotate('Out', (state_out.s, state_out.T), textcoords="offset points", xytext=(10,10), ha='center', color='blue')
plt.scatter([state_out_2.s], [state_out_2.T], color='green', label='State Out 2', zorder=5)
plt.annotate('Out 2', (state_out_2.s, state_out_2.T), textcoords="offset points", xytext=(10,10), ha='center', color='green')
plt.xlabel('Entropy [J/kg-K]')
plt.ylabel('Temperature [K]')
plt.title('T-s Diagram of Propane (R290)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

