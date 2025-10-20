import CoolProp 

class State() :

    def __init__(self, heos, T=None, p=None, Q=None, h=None, s=None) :
        """
        Initialize a thermodynamic state with given properties.

        Parameters:
        T (float): Temperature in Kelvin.
        p (float): Pressure in Pascal.
        Q (float): Vapor quality (0 to 1).
        h (float): Enthalpy in J/kg.
        s (float): Entropy in J/kg-K.
        fluid (str): The working fluid.


        At least two independent properties must be provided to define the state.
        """
        self.heos = heos
        self.e = None

        self.T = T
        self.p = p
        self.Q = Q
        self.h = h
        self.s = s
        self.fluid = heos.fluid_names()[0]

        if self.Q is not None:
            if self.Q == -1:
                raise ValueError("Point not under the saturation curve")

            if self.T is not None:
                self.heos.update(CoolProp.QT_INPUTS, self.Q, self.T)
                self.p = self.heos.p()
                self.h = self.heos.hmass()
                self.s = self.heos.smass()

            elif self.p is not None:
                self.heos.update(CoolProp.PQ_INPUTS, self.p, self.Q)
                self.T = self.heos.T()
                self.h = self.heos.hmass()
                self.s = self.heos.smass()

            elif self.h is not None:
                self.heos.update(CoolProp.HmassQ_INPUTS, self.h, self.Q)
                self.T = self.heos.T()
                self.p = self.heos.p()
                self.s = self.heos.smass()

            elif self.s is not None:
                self.heos.update(CoolProp.QSmass_INPUTS, self.Q, self.s)
                self.T = self.heos.T()
                self.p = self.heos.p()
                self.h = self.heos.hmass()
                
            else:
                raise ValueError("Need T, p, h, or s when Q is specified")
            
        # --- Case outside the saturation curve ---
        elif self.T is not None and self.p is not None:
            self.heos.update(CoolProp.PT_INPUTS, self.p, self.T)
            self.h = self.heos.hmass()
            self.s = self.heos.smass()
            self.Q = self.heos.Q()

        elif self.T is not None and self.s is not None:
            self.heos.update(CoolProp.SmassT_INPUTS, self.s, self.T)
            self.p = self.heos.p()
            self.h = self.heos.hmass()
            self.Q = self.heos.Q()

        elif self.p is not None and self.h is not None:
            self.heos.update(CoolProp.HmassP_INPUTS, self.h, self.p)
            self.T = self.heos.T()
            self.s = self.heos.smass()
            self.Q = self.heos.Q()

        elif self.p is not None and self.s is not None:
            self.heos.update(CoolProp.PSmass_INPUTS, self.p, self.s)
            self.T = self.heos.T()
            self.h = self.heos.hmass()
            self.Q = self.heos.Q()

        elif self.h is not None and self.s is not None:
            self.heos.update(CoolProp.HmassSmass_INPUTS, self.h, self.s)
            self.T = self.heos.T()
            self.p = self.heos.p()
            self.Q = self.heos.Q()
        
        elif self.h is not None and self.T is not None:
            self.heos.update(CoolProp.HmassT_INPUTS, self.h, self.T)
            self.p = self.heos.p()
            self.s = self.heos.smass()
            self.Q = self.heos.Q()

        else:
            raise ValueError("Invalid or insufficient property combination")
        
        
    def exergy(self, T0=298.15, p0=101325):
        """
        Calculate the specific exergy of the state.

        Parameters:
        T0 (float): Reference temperature in Kelvin.
        p0 (float): Reference pressure in Pascal.

        Returns:
        float: Specific exergy in J/kg.
        """
        self.heos.update(CoolProp.PT_INPUTS, p0, T0)
        h0 = self.heos.hmass()
        s0 = self.heos.smass()
        self.e = (self.h - h0) - T0 * (self.s - s0)
        return self.e