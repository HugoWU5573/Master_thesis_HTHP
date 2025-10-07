
from CoolProp.CoolProp import PropsSI 

class HEX():

    """
    Creates a heat exchanger object based on the following inputs:
        - Tin: list of inlet temperatures [K] for cold and hot streams
        - pin: list of inlet pressures [Pa] for cold and hot streams
        - mdot: list of mass flow rates [kg/s] for cold and hot streams
        - fluid: list of fluid names for cold and hot streams

    """
    def __init__(self, Tin, pin, mdot, fluid):
        
        self.Tin_c = Tin[0]
        self.Tin_h = Tin[1]
        self.pin_c = pin[0]
        self.pin_h = pin[1]
        self.mdot_c = mdot[0]
        self.mdot_h = mdot[1]
        self.fluid_c = fluid[0]
        self.fluid_h = fluid[1]
        self.hin_c = PropsSI('H', 'T', self.Tin_c, 'P', self.pin_c, self.fluid_c)
        self.hin_h = PropsSI('H', 'T', self.Tin_h, 'P', self.pin_h, self.fluid_h)
    
    """
    This method calculates the  maximum heat transfer rate given by the external pinching

    """
    def _Qmax_ext(self):
        hout_h = PropsSI('H', 'T', self.Tin_c, 'P', self.pin_h, self.fluid_h)
        hout_c = PropsSI('H', 'T', self.Tin_h, 'P', self.pin_c, self.fluid_c)
        Qmax_h = self.mdot_h * (self.hin_h - hout_h)
        Qmax_c = self.mdot_c * (hout_c - self.hin_c)
        self.Qmax_ext = min(Qmax_h, Qmax_c)
        return self.Qmax_ext

    

    
