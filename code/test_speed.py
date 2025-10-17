import CoolProp
from CoolProp.CoolProp import PropsSI
import time

import components.state as state_1
import components.state_2 as state_2

p = 1e5 
h = 200e3
# Timing PropsSI
start_propssi = time.time()
for _ in range(1000):
    state_1.State(p=p, h = h,fluid='water')
end_propssi = time.time()
propssi_time = end_propssi - start_propssi

# Timing AbstractState
start_abstract = time.time()
HEOS = CoolProp.AbstractState("HEOS", 'water')
for _ in range(1000):
    state_2.State(HEOS, p=p, h = h)
end_abstract = time.time()
abstract_time = end_abstract - start_abstract

print(f"PropsSI time for 1000 calls: {propssi_time:.4f} seconds")
print(f"AbstractState time for 1000 calls: {abstract_time:.4f} seconds")
