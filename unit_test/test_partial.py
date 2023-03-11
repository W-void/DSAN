from functools import *


# A normal function
def add(a, b, c, d, e):
	return 100 * a + 10 * b + c


# A partial function with b = 1 and c = 2
add_part = partial(add, c=3, e=5)

# Calling partial function
print(add_part(a=3, b=2, d=3))
