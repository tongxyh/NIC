import math
from decimal import *
import decimal

precise = 16
decimal.getcontext().prec = precise

half = Decimal(0.5)
sqrt2= Decimal(2).sqrt()
zero = Decimal(0)
n1, n2, n3, n4, n5, n6 = Decimal(1), Decimal(2), Decimal(3), Decimal(4), Decimal(5), Decimal(6)
n16 = Decimal(16)

a1 = Decimal(0.0705230784)
a2 = Decimal(0.0422820123)
a3 = Decimal(0.0092705272)
a4 = Decimal(0.0001520143)
a5 = Decimal(0.0002765672)
a6 = Decimal(0.0000430638)

def quantized_erf(x):
    x = Decimal(x)
    # https://en.wikipedia.org/wiki/Error_function
    # fixed point approximation of erf(x)
    # maximun error: 3x10-7
    return n1-n1/(n1+a1*x+a2*(x**n2)+a3*(x**n3)+a4*(x**n4)+a5*(x**n5)+a6*(x**n6))**Decimal(n16)

def Decimal_cdf(x,mean,scale):
    x = (Decimal(x) - Decimal(mean))/(sqrt2*Decimal(scale))
    
    if x > zero:
        return (half + half*quantized_erf(x))
    else:
        return (half - half*quantized_erf(-x))
    
# for x in [-4.0,-3.0,-2.0,-1.0,0,1,2,3,4]:
#     print(Decimal_cdf(x,0,1))