import math
from decimal import *
import decimal
import sys
from functools import partial
sys.setrecursionlimit(1000000)

precise = 16
D = Decimal
decimal.getcontext().prec = precise
# print(decimal.getcontext().rounding)
half = Decimal("0.5")
sqrt2= Decimal(2).sqrt()
zero = Decimal(0)
n1, n2, n3, n4, n5, n6 = Decimal(1), Decimal(2), Decimal(3), Decimal(4), Decimal(5), Decimal(6)
n16 = Decimal(16)

table_num = 350000
quan_prec = Decimal('0.00000')

# maximun error: 5x10-5
a01 = Decimal(0.278393)
a02 = Decimal(0.230389)
a03 = Decimal(0.000972)
a04 = Decimal(0.078108)

# maximun error: 3x10-7
a1 = Decimal(0.0705230784)
a2 = Decimal(0.0422820123)
a3 = Decimal(0.0092705272)
a4 = Decimal(0.0001520143)
a5 = Decimal(0.0002765672)
a6 = Decimal(0.0000430638)

lower = Decimal(0)
upper = Decimal(5)

def index_to_loc(x, H_PAD, W_PAD):
    c_index = x // int(H_PAD/16*W_PAD/16)
    h_index = (x % int(H_PAD/16*W_PAD/16)) // int(W_PAD/16)
    w_index = (x % int(H_PAD/16*W_PAD/16)) % int(W_PAD/16)
    return c_index, h_index, w_index

def round_check(x):
    x_m = D(x).quantize(D("0.01"))
    
    # max observed error 8e-6
    # closer to zero, less error, can it help?
    err_step = 2e-6
    if abs(x) >= 1:
        x_l = D(x-(abs(x)*err_step)).quantize(D("0.01"))
        x_u = D(x+(abs(x)*err_step)).quantize(D("0.01"))
    else:
        x_l = D(x-err_step).quantize(D("0.01"))
        x_u = D(x+err_step).quantize(D("0.01"))
 
    if x_l == x_u:
        return x_m, 0
    if x_m == x_l:
        # print(x, x_m, x_l, x_u)
        return x_l, -1
    else:
        # print(x, x_m, x_l, x_u)
        return x_u, 1
    return D(x).quantize(D("0.01")), 0

def round_check_test(x):
    return D(x).quantize(D("0.01"))

def quick_search(x, i):
    global lower,upper
    if quantized_erf(x).quantize(Decimal(quan_prec)) == Decimal(i)/table_num:
        # print(x,2*Decimal(i)/table_num)
        return x
    if quantized_erf(x).quantize(Decimal(quan_prec)) < Decimal(i)/table_num:
        lower = x
        # print(x,quantized_erf(x).quantize(Decimal('0.0000')), "< " ,Decimal(i)/10000,  upper)
        x = Decimal((x + upper)/2)
        return quick_search(x, i)
    if quantized_erf(x).quantize(Decimal(quan_prec)) > Decimal(i)/table_num:
        upper = x
        # print(x,Decimal(i)/10000,  ">", lower)
        x = Decimal((x+lower)/2)
        return quick_search(x, i)

# def cdf_table():
#     global lower,upper
#     table = []
#     append = table.append
#     for i in range(0,table_num+1):
#         x = quick_search(lower, i)
#         append(x)
#         lower = x
#         upper = 4
#     print(len(table))
#     return table
table_num = 35000
level_table = int(table_num*10/35)
def cdf_table():
    table = []

    decimal.getcontext().prec = 16
    append = table.append
    for i in range(0,table_num+1): # [0.00000 - 3.50000]
        # print(i/100000, quantized_erf(Decimal(i)/100000))
        append(quantized_erf(Decimal(3.5)*Decimal(i)/table_num))
    return table

def quantized_erf(x, level=1):
    # x = Decimal(x)
    # https://en.wikipedia.org/wiki/Error_function
    # fixed point approximation of erf(x)
    
    # maximun error: 5x10-5
    if level == 0:
        #return n1-n1/(n1+a01*x+a02*(x*x)+a03*(x**n3)+a04*(x**n4))**n4
        tmp = (((a04*x+a03)*x + a02)*x + a01)*x + 1
        return 1 - 1/(tmp**4)
        
    # maximun error: 3x10-7
    if level == 1:
        return 1-1/(1+a1*x+a2*(x**n2)+a3*(x**n3)+a4*(x**4)+a5*(x**5)+a6*(x**6))**16

eps = D("0.00001")
def Decimal_cdf(x,mean,scale):
    x = (Decimal(x)-mean[0]-half)/(sqrt2*scale[0]+eps)
    sign = half if x >= 0 else -half
    if abs(x) > Decimal(3.5):
        return half + sign
    else:
        return half + sign*table[int(abs(x*level_table))]

def Decimal_cdf_fast(x,mean,scale):
    x = (x-mean)/scale
    sign = half if x >= 0 else -half
    if abs(x) > Decimal(3.5):
        return half + sign
    else:
        return half + sign*table[int(abs(x*level_table))]

def Decimal_cdf_(x, mean=0, scale=D(1e-6)):
    # batch version
    mean = mean + half
    scale = scale*sqrt2 + eps
    partial_func = partial(Decimal_cdf_fast,mean=mean,scale=scale)
    return list(map(partial_func, x))

def erf(x, level=0):
    if level == 0:
        #return n1-n1/(n1+a01*x+a02*(x*x)+a03*(x**n3)+a04*(x**n4))**n4
        tmp = (((0.07810*x+0.000972)*x + 0.230389)*x + 0.278393)*x + 1
        return 1 - 1/(tmp**4)

def cdf(x, mean, scale):
    x = (x - mean)/((2**0.5)*scale)
    sign = 1 if x >= 0 else -1
    return 0.5 + sign*0.5*erf(abs(x))

def Decimal_cdf_np(x, mean, scale):
    pass

table = cdf_table()
# sample = [1,2,3]
# Decimal_cdf_partial = partial(Decimal_cdf_, x=sample)
# def partial_cdf(mean, scale):
#     return Decimal_cdf_partial(mean=mean, scale=scale)
# partial_cdf([1,1],[1,1])
# import profile
# profile.run('Decimal_cdf_([0,0,0,0,0],[1,1,1,1,1,1],[2,2,2,2,2,2])')
# cdf_table()
# from line_profiler import LineProfiler
# lp = LineProfiler()
# lp_wrapper = lp(Decimal_cdf)
# for i in range(2000):
#     lp_wrapper(0,1,2)
# lp.print_stats()
# print("DEBUGING")
# print(65450*Decimal_cdf(x=1,mean=[D("-0.01"),0], scale=[D("0.59"),0]))
# print(65450*Decimal_cdf(x=0,mean=[D("-0.01"),0], scale=[D("0.59"),0]))