import math
# import numpy as np

# def g_poly_convex(l,L):
#     d = 2
#     return 1 - (1-l/L)**d

# def g_poly_concav(l,L):
#     d = 2
#     return (l/L)**d

# def g_linear(l,L):
#     return (l/L)

# def g_composite(l,L):
#     return 1-(1/2)*(math.cos(math.pi * l / L) + 1)


def g_convex(epoch, y1=1,y2=0.5,x1=1,x2=120):
    a = y1-y2
    hs = math.pi/(2*(x2-x1))
    return a * math.cos(hs*(epoch+x2-2*x1)) + y1

def g_concave(epoch, y1=1,y2=0.5,x1=1,x2=120):
    a = y1-y2
    hs = math.pi/(2*(x2-x1))
    return a * math.cos(hs*(epoch-x1)) + y2

def g_linear(epoch, y1=1,y2=0.5,x1=1,x2=120):
    m = (y2 - y1)/(x2-x1)
    b = y1 - (y2 - y1)/(x2 - x1) * x1
    return m * epoch + b

def g_composite_1(epoch, y1=1,y2=0.5,x1=1,x2=120):
    a = y1-y2
    hs = math.pi/(2*(x2-x1))
    return (a * math.cos(2*hs*(epoch-x1)) + y2+y1)/2

def g_composite_2(epoch, y1=1,y2=0.5,x1=1,x2=120):
    return g_linear(epoch, y1, y2, x1, x2) - (g_composite_1(epoch, y1, y2, x1, x2) - g_linear(epoch, y1, y2, x1, x2))