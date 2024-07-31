import math

def g_poly_convex(l,L):
    d = 2
    return 1 - (1-l/L)**d

def g_poly_concav(l,L):
    d = 2
    return (l/L)**d

def g_linear(l,L):
    return (l/L)

def g_composite(l,L):
    return 1-(1/2)*(math.cos(math.pi * l / L) + 1)