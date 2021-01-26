
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

cr = 0.10
cv = 0.15
br = 0.19
bv = 0.25
ar = 0.04
av = 0.00
pr = 0.13
pv = 0.15

def ports(slopes, ints):
    """
    Plot demand / supply curve line
    """
    axes = plt.gca()
    axes.set_xlim([0.0, 0.50])
    axes.set_ylim([0.0, 0.50])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = ints[0] + slopes[0] * x_vals
    y_vals2 = ints[1] + slopes[1] * x_vals
    y_vals3 = ints[2] + slopes[2] * x_vals
    y_vals4 = ints[3] + slopes[3] * x_vals
    plt.plot(x_vals, y_vals1, '--', color='black', label="Port A")
    plt.plot(x_vals, y_vals2, '--g', label= "Port B")
    plt.plot(x_vals, y_vals3, '--b', label="Port C")
    plt.plot(x_vals, y_vals4, '-r', label="Cust Port")
    plt.xlabel("Risk", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.legend()
    plt.savefig("ports.png",dpi=400)
    plt.close()


def frontier():
    """
    Plot the frontier
    """
    axes = plt.gca()
    axes.set_xlim([0, 0.50])
    axes.set_ylim([0, 0.5])
    x_vals = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 50)
    
    ys = []
    xs = []
    for x in x_vals:
        w = (br * cv**2) / ((br * cv**2) + (cr * bv**2))
        er = w * br + (1-w) * cr
        std = ((w**2 * bv**2) + ((1-w)**2 * cv**2))**(1/2)
        w2 = x / std
        er2 = er * w2 + ar * (1 - w2)
        ys.append(er2)
    
    b_sharpe = (br - ar) / bv
    c_sharpe = (cr - ar) / cv
    
    y_vals1 = ar + b_sharpe * x_vals
    y_vals2 = ar + c_sharpe * x_vals
    
    plt.plot(x_vals, ys, '-b', label="MVE")
    plt.plot(x_vals, y_vals1, '--r', label= "B port CAL")
    plt.plot(x_vals, y_vals2, '--r', label= "C Port CAL")
    plt.xlabel("St dev", fontsize=12)
    plt.ylabel("E(r)", fontsize=12)
    plt.legend()
    plt.savefig("frontier.png",dpi=400)
    plt.close()

# sharpes = []
# for r, v in zip([cr, br, ar, pr], [cv, bv, av, pv]):
#     pdb.set_trace()
#     if v != 0:
#         sharpes.append((r - ar)/v)
#     else:
#         sharpes.append(0)

# port_points([cv, bv, av, pv], [cr, br, ar, pr])
frontier()