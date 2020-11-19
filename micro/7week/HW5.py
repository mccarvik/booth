
import matplotlib.pyplot as plt 
import numpy as np   

def prob12():
    """
    Plot demand / supply curve line
    """
    int1 = 200
    slope1 = -0.1
    int2 = 0
    slope2 = 0.00167
    int3 = 0
    slope3 = 0.01
    axes = plt.gca()
    axes.set_xlim([1500, 2000])
    axes.set_ylim([0, 100])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    y_vals3 = int3 + slope3 * x_vals
    y_vals4 = int2 + (slope3 + slope2) * x_vals
    # plt.plot(6, 8, "o", color='purple')
    # axes.annotate("EQ (6, 8)", (6, 8))
    plt.plot(x_vals, y_vals1, '-b', label="D")
    plt.plot(x_vals, y_vals2, '--r', label="MEC")
    plt.plot(x_vals, y_vals3, '--r', label="MPC")
    plt.plot(x_vals, y_vals4, '-r', label="MSC")
    
    # random lines
    plt.plot([0,1818.18], [18.18, 18.18], "--k")
    plt.plot([0,1790.99], [20.9, 20.9], "--k")
    plt.plot([0,1790.99], [20.9, 20.9], "--k")

    plt.plot([1818.18,1818.18], [0, 18.18], "--k")
    plt.plot([1790.99, 1790.99], [0, 20.9], "--k")
    
    
    
    plt.legend()
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob15.png",dpi=400)
    plt.close()

prob12()