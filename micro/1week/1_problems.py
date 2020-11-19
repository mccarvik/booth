
import matplotlib.pyplot as plt 
import numpy as np    

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    axes.set_xlim([0, 2800])
    axes.set_ylim([0, 50])
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(1000, 26.5, "or")
    axes.annotate("P1", (1000, 26.5))
    plt.plot(2000, 9.94, "or")
    axes.annotate("P2", (2000, 9.94))
    plt.plot(x_vals, y_vals, '--b')
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob2.png",dpi=400)
    plt.close


def sup_dem_line(slope1, intercept1, slope2, intercept2):
    """
    Plot demand / supply curve line
    """
    axes = plt.gca()
    axes.set_xlim([0, 2000])
    axes.set_ylim([0, 150])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = intercept1 + slope1 * x_vals
    y_vals2 = intercept2 + slope2 * x_vals
    plt.plot(1000, 50, "o", color='purple')
    axes.annotate("EQ (1000, 50)", (1000, 50))
    plt.plot(x_vals, y_vals1, '--b')
    plt.plot(x_vals, y_vals2, '--r')
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob1.png",dpi=400)
    plt.close()


def dem_q3(slope1, intercept1, slope2, intercept2):
    """
    Plot demand / supply curve line
    """
    axes = plt.gca()
    # axes.set_xlim([40, 80])
    # axes.set_ylim([12, 20])
    axes.set_xlim([0, 150])
    axes.set_ylim([0, 32])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = intercept1 + slope1 * x_vals
    y_vals2 = intercept2 + slope2 * x_vals
    plt.plot(x_vals, y_vals1, '--b', label='Original')
    plt.plot(x_vals, y_vals2, '-r', label='New')
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    # plt.savefig("prob3c_zoomed.png",dpi=400)
    plt.savefig("prob3c.png",dpi=400)
    plt.close()
    

def sup_dem2(slopes, ints):
    """
    Plot demand / supply curve line
    """
    axes = plt.gca()
    axes.set_xlim([0, 2000])
    axes.set_ylim([0, 150])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = ints[0] + slopes[0] * x_vals
    y_vals2 = ints[1] + slopes[1] * x_vals
    y_vals3 = ints[2] + slopes[2] * x_vals
    y_vals4 = ints[3] + slopes[3] * x_vals
    plt.plot(865.385, 53.845, "o", color='purple')
    axes.annotate("EQ (865.385, 53.845)", (150, 53.845))
    plt.plot(x_vals, y_vals1, '--b', label="Old D")
    plt.plot(x_vals, y_vals2, '--r', label= "Old S")
    plt.plot(x_vals, y_vals3, '-b', label="New D")
    plt.plot(x_vals, y_vals4, '-r', label="New S")
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.savefig("prob1_dub.png",dpi=400)
    plt.close()


def shrimp_prob1(slopes, ints):
    """
    Plot demand / supply curve line
    """
    axes = plt.gca()
    axes.set_xlim([0, 1200])
    axes.set_ylim([0, 210])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = ints[0] + slopes[0] * x_vals
    y_vals2 = ints[1] + slopes[1] * x_vals
    y_vals3 = ints[2] + slopes[2] * x_vals
    y_vals4 = ints[3] + slopes[3] * x_vals
    plt.plot(500, 100, "o", color='purple')
    # plt.plot(560, 100, "o", color='purple')
    plt.plot(560, 106, "o", color='purple')
    axes.annotate("EQ Old (500, 100)", (150, 95))
    # axes.annotate("EQ New (560, 100)", (600, 95))
    axes.annotate("EQ New (560, 106)", (600, 101))
    plt.plot(x_vals, y_vals1, '--b', label="Old D")
    # plt.plot(x_vals, y_vals2, '--r', label= "Old S")
    plt.plot(x_vals, y_vals2, '--r', label= "Supply")
    plt.plot(x_vals, y_vals3, '-b', label="New D")
    # plt.plot(x_vals, y_vals4, '-r', label="New S")
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.savefig("prob1_shrimp.png",dpi=400)
    plt.close()


# abline(-0.01656, 43.06)
# sup_dem_line(-0.1, 150, 0.025, 25)

# dem_q3(-0.233, 30.71, -0.233, 30.44)
# sup_dem2([-0.1, 0.025, -0.111, 0.0333], [150, 25, 150, 25])
# shrimp_prob1([-0.2, 0.1, -0.1786, 0.0892], [200, 50, 200, 50])
shrimp_prob1([-0.2, 0.1, -0.16786, 0], [200, 50, 200, 0])
