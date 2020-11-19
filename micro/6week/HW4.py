
import matplotlib.pyplot as plt 
import numpy as np   

def prob12():
    """
    Plot demand / supply curve line
    """
    int1 = 20
    slope1 = -2
    int2 = 2
    slope2 = 1
    axes = plt.gca()
    axes.set_xlim([0, 12])
    axes.set_ylim([0, 22])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    plt.plot(6, 8, "o", color='purple')
    axes.annotate("EQ (6, 8)", (6, 8))
    plt.plot(x_vals, y_vals1, '-b')
    plt.plot(x_vals, y_vals2, '-r')
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob12.png",dpi=400)
    plt.close()


def prob12c():
    """
    Plot demand / supply curve line
    """
    int1 = 20
    slope1 = -2
    int2 = 2
    slope2 = 1
    int3 = 1
    slope3 = 1
    axes = plt.gca()
    # axes.set_xlim([0, 12])
    # axes.set_ylim([0, 22])
    axes.set_xlim([4, 8])
    axes.set_ylim([5, 13])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    y_vals3 = int3 + slope3 * x_vals
    plt.plot(6, 8, "o", color='purple')
    axes.annotate("Old EQ (6, 8)", (5.5, 8.5))
    plt.plot(6.33, 7.33, "o", color='purple')
    axes.annotate("New EQ (6.33, 7.33)", (5.83, 6.83))
    plt.plot(6.33, 8.33, "o", color='purple')
    axes.annotate("(6.33, 8.33)", (6, 8.83))
    plt.plot(x_vals, y_vals1, '-b', label="D")
    plt.plot(x_vals, y_vals2, '-r', label="S old")
    plt.plot(x_vals, y_vals3, '--r', label="S new")
    
    # random lines
    plt.plot([0,6.33], [7.33, 7.33], "--k")
    plt.plot([0,6], [8, 8], "--k")
    plt.plot([0,6.33], [8.33, 8.33], "--k")
    plt.plot([6,6], [0, 8], "--k")
    plt.plot([6.33,6.33], [0, 8.33], "--k")
    
    plt.legend()
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob12c.png",dpi=400)
    plt.close()


def prob12d():
    """
    Plot demand / supply curve line
    """
    int1 = 20
    slope1 = -2
    int2 = 2
    slope2 = 1
    int3 = 21
    slope3 = -2
    axes = plt.gca()
    # axes.set_xlim([0, 12])
    # axes.set_ylim([0, 22])
    axes.set_xlim([4, 8])
    axes.set_ylim([5, 13])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    y_vals3 = int3 + slope3 * x_vals
    # plt.plot(6, 8, "o", color='purple')
    # axes.annotate("EQ (6, 8)", (6, 8))
    plt.plot(x_vals, y_vals1, '-b', label="D")
    plt.plot(x_vals, y_vals2, '-r', label="S old")
    plt.plot(x_vals, y_vals3, '--r', label="S new")
    
    plt.plot(6, 8, "o", color='purple')
    axes.annotate("Old EQ (6, 8)", (5.5, 8.5))
    plt.plot(6.33, 7.33, "o", color='purple')
    axes.annotate("(6.33, 7.33)", (6, 6.83))
    plt.plot(6.33, 8.33, "o", color='purple')
    axes.annotate("New EQ (6.33, 8.33)", (6, 8.83))
    
    # random lines
    plt.plot([0,6.33], [8.34, 8.34], "--k")
    plt.plot([0,6], [8, 8], "--k")
    plt.plot([0,6], [9, 9], "--k")
    plt.plot([6,6], [0, 9], "--k")
    plt.plot([6.33,6.33], [0, 8.34], "--k")
    
    plt.legend()
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob12d.png",dpi=400)
    plt.close()
    

def prob13_ceil():
    """
    Plot demand / supply curve line
    """
    int1 = 20
    slope1 = -2
    int2 = 2
    slope2 = 1
    axes = plt.gca()
    axes.set_xlim([0, 12])
    axes.set_ylim([0, 22])
    # axes.set_xlim([4, 8])
    # axes.set_ylim([5, 13])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    # plt.plot(6, 8, "o", color='purple')
    # axes.annotate("EQ (6, 8)", (6, 8))
    plt.plot(x_vals, y_vals1, '-b', label="D")
    plt.plot(x_vals, y_vals2, '-r', label="S")
    
    # random lines
    plt.plot([0,3], [14, 14], "--k")
    plt.plot([0,4], [12, 12], "--k")
    plt.plot([0,6], [8, 8], "--k")
    plt.plot([0,7], [6, 6], "-g", label='P Ceil')

    
    plt.plot([6,6], [0, 8], "--k")
    plt.plot([3,3], [0, 14], "--k")
    plt.plot([4,4], [0, 12], "--k")
    plt.plot([7,7], [0, 6], "--k")
    
    plt.plot(6, 8, "o", color='purple')
    axes.annotate("Old EQ (6, 8)", (5.5, 8.5))
    plt.plot(4, 6, "o", color='purple')
    axes.annotate("New EQ (4, 6)", (3, 4.5))
    
    plt.legend()
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob13_ceil.png",dpi=400)
    plt.close()
    
    
def prob13_floor():
    """
    Plot demand / supply curve line
    """
    int1 = 20
    slope1 = -2
    int2 = 2
    slope2 = 1
    axes = plt.gca()
    axes.set_xlim([0, 12])
    axes.set_ylim([0, 22])
    # axes.set_xlim([4, 8])
    # axes.set_ylim([5, 13])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    # plt.plot(6, 8, "o", color='purple')
    # axes.annotate("EQ (6, 8)", (6, 8))
    plt.plot(x_vals, y_vals1, '-b', label="D")
    plt.plot(x_vals, y_vals2, '-r', label="S")
    
    # random lines
    plt.plot([0,10], [12, 12], "-g", label="P floor")
    plt.plot([0,6], [8, 8], "--k")
    plt.plot([0,4], [6, 6], "--k")

    
    plt.plot([6,6], [0, 12], "--k")
    plt.plot([10,10], [0, 12], "--k")
    plt.plot([4,4], [0, 12], "--k")
    
    plt.plot(6, 8, "o", color='purple')
    axes.annotate("Old EQ (6, 8)", (5.5, 8.5))
    plt.plot(4, 12, "o", color='purple')
    axes.annotate("New EQ (4, 12)", (3, 12.5))
    
    plt.legend()
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob13_floor.png",dpi=400)
    plt.close()


def prob13a():
    """
    Plot demand / supply curve line
    """
    int1 = 5
    slope1 = -0.05
    int2 = 1
    slope2 = 0.0033
    axes = plt.gca()
    axes.set_xlim([0, 110])
    axes.set_ylim([0, 6])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    plt.plot(75.0469, 1.24765, "o", color='purple')
    axes.annotate("EQ (75.04, 1.24)", (75.0469, 1.24765))
    plt.plot(x_vals, y_vals1, '-b')
    plt.plot(x_vals, y_vals2, '-r')
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob13a.png",dpi=400)
    plt.close()


def prob13b():
    """
    Plot demand / supply curve line
    """
    int1 = 5
    slope1 = -0.05
    int2 = 1
    slope2 = 0.0033
    axes = plt.gca()
    axes.set_xlim([0, 110])
    axes.set_ylim([0, 6])
    x_vals = np.array(axes.get_xlim())
    y_vals1 = int1 + slope1 * x_vals
    y_vals2 = int2 + slope2 * x_vals
    plt.plot(75.0469, 1.24765, "o", color='purple')
    axes.annotate("Old EQ (75.04, 1.24)", (75.0469, 1.36))
    plt.plot(50, 2.5, "o", color='purple')
    axes.annotate("New EQ (50, 2.5)", (50, 2.6))
    plt.plot(x_vals, y_vals1, '-b', label="D")
    plt.plot(x_vals, y_vals2, '-r', label="S")
    
    # random lines
    plt.plot([0,200], [2.5, 2.5], "--k")
    plt.plot([0,75.04], [1.247, 1.247], "--k")

    plt.plot([50,50], [0, 2.5], "-g", label="Quota")
    plt.plot([75.04, 75.04], [0, 1.247], "--k")
    
    plt.legend()
    plt.xlabel("Quantity", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.savefig("prob13b.png",dpi=400)
    plt.close()


    
# prob12()
# prob12c()
# prob12d()
# prob13_ceil()
# prob13_floor()
prob13a()
prob13b()