
import pdb
import matplotlib.pyplot as plt 
import numpy as np

def prob7():
    axes = plt.gca()
    axes.set_xlim([0, 110000])
    axes.set_ylim([0, 118000])
    x_vals = np.linspace(0, 100000, 1000)
    y_vals = np.array([slope(x) for x in x_vals])
    plt.plot(x_vals, y_vals, '-k', label="I")
    
    U_1 = 119000 - 200 * np.sqrt(x_vals)
    U_11 = 103000 - 200 * np.sqrt(x_vals)
    U_111 = 135000 - 200 * np.sqrt(x_vals)
    U_2 = 119000 - 700 * (x_vals-40000)**0.47
    U_22 = 119000 - 700 * (x_vals-30000)**0.47
    U_222 = 119000 - 700 * (x_vals-50000)**0.47
    # U_2 = [90000 - 600 * (x-38500)**0.44 if x > 38500 else 1000000 for x in x_vals]
    # U_22 = [110000 - 600 * (x-46500)**0.44 if x > 46500 else 1000000 for x in x_vals]
    # U_222 = [70000 - 600 * (x-30500)**0.44 if x > 30500 else 1000000 for x in x_vals]
    # U_2 = 215000 - 700 * np.sqrt(x_vals)
    # U_22 = 195000 - 700 * np.sqrt(x_vals)
    # U_222 = 235000 - 700 * np.sqrt(x_vals)
    # U_222 = np.array([U2(x, -10000) for x in x_vals])
    plt.plot(x_vals, U_1, '-b', label="U_1")
    plt.plot(x_vals, U_11, '--b')
    plt.plot(x_vals, U_111, '--b')
    plt.plot(x_vals, U_2, '-r', label="U_2")
    plt.plot(x_vals, U_22, '--r')
    plt.plot(x_vals, U_222, '--r')
    # plt.plot(x_vals, U_222, '--r')
    plt.xlabel("This Years Consumption", fontsize=12)
    plt.ylabel("Next Years Consumption", fontsize=12)
    plt.legend()
    # plt.legend(handles=[I, Uf, U12, U13, U16, U17])
    plt.savefig("prob7.png",dpi=400)
    plt.close()

def slope(borrow):
    budget = 110000
    if borrow <= 25000:
        return budget - (1.1)*borrow
    else:
        return budget - (1.1)*25000 - (1.2) * (borrow-25000) 

def U2(x_val):
    if x_val > 65000:
        return 10000**2 / (x_val-65000) + 12800
    else:
        return 100000000
        
def U22(x_val):
    if x_val > 65000:
        return 10000**2 / (x_val-65000) + 12800
    else:
        return 100000000


prob7()
