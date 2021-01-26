
import matplotlib.pyplot as plt 
import numpy as np

def prob5a1():
    axes = plt.gca()
    axes.set_xlim([0, 15])
    axes.set_ylim([0, 15])
    x_vals = np.linspace(0.01, axes.get_xlim(), 100)
    U=5
    y_vals1 = (U**2) / x_vals
    U=4
    y_vals2 = (U**2) / x_vals
    U=3
    y_vals3 = (U**2) / x_vals
    U5, _ = plt.plot(x_vals, y_vals1, '--b', label="U=5")
    U4, _ = plt.plot(x_vals, y_vals2, '--g', label="U=4")
    U3, _ = plt.plot(x_vals, y_vals3, '--r', label="U=3")
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(handles=[U5, U4, U3])
    plt.savefig("U_func1.png",dpi=400)
    plt.close()


def prob5a2():
    axes = plt.gca()
    axes.set_xlim([0, 4])
    axes.set_ylim([0, 16])
    x_vals = np.linspace(0.01, axes.get_xlim(), 100)
    U=5
    y_vals1 = U - 3 * np.sqrt(x_vals)
    U=10
    y_vals2 = U - 3 * np.sqrt(x_vals)
    U=15
    y_vals3 = U - 3 * np.sqrt(x_vals)
    U5, _ = plt.plot(x_vals, y_vals1, '--b', label="U=5")
    U10, _ = plt.plot(x_vals, y_vals2, '--g', label="U=10")
    U15, _ = plt.plot(x_vals, y_vals3, '--r', label="U=15")
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(handles=[U5, U10, U15])
    plt.savefig("U_func2.png",dpi=400)
    plt.close()


def prob5c():
    axes = plt.gca()
    axes.set_xlim([0, 22])
    axes.set_ylim([0, 22])
    x_vals = np.linspace(0.01, axes.get_xlim(), 100)
    I_vals = 10 - (1/2) * x_vals
    I, _ = plt.plot(x_vals, I_vals, '--b', label="I")
    # U = y / x = 1/2
    U = 5
    y_vals5 = (U**2) / x_vals
    U = 6
    y_vals6 = (U**2) / x_vals
    U = 8
    y_vals8 = (U**2) / x_vals
    U = 9
    y_vals9 = (U**2) / x_vals
    U = 7.07
    y_vals_f = (U**2) / x_vals
    U5, _ = plt.plot(x_vals, y_vals5, '--k', label="U=5")
    U6, _ = plt.plot(x_vals, y_vals6, '--k', label="U=6")
    U8, _ = plt.plot(x_vals, y_vals8, '--k', label="U=8")
    U9, _ = plt.plot(x_vals, y_vals9, '--k', label="U=9")
    Uf, _ = plt.plot(x_vals, y_vals_f, '--r', label="U=7.07")
    plt.plot(10, 5, "o", color='purple')
    axes.annotate("Consumptuon = (X=10, Y=5)", (11, 9))
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(handles=[I, Uf, U5, U6, U8, U9])
    plt.savefig("I_func1.png",dpi=400)
    plt.close()
    
    
    axes = plt.gca()
    axes.set_xlim([0, 22])
    axes.set_ylim([0, 22])
    x_vals = np.linspace(0.01, axes.get_xlim(), 100)
    I_vals = 10 - (1/2) * x_vals
    I, _ = plt.plot(x_vals, I_vals, '--b', label="I")
    # U = y / x = 1/2
    U = 12
    y_vals12 = U - 3 * np.sqrt(x_vals)
    U = 13
    y_vals13 = U - 3 * np.sqrt(x_vals)
    U = 16
    y_vals16 = U - 3 * np.sqrt(x_vals)
    U = 17
    y_vals17 = U - 3 * np.sqrt(x_vals)
    U = 14.5
    y_vals_f = U - 3 * np.sqrt(x_vals)
    U12, _ = plt.plot(x_vals, y_vals12, '--k', label="U=11")
    U13, _ = plt.plot(x_vals, y_vals13, '--k', label="U=12")
    U16, _ = plt.plot(x_vals, y_vals16, '--k', label="U=14")
    U17, _ = plt.plot(x_vals, y_vals17, '--k', label="U=15")
    Uf, _ = plt.plot(x_vals, y_vals_f, '--r', label="U=14.5")
    plt.plot(9, 5.5, "o", color='purple')
    axes.annotate("Consumptuon = (X=9, Y=5.5)", (10, 8))
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(handles=[I, Uf, U12, U13, U16, U17])
    plt.savefig("I_func2.png",dpi=400)
    plt.close()
    

    
# prob5a1()
prob5a2()
prob5c()
