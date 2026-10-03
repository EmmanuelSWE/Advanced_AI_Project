import matplotlib.pyplot as plt
import numpy as np 
t = np.arange(0.,5.,0.2)


def plotThis():
    plt.plot([1,2,3,4], 'ro', t, t**2, 'g^')
    plt.ylabel('some numbers')
    plt.show()
