# Small matplotlib demo plot.
import matplotlib.pyplot as plt
import numpy as np 
# x values 0.0, 0.2, ..., 4.8 for the demo curves
t = np.arange(0.,5.,0.2)


def plotThis():
    # Show a demo plot (red dots for the numbers, green triangles for t squared).
    plt.plot([1,2,3,4], 'ro', t, t**2, 'g^')
    plt.ylabel('some numbers')
    plt.show()
