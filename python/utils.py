import numpy as np 
import matplotlib.pyplot as plt

def generate_polynomial_data(coeffs, xvals):
    #coeffs = np array nx1
    #xvals contains list of x values, 1 dimensional
    #d = length of coeffs, n = length of xvals
    X = [] 
    Y = []
    for x in xvals:
        phix = np.array([[x**i for i in range(len(coeffs))]])
        X.append(phix)
        y = coeffs.T @ phix.T
        Y.append(y[0,0])
    X = np.concatenate(X, axis = 0)
    Y = np.array(Y)
    return X, Y

def plot_polynomial(coeffs):
    xvals = np.linspace(-10, 10, 2000)
    _, Y = generate_polynomial_data(coeffs, xvals)
    plt.plot(xvals, Y, 'm--')
    plt.title("Polynomial for coeffs: %s"%coeffs.T)
    plt.show()


plot_polynomial(np.array([[1, 3, -500, 10, 20]]).T)