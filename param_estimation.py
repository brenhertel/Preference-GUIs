import numpy as np
import matplotlib.pyplot as plt

def ema(y, beta):
    """Exponentially weighted average."""
    n = len(y)
    zs = np.zeros(n)
    z = 0
    for i in range(n):
        z = beta * z + (1 - beta) * y[i]
        zs[i] = z
    return zs


def ema_debiased(y, beta):
    """Exponentially weighted average with bias correction."""
    n = len(y)
    zs = np.zeros(n)
    z = 0
    for i in range(n):
        z = beta * z + (1 - beta) * y[i]
        zc = z / (1 - beta ** (i + 1))
        zs[i] = zc
    return zs

def MLE_1D(y):
    """Maximum Likelihood Estimation"""
    n = len(y)
    zs = np.zeros(n)
    for i in range(n):
        zc = np.mean(y[:i+1])
        zs[i] = zc
    return zs
    

def bayesian_1d(y):
    """Maximum A Posteriori Estimation"""
    n = len(y)
    zs = np.zeros(n)
    zs[0] = y[0]
    sigma_m = 1
    sigma_v = 1
    for i in range(1, n):
        mu_MAP = (sigma_m**2 * y[i] + sigma_v**2 * zs[i-1]) / (sigma_m**2 + sigma_v**2)
        zs[i] = mu_MAP
        sigma_m = np.std(y[:i+1])
    return zs
    
def bayesian_2d():
    pass
    
def bayesian_3d():
    pass
    
if __name__ == '__main__':
    
    np.random.seed(0)
    n = 50
    x = np.arange(n) * np.pi
    y = np.cos(x) * np.exp(x / 100) - 10 * np.exp(-0.01 * x)
    
    est_ema = ema(y, 0.9)
    est_ema_debiased = ema_debiased(y, 0.9)
    mle = MLE_1D(y)
    map = bayesian_1d(y)
    
    plt.figure()
    plt.plot(x, y, "o-", c="black")
    plt.plot(x, est_ema, c="red", label="EMA")
    plt.plot(x, est_ema_debiased, c="orange", label="EMA with bias correction")
    plt.plot(x, mle, c="green", label="MLE")
    plt.plot(x, map, c="cyan", label="MAP")
    plt.title("beta = 0.9")
    plt.legend()
    plt.show()