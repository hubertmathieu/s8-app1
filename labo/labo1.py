import numpy as np
import matplotlib.pyplot as plt

def prob1():
    A = A = np.array([[3,4,1], [5,2,3], [6,2,2]], dtype=np.float32)
    B = np.random.rand(3, 3)

    N = 1000
    U = 0.001

    At = A.T

    I = np.eye(3)

    losses = []

    for i in range(N):
        loss = np.sum((B @ A - I) ** 2)
        losses.append(loss)
        
        G = 2 * (B @ A - I) @ At   # recompute gradient
        B = B - U * G

    print("B actual =\n", B)

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Frobenius norm squared)")
    plt.title("Convergence of Gradient Descent for Matrix Inverse")
    plt.grid(True)
    plt.show()

def prob2():
    x = np.array([-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04])
    y = [0.02, 0.03, -0.17, -0.12, -0.37, -0.25, -0.10, 0.14, 0.53, 0.71, 1.53]
    a = np.zeros((1,3))

    # powers = np.arange(11, 0)

    # X_i = x[:, np.newaxis] ** powers

    # Y_pred = a @ X_i.T


    xys = np.column_stack((x, y))

    N = 1000
    I = xys.shape[0]
    Ls = np.zeros(N)
    mu = 0.001
    L=0

    for n in range(N):
        L = 0
        dLda = np.zeros((1,3))

        for i in range(11):
            xi = xys[i, 0]
            yi = xys[i, 1]

            xvi = np.zeros((1,3))
            xvi[0, 0] = 1
            xvi[0,1] = xi
            xvi[0,2] = xi**2
            yip = a @ xvi.T

            L += (yip - yi)**2
            
            dLda += 2 * (yip - yi) * xvi

        Ls[n] = L
        a = a - mu * dLda

    xs = np.linspace(-1.25, +1.25, num=1001)
    yps = a[0,2] * xs**2 + a[0, 1] * xs + a[0,0]
    plt.scatter(xys[:,0], xys[:,1])
    plt.plot(xs, yps)
    plt.show()



 


prob2()

