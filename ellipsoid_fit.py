'''This script performs ellipsoid fitting, plotting and distance computation
   on the prostate point cloud data using Grammalidis's constraint method.
   by ZHUO Xu and LU Yuchen @ SEU, 2023
'''

import numpy as np
import scipy.linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt
import cma
from termcolor import colored
import os

np.set_printoptions(precision=4)
# Create the output directories.
os.makedirs('./outcmaes/history', exist_ok=True)


def readData(path='./ProstateSurfacePoints.xlsx'):
    '''Read the provided Excel data.
    '''
    data = np.array(pd.read_excel(path))
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

    return X, Y, Z


def metaInfoFromGeneral(v):
    '''Retrieve the center, tilt and radii of the ellipsoid 
       from the general form parameters vector v = [A~J].
    '''
    # coefficient matrix
    v = v.astype(np.float64)
    A, B, C, D, E, F, G, H, I, J = v
    Ahat = np.array([
        [A, D / 2, E / 2, G / 2],
        [D / 2, B, F / 2, H / 2],
        [E / 2, F / 2, C, I / 2],
        [G / 2, H / 2, I / 2, J],
    ])
    A = Ahat[:3, :3]

    # center
    c = np.linalg.inv(-A[:3, :3]) @ (v[6:9] / 2)

    # radii and tilt
    T = np.eye(4)
    T[:3, 3] = c
    AcircHat = T.T @ Ahat @ T
    Aprim = AcircHat[:3, :3]
    Jprim = AcircHat[3, 3]
    lambdas, vs = np.linalg.eig(Aprim / Jprim)
    vs = vs.T
    radii = np.sqrt(1 / np.abs(lambdas))

    return c, vs, radii


# Step 1 - Fit the ellipsoid
def fitEllipsoidGrammalidis(X, Y, Z):
    '''Fit the ellipsoid using Grammalidis's constraint method.
       Return parameters vector v = [A~J].
    '''
    # Build the 10x10 constraint matrix.
    C = np.zeros((10, 10), dtype=np.float64)
    C[0, 1] = C[1, 0] = 2  # 4AB
    C[3, 3] = -1  # -D^2

    # Solve the generalized eigen decomposition.
    #             A     B     C     D      E      F      G  H  I  J
    D = np.array([X**2, Y**2, Z**2, X * Y, X * Z, Y * Z, X, Y, Z, np.ones(len(X))], dtype=np.float64).T
    S = D.T @ D
    eigenVals, eigenVecs = linalg.eig(S, C)

    # Find the minimum positive eigenvalue and corresponding eigenvector.
    idx = 0
    minPositive = np.inf
    for i in range(10):
        val = eigenVals[i]
        if not np.isinf(val) and val > 0 and val < minPositive:
            minPositive = val
            idx = i
    v = eigenVecs[:, idx]

    # Perform the post-check.
    A, B, C, D, E, F, G, H, I, J = v
    matA = np.array([
        [A, D / 2, E / 2],
        [D / 2, B, F / 2],
        [E / 2, F / 2, C],
    ])
    check1 = 4 * A * B - D**2
    check2 = (A + B) * np.linalg.det(matA)
    assert check1 > 0, 'Post-check failed for 4AB-D^2>0.'
    assert check2 > 0, 'Post-check failed for (A+B)|matA|>0.'

    return v, check1, check2


print(colored('# Step 1 - Fit the ellipsoid', 'blue'))
X, Y, Z = readData()
v, check1, check2 = fitEllipsoidGrammalidis(X, Y, Z)
print('Parameter Vector v = [A~J]')
print(v)
print('Check 1 and check 2: ', check1, check2)
center, tilt, radii = metaInfoFromGeneral(v)
print('Center: ', center)
print('Tilt: ', tilt)
print('Radii: ', radii)


# Step 2 - Plot the ellipsoid and data points.
def plotSuperEllipsoid(center, radii, tilt, ax, e1=1, e2=1, method='wireframe'):
    '''Plot an ellipsoid or super-ellipsoid.
    '''
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)

    # Use the parameteric form to draw the origin-centered axes-aligned ellipsoid.
    a, b, c = radii
    x = a * np.outer(np.sin(theta)**e1, np.cos(phi)**e2)
    y = b * np.outer(np.sin(theta)**e1, np.sin(phi)**e2)
    z = c * np.outer(np.cos(theta)**e1, np.ones_like(phi))

    # Rotate, and translate to center.
    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j], y[i, j], z[i, j] = np.dot([x[i, j], y[i, j], z[i, j]], tilt) + center

    # plot ellipsoid
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if method == 'wireframe':
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='g', alpha=0.2)
    elif method == 'surface':
        ax.plot_surface(x, y, z, color='g', alpha=0.4)
    else:
        raise 'Invalid method: ' + method


print(colored('# Step 2 - Plot the ellipsoid and data points', 'blue'))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the data points scatter.
ax.scatter(X, Y, Z, s=3, c='b')
# Plot the ellipsoid.
plotSuperEllipsoid(center, radii, tilt, ax)
print('Plot shown.')
plt.savefig('./img/ellipsoid.png', bbox_inches='tight')
plt.show()


# Step 3 - Compute the distance from data points to the ellipsoid or super-ellipsoid.
def distanceToSuperEllipsoid(x, y, z, center, radii, e1=1, e2=1):
    '''Compute the distance from a data point (x, y, z) to the ellipsoid or super-ellipsoid
       surface (axes-aligned).
    '''
    x0, y0, z0 = center
    a, b, c = radii

    # translate so that the start point is the origin
    x, y, z = x - x0, y - y0, z - z0
    # length of the vector
    norm = np.linalg.norm(np.array([x, y, z]))
    # parameters (map to Octant I)
    theta = np.arccos(np.abs(z) / norm)
    phi = np.arctan2(np.abs(y), np.abs(x))

    # the intersection point (always inside Octant I)
    xp = a * np.sin(theta)**e1 * np.cos(phi)**e2 + x0
    yp = b * np.sin(theta)**e1 * np.sin(phi)**e2 + y0
    zp = c * np.cos(theta)**e1 + z0
    # map the data point to Octant I
    xq = np.abs(x) + x0
    yq = np.abs(y) + y0
    zq = np.abs(z) + z0
    # compute the distance
    distance = np.sqrt((xq - xp)**2 + (yq - yp)**2 + (zq - zp)**2)

    return distance


print(colored('# Step 3 - Compute the distance to ellipsoid', 'blue'))
distancesEllipsoid = np.array([distanceToSuperEllipsoid(X[i], Y[i], Z[i], center, radii) for i in range(len(X))])
print('Mean Distance (Data to Ellipsoid): {:.4f}'.format(np.mean(distancesEllipsoid)))
print('RMSE Distance (Data to Ellipsoid): {:.4f}'.format(np.sqrt(np.mean(distancesEllipsoid**2))))


# Step 4 - Fit the super-ellipsoid using CMA-ES optimizer.
def fitSuperEllipsoidCMAES(X, Y, Z, center, radii, metric='mean'):
    '''Fit the super-ellipsoid starting from the ellipsoid using CMA-ES.
    '''
    # Parameters.
    initE1E2 = [1, 1]
    initSigma = 1
    options = {
        'maxiter': 1e6,
        'verb_disp': 10,  # print every 10 epochs
    }

    # Define the error function (mean distance).
    def errorFunction(x):
        e1, e2 = x
        distances = []
        for i in range(len(X)):
            distances.append(distanceToSuperEllipsoid(X[i], Y[i], Z[i], center, radii, e1, e2))
        distances = np.array(distances)

        if metric == 'mean':
            return np.mean(distances)
        elif metric == 'rmse':
            return np.sqrt(np.mean(distances**2))
        else:
            raise 'invalid metric: '+ metric

    # Run the optimizer.
    result = cma.fmin(errorFunction, initE1E2, initSigma, options)
    prediction = result[0]  # the optimal e1, e2
    predictionError = errorFunction(prediction)

    return prediction, predictionError


print(colored('# Step 4 - Fit the super-ellipsoid using CMA-ES optimizer', 'blue'))
prediction, predictionError = fitSuperEllipsoidCMAES(X, Y, Z, center, radii, 'rmse')
print('Prediction of (e1, e2)', prediction)
# Compute the distances again.
distancesSuperEllipsoid = np.array(
    [distanceToSuperEllipsoid(X[i], Y[i], Z[i], center, radii, prediction[0], prediction) for i in range(len(X))])
print('Mean Distance (Data to Super-Ellipsoid): {:.4f}'.format(np.mean(distancesSuperEllipsoid)))
print('RMSE Distance (Data to Super-Ellipsoid): {:.4f}'.format(np.sqrt(np.mean(distancesSuperEllipsoid**2))))


# Plot the evolution of optimization.
def readE1E2History(path='./outcmaes/xmean.dat'):
    '''Read and parse the output file of CMA-ES that records every (e1, e2) during the optimization.
    '''
    with open(path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines[1:]:
        columns = line.strip().split()
        data.append([float(column) for column in columns[-2:]])

    return np.array(data).T


def filterOctantI(X, Y, Z, center):
    '''Pick those points in Octant I.
    '''
    x0, y0, z0 = center
    idx = np.argwhere((X - x0 > 0) & (Y - y0 > 0) & (Z - z0 > 0))

    return X[idx], Y[idx], Z[idx]


e1s, e2s = readE1E2History('./outcmaes/xmean.dat')
for i in range(len(e1s)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*(filterOctantI(X, Y, Z, center)))
    plotSuperEllipsoid(center, radii, np.eye(3), ax, e1s[i], e2s[i], 'surface')
    plt.savefig(f'./outcmaes/history/epoch_{i}.png')
    plt.close()

# Done.