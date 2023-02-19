import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.io import loadmat
from scipy.interpolate import interp2d
from scipy.optimize import minimize, Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import trust_constr
from scipy.optimize import NewtonConstraint
from scipy.optimize import SLSQP
from scipy.optimize import NonlinearConstraint
from scipy import stats
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.qhull import Delaunay
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Parameters
Accuracy = 8  # Degree of accuracy (integer, higher means more time more accurate)
N = 2**Accuracy + 1  # Resolution
lamda = 0.1  # Wavelength (um)
TH = 0.4  # Threshold intensity (normalized)
NA = 0.85  # Numerical aperature
L = 0.09  # Smallest possible length in the technology

# Reading mask shape from file
File, Path = uigetfile("*.txt")
Y = importdata([Path, File])
X = Y[:, 0]
Y = Y[:, 1]
num_points = len(X) - 1
sides = zeros(num_points, 1)
for i in range(num_points):
    sides[i] = sqrt((X[i] - X[i + 1])**2 + (Y[i] - Y[i + 1])**2)
scale = L / min(sides)
sides = sides * scale
X = scale * X
Y = scale * Y
X = X - min(X)
Y = Y - min(Y)
Lx = 1.4 * max(X)  # Mask real dimensions (um)
Ly = 1.4 * max(Y)  # Mask real dimensions (um)
# Center the polygon
x = linspace(-0.2 * max(X), 1.2 * max(X), N)
y = linspace(-0.2 * max(Y), 1.2 * max(Y), N)
# Create mask matrix
[x, y] = meshgrid(x, y)
x = reshape(x, [], 1)
y = reshape(y, [], 1)
mask = inpolygon(x, y, X, Y)
mask = reshape(mask, N, N)
mask = double(mask)

# Calculations
# Creating different domains
dx = Lx / (N - 1)
dy = Ly / (N - 1)
[nx, ny] = meshgrid(-(N - 1) / 2 : (N - 1) / 2, -(N - 1) / 2 : (N - 1) / 2)  # Create discritized domain
fx = (1 / dx) * (1 / N) * nx  # Discrete frequency domain (1/um)
fy = (1 / dx) * (1 / N) * ny
I = fft2(mask)
I = fftshift(I)
# objective lens pupil function
P = sqrt((fx**2) + (fy**2))
P = double(P < (NA / lamda))
I = ifft2(P * I)
I = real(I * conj(I))
I = I / max(max(I))
aerial = double(I > TH)

# Calculate error
mError = sum(abs(mask - aerial))
mError = 100 * (mError / sum(mask))
print(f"error = {mError}%")

# Plotting
# figure(),imagesc(mask)
# axis('equal'); title('Mask image')
# figure(),imagesc(aerial)
# axis('equal'); title('Aerial image')