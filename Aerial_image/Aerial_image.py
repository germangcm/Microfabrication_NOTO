import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.io import loadmat
from scipy.ndimage import measurements
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Accuracy = 8           # Degree of accuracy (integer, higher means more time more accurate)
N = 2**Accuracy + 1    # Resolution
lamda = 0.1            # Wavelength (um)
TH = 0.4               # Threshold intensity (normalized)
NA = 0.85              # Numerical aperature
L = 0.09               # Smallest possible length in the technology
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Reading mask shape from file
File = 'example2.txt'
Y = np.loadtxt(File)
X = Y[:,0]
Y = Y[:,1]
num_points = len(X)-1
sides = np.zeros(num_points)
for i in range(num_points):
    sides[i] = np.sqrt((X[i]-X[i+1])**2 + (Y[i]-Y[i+1])**2)
scale = L/np.min(sides)
sides = sides*scale
X = scale*X
Y = scale*Y
X = X-np.min(X)
Y = Y-np.min(Y)
Lx = 1.4*np.max(X)    # Mask real dimensions (um)
Ly = 1.4*np.max(Y)    # Mask real dimensions (um)

# Center the polygon
x = np.linspace(-0.2*np.max(X),1.2*np.max(X),N)
y = np.linspace(-0.2*np.max(Y),1.2*np.max(Y),N)

# Create mask matrix
x,y = np.meshgrid(x,y)
x = np.reshape(x,-1)
y = np.reshape(y,-1)
mask = Path(np.column_stack((X,Y)))
mask = mask.contains_points(np.column_stack((x,y)))
mask = np.reshape(mask,(N,N))
mask = mask.astype(float)

# Calculations
# Creating different domains
dx = Lx/(N-1)
dy = Ly/(N-1)
nx, ny = np.meshgrid(np.arange(-(N-1)/2,(N-1)/2+1),np.arange(-(N-1)/2,(N-1)/2+1))
fx = (1/dx)*(1/N)*nx
fy = (1/dx)*(1/N)*ny
# Discrete frequency domain (1/um)
P = np.sqrt((fx**2)+(fy**2))
P = (P < (NA/lamda)).astype(float)
I = fftshift(fft2(mask))
I = ifft2(P*I)
I = np.real(I*np.conj(I))
I = I/np.max(I)
aerial = (I > TH).astype(float)

# Calculate error
mError = np.sum(np.abs(mask-aerial))
mError = 100*(mError/np.sum(mask))
print('error = ', mError, '%')

# Plot mask and aerial images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(mask, cmap='gray', origin='lower')
ax1.set_title('Mask image')
ax2.imshow(aerial, cmap='gray', origin='lower')
ax2.set_title('Aerial image')
plt.tight_layout()

# Plot normalized light intensity in 3D
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(nx, ny, I, cmap='jet', rstride=1, cstride=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Normalized light intensity')
ax.set_title('Normalized light intensity')
plt.show()

# Plot boundary
fig = plt.figure(figsize=(6, 6))
plt.plot(Bm[:, 1], Bm[:, 0], 'k', label='Mask')
plt.plot(Ba_M[:, 1], Ba_M[:, 0], 'r--', label='Aerial image')
plt.axis('equal')
plt.legend()
plt.show()




