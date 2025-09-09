import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)

#print(air_quality.head())

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# build up the numpy logo


#(4,3,4)  (largo, ancho, alto) (x, y,z)  dimensiones
#numpy array of zeros. initialize all to false
n_voxels = np.zeros((4, 3, 4), dtype=bool)
#todos los cubos largo 0, ancho 0 son true (amarillo)
n_voxels[0, 0, :] = True
#todos los cubos largo -1, ancho 0 son true (amarillo). -1 refleja last dimension of array 
n_voxels[-1, 0, :] = True
#el cubo (1,0,2) tambien es amarillo
n_voxels[1, 0, 2] = True

n_voxels[2, 2, 2] = True

#el cubo (2,0,1) tambien es amarillo. Todos los demas son azules
n_voxels[2, 0, 1] = True
#return colors for true values in array
#amarillo, azul
facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
#gaps. amarillo oscuro, azul osuro
edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')

filled = np.ones(n_voxels.shape)

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

# Shrink the gaps
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
ax.set_aspect('equal')

plt.show()

#msg = "python in VSCODE!"

#print(msg)