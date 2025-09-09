#Programa python visualizacion de contenedores
#Contenedores tienen un total de 20 posiciones Solo con nivel 1 A

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#transform data from excel files to dataframes. Then join both tables

#Estado camara (cualquier camara por ahora) (actualizar diariamente)

dfCamaraDia = pd.read_excel('data/CONTENEDOR.xlsx')

#print(dfCamaraDia)

#Archivo con coordenadas (x, y, z) segun modelo por cada posicion
dfCoordenadas = pd.read_excel('data/Coordenadas_Contenedor_Modelo.xlsx')
#print(dfCoordenadas)

posicion = dfCamaraDia['Posicion'][0]
#print("Es el Contenedor "+posicion[2]+"!")

print("Joining dataframes!")

dfJoined = dfCoordenadas.join(dfCamaraDia, None, 'left', lsuffix= "new")
#print(dfJoined)

#printing df of interest

#print(dfJoined[['Coordenada x', 'Coordenada y', 'Coordenada z', 'Pallets', 'Posicion']].head(20))

print("Joined dataframes!")

#Looping through dataframes


def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    # ::2 is a slice of numpy, returning from all index but every second value
    data_e[::2, ::2, ::2] = data
    return data_e

# build up the numpy logo

#(4,3,4)  (largo, ancho, alto) (x, y,z)  dimensiones
#numpy array of zeros. initialize all to false
n_voxels = np.zeros((10, 2, 1), dtype=bool)

#count para registrar cantidad de posiciones vacias en contenedor
vacias = 0

# now use n_voxels[x,y,z] = Pallets to assign the color of the location


for ind in dfJoined.index:
        #print(ind)
        #print(dfJoined['Coordenada x'][ind], dfJoined['Coordenada y'][ind], dfJoined['Coordenada z'][ind])
        n_voxels[dfJoined['Coordenada x'][ind], dfJoined['Coordenada y'][ind], dfJoined['Coordenada z'][ind]] = dfJoined['Pallets'][ind]
        if((dfJoined['Pallets'][ind]) == 0):
            vacias+=1


'''
#todos los cubos largo 0, ancho 0 son true (or 1) (amarillo)
n_voxels[0, 0, 0] = 1
#todos los cubos largo -1, ancho 0 son true (amarillo). -1 refleja last dimension of array
n_voxels[0, 1, 0] = True
#el cubo (1,0,2) tambien es amarillo
n_voxels[1, 1, 0] = True

#el cubo (2,0,1) tambien es amarillo. Todos los demas son azules
n_voxels[5, 0, 0] = True
'''


#return colors for true values in array
#amarillo, azul '#FFD65DC0', '#7A88CCC0'

#ahora rojo, verde transparente

facecolors = np.where(n_voxels, '#f44336', '#ffffff')
#gaps. amarillo oscuro, azul osuro
#'#BFAB6E', '#7D84A6'

edgecolors = np.where(n_voxels, '#f88e86', '#e5f6f0')

filled = np.ones(n_voxels.shape)

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

# Shrink the gaps
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2

#a[start:end:step]

x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

# len(y) = 28
# Haciendo prints 
#print(len(y))
#print(y[25, 2, :])

#espacio central en camara (dimension y)
y[:, 2::1, :] += 0.95

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
ax.set_aspect('equal')

#print(type(ax))

plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.yticks([1,2])

plt.xlabel("Fila")
plt.ylabel("Profundidad")
plt.title("Posiciones Utilizadas Contenedor "+posicion[2]+".\n Hay " + str(vacias) +" (de 20) posiciones vacías (en blanco)")

ax.set_yticklabels(['A', 'A'])

ax.set_zlabel('Altura')
ax.set_zticks([1])

print("Mostrando figura del Contenedor")

plt.show()

#msg = "python in VSCODE!"

#print(msg)