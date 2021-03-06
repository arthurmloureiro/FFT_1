#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Cria uma grid de k em 3D obedecendo a forma como se comporta
	na FFT do python
	v0.1
	v1.0 - Em 3D e plota o esquema como deveria ser
	Arthur E. da Mota Loureiro
		11/12/2013
"""
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


m=n=l= 51
kx0 = np.pi/5					#NENHUM MOTIVO PARA ESTE VALOR
ky0 = np.pi/5
kz0 = np.pi/5

prime_x=np.arange(1,(m+1)/2,1)*kx0		# vai ate (m+2/2) pq o python para um ponto antes, entao sem isso ele parava em m-1
invert_prime_x = -prime_x[::-1]			#inverte a ordem de prime e a deixa negativa
prime_x = np.insert(prime_x, 0,0)		#adiciona o valor zero na posição 0
k_x = np.append(prime_x,invert_prime_x)		#junta todos os vetores
print(len(k_x))

prime_y=np.arange(1,(n+1)/2,1)*ky0		# vai ate (m+2/2) pq o python para um ponto antes, entao sem isso ele parava em m-1
invert_prime_y = -prime_y[::-1]			#inverte a ordem de prime e a deixa negativa
prime_y = np.insert(prime_y, 0,0)		#adiciona o valor zero na posição 0
k_y = np.append(prime_y,invert_prime_y)		#junta todos os vetores
print(len(k_y))

prime_z=np.arange(1,(l+1)/2,1)*kz0		# vai ate (m+2/2) pq o python para um ponto antes, entao sem isso ele parava em m-1
invert_prime_z = -prime_z[::-1]			#inverte a ordem de prime e a deixa negativa
prime_z = np.insert(prime_z, 0,0)		#adiciona o valor zero na posição 0
k_z = np.append(prime_z,invert_prime_z)		#junta todos os vetores
print(len(k_z))

"""
checando algumas coisas e plotando gráficos
"""

cont1=cont2=0
for k in range(len(k_x)):
	if k_x[k] < 0:
		cont1=cont1+1
	else:
		cont2=cont2+1
pl.figure("os ks")	
pl.plot(k_x, '.', label='$k_{x0} = \pi / 5$')
pl.plot(k_y, '*', label='$k_{y0} = \pi / 5$')
pl.plot(k_z, '^', label='$k_{z0} = \pi / 5$')
pl.title('N='+str(m)+" quant de valores negativos: " + str(cont1) + " positivos: " + str(cont2))
pl.xlabel("s")
pl.ylabel("k")
pl.grid(1)
legend = pl.legend(loc='upper left', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
#pl.show()


'''
criando o grid 3D
'''
Grid3D = [[[ np.sqrt(k_x[i]**2 + k_y[j]**2 + k_z[k]**2) for i in range(len(k_x))] for j in range(len(k_y))] for k in range(len(k_z))]
print Grid3D[1][2][1]
print np.sqrt(k_x[1]**2 + k_y[2]**2 + k_z[1]**2)
"""
fig = pl.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(Grid3D[][][], cmap=cm.jet, linewidth=0.2)
pl.show()
"""
pl.figure("2")
pl.imshow(Grid3D[30], cmap=cm.jet)
pl.show()














