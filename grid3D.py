#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Cria uma classe grid de k em 3D obedecendo a forma como se comporta
	na FFT do python
	v0.1
	v1.0 - Em 3D
	v1.5 - pode plotar fatias da matriz
	Arthur E. da Mota Loureiro
		12/12/2013
"""
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

class grid3d:
	'''
	a entrada é o tamanho dos vetores k_x, k_y, k_z
	'''
	def __init__(self,m,n,l):
		self.size_x = m
		self.size_y = n
		self.size_z = l

#		kx0 = np.pi/5					#NENHUM MOTIVO PARA ESTE VALOR
#		ky0 = np.pi/5
#		kz0 = np.pi/5
		
		kx0=(2.*np.pi)*1.1849/m		#k_max do espectro do CAMB
		ky0=(2.*np.pi)*1.1849/n
		kz0=(2.*np.pi)*1.1849/l
		
		prime_x=np.arange(1,(m+1)/2,1)*kx0		
		invert_prime_x = -prime_x[::-1]			
		prime_x = np.insert(prime_x, 0,0)		
		self.k_x = np.append(prime_x,invert_prime_x)		


		prime_y=np.arange(1,(n+1)/2,1)*ky0		
		invert_prime_y = -prime_y[::-1]			
		prime_y = np.insert(prime_y, 0,0)		
		self.k_y = np.append(prime_y,invert_prime_y)		


		prime_z=np.arange(1,(l+1)/2,1)*kz0		#
		invert_prime_z = -prime_z[::-1]			#inverte a ordem de prime e a deixa negativa
		prime_z = np.insert(prime_z, 0,0)		#adiciona o valor zero na posição 0
		self.k_z = np.append(prime_z,invert_prime_z)	#junta todos os vetores
		
		self.matrix = np.asarray([[[ np.sqrt(self.k_x[i]**2 + self.k_y[j]**2 +self.k_z[k]**2) for i in range(len(self.k_x))] for j in range(len(self.k_y))] for k in range(len(self.k_z))])
		pl.figure("Matriz de k")
		self.plot = pl.imshow(self.matrix[3], cmap=cm.jet)
		#pl.show()
