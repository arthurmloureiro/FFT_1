#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Cria uma classe grid de k em 3D obedecendo a forma como se comporta
	na FFT do python
	v0.1
	v1.0 - Em 3D
	v1.5 - pode plotar fatias da matriz
	v1.7 - não usa k_max e sim L (tamanho do lado da caixa)
	v2.0 - usa convencao de Einstein para gerar o grid
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
	def __init__(self,n_x,n_y,n_z,L_x,L_y,L_z):
		self.size_x = n_x
		self.size_y = n_y
		self.size_z = n_z
		self.Lx = L_x
		self.Ly = L_y
		self.Lz = L_z
		
		kx0 = (2*np.pi)/L_x				# k0 tem que ser este valor para que |k| < k_max
		ky0 = (2*np.pi)/L_y
		kz0 = (2*np.pi)/L_z
		
		
		prime_x=np.arange(1,(n_x/2+1),1)*kx0		# tem que ser até m/2 +1 por causa da estrutura do np.arange
		invert_prime_x = -prime_x[::-1]			
		prime_x = np.insert(prime_x, 0,0)		
		self.k_x = np.append(prime_x,invert_prime_x)		
		ident = np.ones_like(self.k_x)


		prime_y=np.arange(1,(n_y/2+1),1)*ky0		
		invert_prime_y = -prime_y[::-1]			
		prime_y = np.insert(prime_y, 0,0)		
		self.k_y = np.append(prime_y,invert_prime_y)		


		prime_z=np.arange(1,(n_z/2+1),1)*kz0		#
		invert_prime_z = -prime_z[::-1]			#inverte a ordem de prime e a deixa negativa
		prime_z = np.insert(prime_z, 0,0)		#adiciona o valor zero na posição 0
		self.k_z = np.append(prime_z,invert_prime_z)	#junta todos os vetores
		
		self.KX2 = np.einsum('i,j,k', self.k_x*self.k_x,ident,ident)
		self.KY2 = np.einsum('i,j,k', ident,self.k_y*self.k_y,ident)
		self.KZ2 = np.einsum('i,j,k', ident,ident,self.k_z*self.k_z)
		#self.matrix = np.asarray([[[ np.sqrt(self.k_x[i]**2 + self.k_y[j]**2 +self.k_z[k]**2) for i in range(len(self.k_x))] for j in range(len(self.k_y))] for k in range(len(self.k_z))])
		
		self.matrix = np.sqrt(self.KX2 + self.KY2 + self.KZ2)
		#self.hist, edge = np.histogramdd(np.array(self.matrix), bins=(n_x,n_y,n_z))
		pl.figure("Matriz de k")
		self.plot = pl.imshow(self.matrix[3], cmap=cm.jet)
		#self.plothist = pl.imshow(self.hist[3], cmap=cm.jet)
		#pl.show()
