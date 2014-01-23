#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Faz o espectro f(k) a partir de P(k)
	v0.8 - Plota tanto os k's quanto a matriz dos delta_x
	v0.9 - Usa valores do CAMB em P(k) e interpola eles -----PROBLEMA: Preciso transformar o meu grid em
		valores físicos
	Arthur E. da Mota Loureiro
		12/12/2013
"""
import numpy as np
import pylab as pl
import grid3D as gr
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy import interpolate

N = 71

k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)		#pega o P(k) do Raul
kmax = np.max(k_r)
k = gr.grid3d(N,N,N,kmax)							#cria o grid e tudo mais de NxNxN
Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)

p_matrix =np.asarray([[[ np.abs(Pk(k.matrix[i][j][n])) for i in range(N-1)] for j in range(N-1)] for n in range(N-1)])
"""
def P(k_):
        return np.abs(np.cos(k_)) + 1
       # return Pk(k_)
"""	
def A_k(P_):
	return np.random.normal(0,P_*2.)				#distribuicao gaussiana media no zero
def phi_k(P_): 
	return (np.random.random(len(p_matrix)))*2.*np.pi - np.pi	#distr. homog. de -pi a +pi
def delta_k(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))				#contraste de densidade em k

#print f_k(k.matrix)

delta_x = np.fft.ifft(delta_k(p_matrix))
delta_x = (2./len(delta_x))*delta_x
k.plot	
pl.colorbar()								#plota a matriz dos k's
pl.figure("P(k)")							#plotando o espectro original
pl.grid(1)
pl.loglog()
pl.xlabel("k")
pl.ylabel('P(k)')
pl.plot(k_r, P_k)
pl.plot(k_r, Pk(k_r))
pl.figure("Mapa")

pl.imshow(delta_x[0].real, cmap=cm.jet)
pl.colorbar()
#pl.imshow(f_k(k.matrix)[0].real, cmap=cm.jet)
pl.grid(1)
pl.title('Fatia do $\delta_x$ gerado apos a ifft de $\delta_k$ com $P(k)$')
pl.show()


