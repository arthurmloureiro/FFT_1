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

N = 128
k = gr.grid3d(N,N,N)							#cria o grid e tudo mais de NxNxN
k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)		#pega o P(k) do Raul
Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)
#kk = k.matrix*k.matrix
#p_matrix = [[[Pk(k.matrix[i][j][m]) for i in range(N-1)] for j in range(N-1)] for m in range(N-1)] 
"""
def P(k_):
	p_matrix = [[[Pk(k_[i][j][m]) for i in range(N-1)] for j in range(N-1)] for m in range(N-1)] 
	return p_matrix
"""
def P(k_):
        return np.abs(np.cos(k_)) + 1
       # return Pk(k_)
	
def A_k(P_):
	return np.random.normal(0,P(P_)*2.)				#distribuicao gaussiana media no zero
def phi_k(P_): 
	return (np.random.random(len(A_k(P_))))*2.*np.pi - np.pi	#distr. homog. de -pi a +pi
def delta_k(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))				#contraste de densidade em k

#print f_k(k.matrix)
delta_x = np.fft.ifft(delta_k(k.matrix)).real
k.plot									#plota a matriz dos k's
pl.figure("P(k)")							#plotando o espectro original
pl.grid(1)
pl.loglog()
pl.xlabel("k")
pl.ylabel('P(k)')
pl.plot(k_r, P_k)
pl.figure("Mapa")

pl.imshow(delta_x[0], cmap=cm.jet)
#pl.imshow(f_k(k.matrix)[0].real, cmap=cm.jet)
pl.grid(1)
pl.title('Fatia do $f_x$ gerado apos a ifft de $f_k$ com $P(k) = |cos(k)| + 1$')
pl.show()


