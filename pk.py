#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Faz o espectro f(k) a partir de P(k)
	v0.8 - Plota tanto os k's quanto a matriz dos delta_x
	v0.9 - Usa valores do CAMB em P(k) e interpola eles -----PROBLEMA: Preciso transformar o meu grid em
		valores físicos
	v1.0 - é feita uma DFT em N dimensões, problemas quanto a normalização e unidades físicas
	v1.5 - A normalização está quase correta
	Arthur E. da Mota Loureiro
		12/12/2013
"""
import numpy as np
import pylab as pl
import grid3D as gr
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy import interpolate

N = 71									#Numero de celulas
vol_celula = 18.							#volume da celula
L = vol_celula*N							#Tamanho em Mpc

k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)		#pega o P(k) do Raul
k_r = np.insert(k_r,0,0.)						#insere P(k=0) = 0 antes de interpolar
P_k = np.insert(P_k,0,0.)						#mesma coisa que acima


k = gr.grid3d(N,N,N,L)
volume = np.power(L,3.)			#cria o grid e tudo mais de NxNxN
Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)

p_matrix =np.asarray([[[ Pk(k.matrix[i][j][n]) for i in range(len(k.k_x))] for j in range(len(k.k_y))] for n in range(len(k.k_z))])

def A_k(P_):
	return np.random.normal(0,np.sqrt(2.*P_*volume))		#distribuicao gaussiana media no zero E DESVIO SQRT(2*P_k)
	#return np.random.normal(0,(2.*P_*volume))		#distribuicao gaussiana media no zero E DESVIO SQRT(2*P_k)
									#incluído volume para fechar as unidades da maneira certa
def phi_k(P_): 
#	return (np.random.random(len(p_matrix)))*2.*np.pi - np.pi	#distr. homog. de -pi a +pi
	return (np.random.random(len(P_)))*2.*np.pi			#segundo Padmanabhan pg 191 ===> É A MESMA COISA
def delta_k(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))				#contraste de densidade em k

#print f_k(k.matrix)

delta_x = ((delta_k(p_matrix).size)/volume)*np.fft.ifftn(delta_k(p_matrix)).real
k_max_cal = np.sqrt(3.)*np.pi*(float(N)/L)
k_max_cal2 = np.sqrt(np.power(np.max(k.k_x),2)+np.power(np.max(k.k_y),2)+np.power(np.max(k.k_z),2))
print "################################################################"
print "                           DADOS                                "
print "################################################################"
print "Lado total: " + str(L) + " Mpc"
print "Lado depois das contas: " + str((np.power(delta_x.size,1./3)))
print "Volume total: " + str(volume) + " Mpc^3"
print "Numero de celulas de delta_x: " + str(len(delta_x))
print "Lado da celula: " + str(np.power(volume,1./3)/N) + " Mpc"	#para /\x = 10. Mpc
print "Celula (depois dos calculos): " + str( np.pi/np.max(k.matrix) )
print "alt1                        : " + str( np.pi/k_max_cal)
print "alt2                        : " + str( np.pi/k_max_cal2)
print "<delta_x^2> = " + str(np.std(delta_x))				#deve ser aprox. 0.9
print "k_max calculado: " + str(k_max_cal) + "----" + str(k_max_cal2)
print "k_max da matriz de k: " + str(np.max(k.matrix))
print "################################################################"
k.plot	
pl.colorbar()								#plota a matriz dos k's
pl.figure("P(k)")							#plotando o espectro original
pl.grid(1)
pl.loglog()
pl.xlabel("k")
pl.ylabel('P(k)')
pl.plot(k_r[1:], P_k[1:])
pl.plot(k_r[1:], Pk(k_r)[1:])
pl.axvline(x=np.max(k.matrix), linewidth=2., color='r')
pl.figure("Mapa")
pl.title("$\delta(x)_{i,0,k}$")
pl.imshow(delta_x[:,0,:], cmap=cm.jet)
pl.colorbar()
#pl.imshow(f_k(k.matrix)[0].real, cmap=cm.jet)
pl.grid(1)
pl.title('Fatia do $\delta_x$ gerado apos a ifft de $\delta_k$ com $P(k)$')
pl.show()


