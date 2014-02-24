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
import sys
import pylab as pl
import grid3D as gr
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy import interpolate
#lado = np.power((4.*np.pi*8.*8.*8.)/3,1./3)
#lado = 8./np.sqrt(3.)
lado = 10.0
########################DADOS INICIAIS###################################
N_x = 81 ; N_y = 81 ; N_z = 81						#Numero de celulas em x,y,z
d_x = lado ; d_y = lado ; d_z = lado					#volume da celula em x,y,z em Mpc
l_x = d_x*N_x ; l_y = d_y*N_y ; l_z = d_z*N_z				#Tamanho da caixa em Mpc em x,y,z
#########################################################################
k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)		#pega o P(k) do Raul
k_r = np.insert(k_r,0,0.)						#insere P(k=0) = 0 antes de interpolar
P_k = np.insert(P_k,0,0.)						#mesma coisa que acima

k = gr.grid3d(N_x,N_y,N_z,l_x,l_y,l_z)					#cria o grid com qualquer tamanho em 3D
volume = l_x*l_y*l_z							

Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)			#interpola os dados do CAMB

p_matrix =np.asarray([[[ Pk(k.matrix[i][j][n]) for i in range(len(k.k_x))] for j in range(len(k.k_y))] for n in range(len(k.k_z))])


def A_k(P_):
	return np.random.normal(0,np.sqrt(P_*volume))		#distribuicao gaussiana media no zero E DESVIO SQRT(2*P_k)
									#incluído volume para fechar as unidades da maneira certa
def phi_k(P_): 
#	return (np.random.random(len(p_matrix)))*2.*np.pi - np.pi	#distr. homog. de -pi a +pi
	return (np.random.random(len(P_)))*2.*np.pi			#segundo Padmanabhan pg 191 ===> É A MESMA COISA

def delta_k(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))				#contraste de densidade em k



############################# FFT #######################################
delta_x = ((delta_k(p_matrix).size)/volume)*np.fft.ifftn(delta_k(p_matrix))

print "Campo complexo: " + str(delta_x[1,2,3])  

delta_xr = delta_x.real
delta_xi = delta_x.imag

print "Parte real: " + str(delta_xr[1,2,3])  + " e parte imaginaria: " + str(delta_xi[1,2,3])

#########################################################################

#k_max_cal = np.sqrt(np.power(np.max(k.k_x),-2)+np.power(np.max(k.k_y),-2)+np.power(np.max(k.k_z),-2))
print "################################################################"
print "                           DADOS                                "
print "################################################################"
print "Lx: " + str(l_x) + " Mpc // Ly: " + str(l_y) + " Mpc // Lz: " + str(l_z) + " Mpc"
#print "Lado depois das contas: " + str((np.power(delta_x.size,1./3)))  #Não faz sentido se não é um cubo
print "Volume total: " + str(volume) + " Mpc^3"
print "Numero de celulas de delta_x: " + str(len(delta_x))
print "Lado da celula: " + str(np.power(volume,1./3)/len(delta_x)) + " Mpc"	#para /\x = 10. Mpc
#print "raio da celula (dados iniciais)    : " + str( np.sqrt(d_x**2 + d_y**2 + d_z**2) )
#print "raio da celula (usando a matriz K) : " + str( np.pi/np.max(k.matrix) ) + " Mpc"
#print "raio da celula (usando a definição): " + str( np.pi/k_max_cal) + " Mpc"
print "<delta_rr^2> = " + str(np.std(delta_xr))				#deve ser aprox. 0.9
print "<delta_ri^2> = " + str(np.std(delta_xi))				#deve ser aprox. 0.9

#print "k_max calculado: " + str(k_max_cal) 
print "k_max da matriz de k: " + str(np.max(k.matrix))
sys.exit(-1)

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


