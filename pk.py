1#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Faz o espectro f(k) a partir de P(k)
	v0.8 - Plota tanto os k's quanto a matriz dos delta_x
	v0.9 - Usa valores do CAMB em P(k) e interpola eles -----PROBLEMA: Preciso transformar o meu grid em
		valores físicos
	v1.0 - é feita uma DFT em N dimensões, problemas quanto a normalização e unidades físicas
	v1.5 - A normalização está quase correta
	v2.0 - Faltava normalizar por sqrt(2) ou trocar 2*P(k) por P(k) na dist. Gaussiana
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
	return np.random.normal(0,np.sqrt(P_*volume))			#dist gaussiana media no zero E DESVIO SQRT(P_k)
									# Tiramos o fator de 2, deu certo! 
									#incluído volume para fechar as unidades 
def phi_k(P_): 
	return (np.random.random(len(P_)))*2.*np.pi			#segundo Padmanabhan pg 191 ===> É A MESMA COISA

def delta_k(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))				#contraste de densidade em k
"""
			organizando em bins
"""
def heav(x):								#funcao heaviside
	if x==0:
		return 0.5
	return 0 if x<0 else 1
heav_vec = np.vectorize(heav)						#heaviside vetorizada
n_bins = 80
k_bar = np.arange(0,n_bins,1)*(np.max(k.matrix)/n_bins)
#M = np.zeros((len(k_bar),N_x,N_y,N_z))
M = np.asarray([heav_vec(k_bar[a+1]-k.matrix[:,:,:])*heav_vec(k.matrix[:,:,:]-k_bar[a])for a in range(len(k_bar)-1)])
############################# FFT #######################################
delta_x = ((delta_k(p_matrix).size)/volume)*np.fft.ifftn(delta_k(p_matrix))
#print "Campo complexo: " + str(delta_x[1,2,3])  
delta_xr = delta_x.real
delta_xi = delta_x.imag
#print "Parte real: " + str(delta_xr[1,2,3])  + " e parte imaginaria: " + str(delta_xi[1,2,3])
#########################################################################

print "################################################################"
print "                           DADOS                                "
print "################################################################"
print "Lx: " + str(l_x) + " Mpc // Ly: " + str(l_y) + " Mpc // Lz: " + str(l_z) + " Mpc"
print "Lado depois das contas: " + str((np.power(delta_x.size,1./3)))  #Não faz sentido se não é um cubo
print "Volume total: " + str(volume) + " Mpc^3"
print "Numero de celulas de delta_x: " + str(len(delta_x))
print "Lado da celula: " + str(np.power(volume,1./3)/len(delta_x)) + " Mpc"	#para /\x = 10. Mpc
print "<delta_r^2> REAL = " + str(np.std(delta_xr))				#deve ser aprox. 0.9
print "<delta_r^2> IMAG = " + str(np.std(delta_xi))				#deve ser aprox. 0.9
print "k_max da matriz de k: " + str(np.max(k.matrix))
#sys.exit(-1)
print "################################################################"

######################### Procurando P(k) ###############################
d_k = (volume/(N_x*N_y*N_z))*np.fft.fftn(delta_xr)				# ifft de d_x.real
P_a = np.einsum("aijl,ijl,ijl->a", M, d_k,np.conj(d_k))

"""
				PLOTS
"""
k.plot	
pl.colorbar()								#plota a matriz dos k's
pl.figure("P(k)")							#plotando o espectro original
pl.grid(1)
pl.loglog()
pl.xlabel("k")
pl.ylabel('P(k)')
pl.plot(k_r[1:], P_k[1:])
pl.plot(k_r[1:], Pk(k_r)[1:])
pl.plot(P_a)
pl.axvline(x=np.max(k.matrix), linewidth=2., color='r')

pl.figure("Mapa")
pl.title("$\delta(x)_{i,0,k}$")
pl.imshow(delta_xr[:,0,:], cmap=cm.BuPu)
pl.colorbar()
pl.grid(1)
pl.title('Fatia do $\delta_x$ gerado apos a ifft de $\delta_k$ com $P(k)$')

pl.figure("Bins de K")
numb_bin =10
pl.title("Mostrando bin numero " + str(numb_bin) + " de " + str(n_bins))
pl.imshow(M[numb_bin,0,:,:])
pl.colorbar()
pl.show()


