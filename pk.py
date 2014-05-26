#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	
	v0.8 - Plota tanto os k's quanto a matriz dos delta_x
	v0.9 - Usa valores do CAMB em P(k) e interpola eles 
	v1.0 - é feita uma DFT em N dimensões, problemas quanto a normalização e unidades físicas
	v1.5 - A normalização está quase correta
	v2.0 - Faltava normalizar por sqrt(2) ou trocar 2*P(k) por P(k) na dist. Gaussiana
	v2.5 - Organiza os d_k em bins e estima o P(k)
   	v2.7 - Tenta fazer um campo lognormal 
   	v2.9 - Tudo funcionando perfeitamente pronto para implementar distr de galaxias
	Arthur E. da Mota Loureiro
		12/12/2013
"""
import numpy as np
import sys
import pylab as pl
import grid3D as gr
import scipy.integrate as inte
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy import interpolate


lado = 10.0                                                             # em Mpc
########################DADOS INICIAIS###################################
N_x = 81 ; N_y = 81 ; N_z = 81						#Numero de celulas em x,y,z
d_x = lado ; d_y = lado ; d_z = lado				        #volume da celula em x,y,z em Mpc
l_x = d_x*N_x ; l_y = d_y*N_y ; l_z = d_z*N_z			        #Tamanho da caixa em Mpc em x,y,z
#########################################################################
k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)	        #pega o P(k) do CAMB
k_r = np.insert(k_r,0,0.)						#insere P(k=0) = 0 antes de interpolar
P_k = np.insert(P_k,0,0.)						#mesma coisa que acima

k = gr.grid3d(N_x,N_y,N_z,l_x,l_y,l_z)					#cria o grid com qualquer tamanho em 3D
volume = l_x*l_y*l_z							
Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)		        #interpola os dados do CAMB
"""
achando a funcao de correlacao
"""
r_k=1.0*np.linspace(0.5,200.5,201)
dk_r=np.diff(k_r)
dk_r=np.append(dk_r,[0.0])
#print dk_r[-5:-1]
#print k_r[-5:-1]
krk=np.einsum('i,j',k_r,r_k)
sinkr=np.sin(krk)
dkkPk=dk_r*k_r*P_k*np.exp(-1.0*np.power(k_r/0.8,6.0))
rm1=np.power(r_k,-1.0)
termo2=np.einsum('i,j',dkkPk,rm1)

integrando=sinkr*termo2

corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*np.sum(integrando,axis=0)

"""
pl.figure()
pl.grid()
pl.plot(r_k, r_k*r_k*corr_ln, linewidth=1.5)
pl.show()
sys.exit(-1)
"""
corr_g = np.log(1.+corr_ln)
"""
Achando o espectro gaussiano
"""
dr = np.diff(r_k)
dr = np.append(dr,[0.0])
rkr = np.einsum('i,j', r_k,k_r)
sinrk2 = np.sin(rkr)
drCorr = dr*r_k*corr_g
km1 = np.power(k_r,-1.)
terms = np.einsum('i,j', drCorr,km1)
integrando2 = sinrk2*terms

P_k_gauss = 4.0*np.pi*np.sum(integrando2, axis=0)
#P_k_gauss = np.insert(P_k_gauss,0,0.)
#pl.figure()
#pl.loglog()
#pl.plot(k_r,P_k, label = "original")
#pl.plot(k_r, P_k_gauss, label = "gaussiano")
#pl.show()
#sys.exit(-1)
P_k_gauss[0] = 0.0

Pkg = interpolate.InterpolatedUnivariateSpline(k_r,P_k_gauss)		        #interpola os dados do CAMB


p_matrix =np.asarray([[[ Pkg(k.matrix[i][j][n]) for i in range(len(k.k_x))] for j in range(len(k.k_y))] for n in range(len(k.k_z))])

def A_k(P_):
	return np.random.normal(0.0,np.sqrt(2.*P_*volume))		#dist gaussiana media no zero E DESVIO SQRT(P_k)
#	return np.random.normal(0.0,np.sqrt(1.*P_*volume))		#dist gaussiana media no zero E DESVIO SQRT(P_k)
								        # Tiramos o fator de 2, deu certo! 
								        #incluído volume para fechar as unidades 
def phi_k(P_): 
	return (np.random.random(len(P_)))*2.*np.pi		        #segundo Padmanabhan pg 191 ===> É A MESMA COISA

def delta_k_g(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))			        #contraste de densidade em k
	
def delta_x_ln(d_,sigma_):
	return np.exp(d_ - (sigma_)/2.0) -1.
"""
			ORGANIZANDO EM BINS
"""
def heav(x):							#funcao heaviside
	if x==0:
		return 0.5
	return 0 if x<0 else 1
heav_vec = np.vectorize(heav)					         #heaviside vetorizada
n_bins = 50
k_bar = np.arange(0,n_bins,1)*(np.max(k.matrix)/n_bins)

M = np.asarray([heav_vec(k_bar[a+1]-k.matrix[:,:,:])*heav_vec(k.matrix[:,:,:]-k_bar[a])for a in range(len(k_bar)-1)])

############################# FFT #######################################
delta_x_gaus = ((delta_k_g(p_matrix).size)/volume)*np.fft.ifftn(delta_k_g(p_matrix))
var_gr = np.var(delta_x_gaus.real)
var_gi = np.var(delta_x_gaus.imag)
#print "Campo complexo: " + str(delta_x[1,2,3])  
delta_xr_g = delta_x_gaus.real
delta_xi_g = delta_x_gaus.imag
delta_xr = delta_x_ln(delta_xr_g, var_gr)
delta_xi = delta_x_ln(delta_xi_g, var_gi)

print np.mean(np.ravel(delta_xr))
print np.mean(np.ravel(delta_xi))

print np.var(np.ravel(delta_xr))
print np.var(np.ravel(delta_xi))

print var_gr
print var_gi

#sys.exit(-1)

#print "Parte real: " + str(delta_xr[1,2,3])  + " e parte imaginaria: " + str(delta_xi[1,2,3])
#########################################################################

print "################################################################"
print "                           DADOS                                "
print "################################################################"
print "Lx: " + str(l_x) + " Mpc // Ly: " + str(l_y) + " Mpc // Lz: " + str(l_z) + " Mpc"
print "Lado depois das contas: " + str((np.power(delta_xr.size,1./3))) 		#Não faz sentido se não é um cubo
print "Volume total: " + str(volume) + " Mpc^3"
print "Numero de celulas de delta_x: " + str(len(delta_xr))
print "Lado da celula: " + str(np.power(volume,1./3)/len(delta_xr)) + " Mpc"	#para /\x = 10. Mpc
print "<delta_r^2> REAL = " + str(np.std(delta_xr))				#deve ser aprox. 0.9
print "<delta_r^2> IMAG = " + str(np.std(delta_xi))				#deve ser aprox. 0.9
print "k_max da matriz de k: " + str(np.max(k.matrix))
#sys.exit(-1)
print "################################################################"

######################### Procurando P(k) ###############################
d_k = (volume/(N_x*N_y*N_z))*np.fft.fftn(delta_xr)			       	   # ifft de d_x.real
#P_a = np.einsum("aijl,ijl,ijl->a", M, d_k,np.conj(d_k))/volume
P_a2 = np.einsum("aijl,ijl,ijl->a", M, d_k, np.conj(d_k))/(np.einsum("aijl->a", M)*volume)
	#esta normalização parece ser a correta!
#########################################################################
"""
				PLOTS
"""
k.plot	
pl.colorbar()								            #plota a matriz dos k's
pl.figure("P(k)")							                            #plotando o espectro original
pl.grid(1)
#pl.loglog()
pl.yscale("log")
pl.xlabel("$k [hMpc^{-1}]$")
pl.ylabel('$P(k)$')
pl.plot(k_r[1:], P_k[1:], label="CAMB")		
#pl.plot(k_r[1:], Pkg(k_r[1:]), label="$P_G(k)$")				            #DADOS
#pl.plot(k_r[1:], Pk(k_r)[1:], label="Gaussiano interpolado")				  #INTERPOLADO
#pl.plot(P_a, label="1")
#sys.exit(-1)
#kkk = np.arange(0,len(P_a2),1)*(2*np.pi*np.power(l_x,-1.))

#kkk = np.arange(0,len(P_a2.real),1)*(np.max(k.matrix)/n_bins)
#Pa_interp = interpolate.InterpolatedUnivariateSpline(kkk,P_a2.real)	

pl.plot(k_bar[1:],P_a2, color="k", label=r"Estimated using $\delta_LN (\vec{k})$")				#ESTIMADO
legend = pl.legend(loc=0, shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
pl.axvline(x=np.max(k.matrix), linewidth=2., color='r')

pl.figure("Mapa")
pl.imshow(delta_xr[:,24,:], cmap=cm.jet)
pl.colorbar()
pl.grid(1)
pl.title(r'Slice of the $\delta_{LN}(\vec{x})$ map in unities of $[L_{cell}]$')
pl.show()
"""
pl.figure("Bins de K")
numb_bin =np.random.random_integers(0,n_bins-1)
pl.title("Mostrando bin numero " + str(numb_bin) + " de " + str(n_bins))
pl.imshow(M[numb_bin,0,:,:])
pl.colorbar()
pl.show()
"""
