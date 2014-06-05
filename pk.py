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
    	v3.0 - Calcula o espectro da distr. de galáxias e estima o bias e o shot-noise
    	v3.2 - Varia as realizacoes gaussiana e poissoniana
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
from scipy.optimize import curve_fit


lado = 15.625                                                             # em Mpc
########################DADOS INICIAIS###################################
N_x = 32 ; N_y = 32 ; N_z = 32						#Numero de celulas em x,y,z
d_x = lado ; d_y = lado ; d_z = lado				        #volume da celula em x,y,z em Mpc
l_x = d_x*N_x ; l_y = d_y*N_y ; l_z = d_z*N_z			        #Tamanho da caixa em Mpc em x,y,z
volume = l_x*l_y*l_z							#Total volume
dV = (volume/(N_x*N_y*N_z))
n_bar = 30.0 								# densidade de galaxias na celula
N_bar = n_bar*volume				
bias = 1.0	
#########################################################################
realiz = 500								#numero de realizacoes
k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)	        #pega o P(k) do CAMB
k_r = np.insert(k_r,0,0.)						#insere P(k=0) = 0 antes de interpolar
P_k = np.insert(P_k,0,0.)						#mesma coisa que acima
print("Generating the k-space Grid...\n")
k = gr.grid3d(N_x,N_y,N_z,l_x,l_y,l_z)					#cria o grid com qualquer tamanho em 3D
Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)		        #interpola os dados do CAMB
###################################
# finding the correlation function
###################################
print("Finding the Correlation Function...\n")
r_k=1.0*np.linspace(0.5,200.5,201)                          # r vai de 0.5 MPc ate 201 MPc
dk_r=np.diff(k_r)                                           #faz a diferenca entre k e k+dk
dk_r=np.append(dk_r,[0.0])

krk=np.einsum('i,j',k_r,r_k)
sinkr=np.sin(krk)
dkkPk=dk_r*k_r*P_k*np.exp(-1.0*np.power(k_r/0.8,6.0))
rm1=np.power(r_k,-1.0)
termo2=np.einsum('i,j',dkkPk,rm1)

integrando=sinkr*termo2

corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*np.sum(integrando,axis=0)        #tira o traco no eixo r realizando a integral

corr_g = np.log(1.+corr_ln)
######################################
# finding the gaussian power spectrum
######################################
print("Calculating the Gaussian P(k)...\n")
dr = np.diff(r_k)
dr = np.append(dr,[0.0])
rkr = np.einsum('i,j', r_k,k_r)
sinrk2 = np.sin(rkr)
drCorr = dr*r_k*corr_g
km1 = np.power(k_r,-1.)
terms = np.einsum('i,j', drCorr,km1)
integrando2 = sinrk2*terms

P_k_gauss = 4.0*np.pi*np.sum(integrando2, axis=0)

P_k_gauss[0] = 0.0

Pkg = interpolate.InterpolatedUnivariateSpline(k_r,P_k_gauss)		        #interpola o espectro gaussiano
###############################################################
# generating the P(K) grid using the gaussian interpolated Pkg
###############################################################
print("Calculating the P(k)-Grid...\n")
Pkg_vec = np.vectorize(Pkg)
p_matrix = Pkg_vec(k.matrix)

p_matrix[0][0][0] = 1.

def A_k(P_):
	return np.random.normal(0.0,np.sqrt(2.*P_*volume))		#dist gaussiana media no zero E DESVIO SQRT(P_k)

								        #incluído volume para fechar as unidades 
def phi_k(P_): 
	return (np.random.random(len(P_)))*2.*np.pi		        #segundo Padmanabhan pg 191 ===> É A MESMA COISA

def delta_k_g(P_):							
	return A_k(P_)*np.exp(1j*phi_k(P_))			        #contraste de densidade em k
###############################
# the log-normal density field
###############################
def delta_x_ln(d_,sigma_):
	return np.exp(bias*d_ - ((bias**2.)*(sigma_))/2.0) -1.
##########################
# organizing in bins of k
##########################
def heav(x):							#funcao heaviside
	if x==0:
		return 0.5
	return 0 if x<0 else 1
heav_vec = np.vectorize(heav)					         #heaviside vetorizada
n_bins = 25
k_bar = np.arange(0,n_bins,1)*(np.max(k.matrix)/n_bins)

M = np.asarray([heav_vec(k_bar[a+1]-k.matrix[:,:,:])*heav_vec(k.matrix[:,:,:]-k_bar[a])for a in range(len(k_bar)-1)]) 
#"""
"""
                                    FFT
"""
PN = np.zeros((len(k_bar[1:]), realiz))
print("Doing the heavy calculation...\n")	
file = open('supergrid.dat','w')
############################################################
# does the realizations varying both gaussian and poissonian
############################################################		
for m in range(realiz):
	#########################
	# gaussian density field
	#########################
	delta_x_gaus = ((delta_k_g(p_matrix).size)/volume)*np.fft.ifftn(delta_k_g(p_matrix))
	var_gr = np.var(delta_x_gaus.real)
	var_gi = np.var(delta_x_gaus.imag)
	delta_xr_g = delta_x_gaus.real
	delta_xi_g = delta_x_gaus.imag
	###########################
	# Log-Normal Density Field
	###########################
	delta_xr = delta_x_ln(delta_xr_g, var_gr)
	delta_xi = delta_x_ln(delta_xi_g, var_gi)
	d_k = (volume/(N_x*N_y*N_z))*np.fft.fftn(delta_xr)			       	   # ifft de d_x.real
	P_a2 = np.einsum("aijl,ijl,ijl->a", M, d_k, np.conj(d_k))/(np.einsum("aijl->a", M)*volume)
	#######################
	#poissonian realization
	#######################
	N_r = np.random.poisson(n_bar*(1.+delta_xr))
	#N_i = np.random.poisson(n_bar*(1.+delta_xi))
	###############################################################
	# this loop saves the galaxy map so Lucas' program can read it 
	###############################################################
	for i in range(N_x):
		for j in range(N_y):
			for l in range(N_z):
				file.write(",%d"%int(N_r[i,j,l]))
	##########################
	# galaxy density contrast
	##########################
	delta_gg_r = (N_r - np.mean(N_r))/np.mean(N_r)
	delta_gg_k = (volume/(N_x*N_y*N_y))*np.fft.fftn(delta_gg_r.real)
	P_gg = np.einsum("aijl,ijl,ijl->a", M, delta_gg_k, np.conj(delta_gg_k))/(np.einsum("aijl->a", M)*volume)
	######################################################################
	# saves all the realization's power spectra in the rows of this matrix
	######################################################################
	PN[:,m] = P_gg.real

file.close()
print("Done.")
print("Calculating the error bars and saving into the file: error_gauss.dat")
error_bar = np.zeros(len(k_bar[1:]))
for i in range(len(k_bar[1:])):
	error_bar[i] = np.std(PN[i,:])/np.mean(PN[i,:])
#############################
# saves the errors in a file
#############################
np.savetxt("error_gauss.dat", np.transpose((k_bar[1:],error_bar)))

print("So long and thanks for all the fish!")
sys.exit(-1)
############################################
# if you want plots, comment the line above
############################################
"""
				                PLOTS
"""
k.plot	
pl.colorbar()								            #plota a matriz dos k's

pl.figure("P(k)")							            #plotando o espectro original
pl.grid(1)
#pl.loglog()
pl.yscale("log")
pl.xlabel("k")
pl.ylabel('P(k)')
pl.plot(k_r[1:], P_k[1:], label="CAMB")						#DADOS
pl.plot(k_bar[1:],P_a2, color="k", label="Estimado")				#ESTIMADO
pl.plot(k_bar[1:],P_gg.real,"*", label="Espectro de galaxias")
pl.plot(k_bar[1:], (P_gg.real/(bias**2.) - dV/n_bar), "^", label=r"$P_g(k)/b^2 - dV/\bar{n}$")
legend = pl.legend(loc=0, shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
pl.axvline(x=np.max(k.matrix), linewidth=2., color='r')

pl.figure("Mapa")
pl.title("$\delta(x)_{i,0,k}$")
pl.imshow(delta_xr[:,24,:], cmap=cm.jet)
pl.colorbar()
pl.grid(1)
pl.title('Fatia do $\delta_x$ gerado apos a ifft de $\delta_k$ com $P(k)$')
pl.show()
"""
pl.figure("Bins de K")
numb_bin =np.random.random_integers(0,n_bins-1)
pl.title("Mostrando bin numero " + str(numb_bin) + " de " + str(n_bins))
pl.imshow(M[numb_bin,0,:,:])
pl.colorbar()
pl.show()
"""
