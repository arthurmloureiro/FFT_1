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


lado = 10.0                                                             # em Mpc
########################DADOS INICIAIS###################################
N_x = 81 ; N_y = 81 ; N_z = 81						#Numero de celulas em x,y,z
d_x = lado ; d_y = lado ; d_z = lado				        #volume da celula em x,y,z em Mpc
l_x = d_x*N_x ; l_y = d_y*N_y ; l_z = d_z*N_z			        #Tamanho da caixa em Mpc em x,y,z
volume = l_x*l_y*l_z							
dV = (volume/(N_x*N_y*N_z))
n_bar = 2.0 								# densidade de galaxias na celula
N_bar = n_bar*volume				
bias = 1.0	
#########################################################################
realiz = 500							#numero de realizacoes
k_r , P_k = np.loadtxt('fid_matterpower.dat', unpack=True)	        #pega o P(k) do CAMB
k_r = np.insert(k_r,0,0.)						#insere P(k=0) = 0 antes de interpolar
P_k = np.insert(P_k,0,0.)						#mesma coisa que acima

k = gr.grid3d(N_x,N_y,N_z,l_x,l_y,l_z)					#cria o grid com qualquer tamanho em 3D
Pk = interpolate.InterpolatedUnivariateSpline(k_r,P_k)		        #interpola os dados do CAMB
"""
                                achando a funcao de correlacao
"""
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

P_k_gauss[0] = 0.0

Pkg = interpolate.InterpolatedUnivariateSpline(k_r,P_k_gauss)		        #interpola o espectro gaussiano
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
	
def delta_x_ln(d_,sigma_):
	return np.exp(bias*d_ - ((bias**2.)*(sigma_))/2.0) -1.
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
"""
                                    FFT
"""
PN_gauss = np.loadtxt("gauss500.dat")
PN_poiss = np.loadtxt("poisson500.dat")
P_bla_gauss = np.zeros(49)
P_bla_poiss = np.zeros(49)
err_gauss = np.zeros(49)
err_poiss = np.zeros(49)
bias_esti_gauss1 = np.zeros(realiz)
bias_esti_poiss1 = np.zeros(realiz)

for i in range(realiz): 
	bias_esti_gauss1[i] = np.mean((PN_gauss[:,i] - (dV/n_bar + 50))/Pk(k_bar[1:]))
	bias_esti_poiss1[i] = np.mean((PN_poiss[:,i] - (dV/n_bar + 50))/Pk(k_bar[1:]))
b_gauss = np.mean(bias_esti_gauss1)
b_poiss = np.mean(bias_esti_poiss1)
pl.figure()
#pl.yscale("log")
for i in range(realiz):
	pl.plot(k_bar[1:], ((PN_gauss[:,i]-(dV/n_bar + 50) )/b_gauss)/Pk(k_bar[1:]), "*")
#pl.plot(k_r,P_k)
pl.figure()
#pl.yscale("log")
for i in range(realiz):
	pl.plot(k_bar[1:], ((PN_poiss[:,i]-(dV/n_bar + 50) )/b_poiss)/Pk(k_bar[1:]), "o")
#pl.plot(k_r,P_k)
for i in range(49):
	P_bla_gauss[i] = (np.mean((PN_gauss[i,:] - dV/n_bar)/b_gauss) - Pk(k_bar[1:])[i])/Pk(k_bar[1:])[i]
	P_bla_poiss[i] = (np.mean((PN_poiss[i,:] - dV/n_bar)/b_poiss) - Pk(k_bar[1:])[i])/Pk(k_bar[1:])[i]
	err_gauss[i] = np.std(PN_gauss[i,:])/np.mean(PN_gauss[i,:])
	err_poiss[i] = np.std(PN_poiss[i,:])/np.mean(PN_poiss[i,:])

pl.figure()
pl.title("Erros")
pl.ylabel("$((<P_g(k)-SN)/b^2> - P_{camb}(k)/P_{camb}(k)$")
pl.xlabel("$k$")
pl.errorbar(k_bar[1:],(P_bla_gauss-P_bla_gauss), yerr=err_gauss, label="Gauss + Poiss")
pl.errorbar(k_bar[1:],(P_bla_poiss-P_bla_poiss), yerr=err_poiss, label="Poiss")
legend = pl.legend(loc=0, shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
pl.show()
sys.exit(-1)			
PN = np.zeros((len(k_bar[1:]), realiz))	
for i in range(realiz):
	delta_x_gaus = ((delta_k_g(p_matrix).size)/volume)*np.fft.ifftn(delta_k_g(p_matrix))
	var_gr = np.var(delta_x_gaus.real)
	var_gi = np.var(delta_x_gaus.imag)
	delta_xr_g = delta_x_gaus.real
	delta_xi_g = delta_x_gaus.imag
	delta_xr = delta_x_ln(delta_xr_g, var_gr)
	delta_xi = delta_x_ln(delta_xi_g, var_gi)
	d_k = (volume/(N_x*N_y*N_z))*np.fft.fftn(delta_xr)			       	   # ifft de d_x.real
	P_a2 = np.einsum("aijl,ijl,ijl->a", M, d_k, np.conj(d_k))/(np.einsum("aijl->a", M)*volume)
	N_r = np.random.poisson(n_bar*(1.+delta_xr))
	delta_gg_r = (N_r - np.mean(N_r))/np.mean(N_r)
	delta_gg_k = (volume/(N_x*N_y*N_y))*np.fft.fftn(delta_gg_r)
	P_gg = np.einsum("aijl,ijl,ijl->a", M, delta_gg_k, np.conj(delta_gg_k))/(np.einsum("aijl->a", M)*volume)
	PN[:,i] = P_gg.real 					#linhas = dif k's ; colunas = dif mapas


sys.exit(-1)

"""
				                PLOTS
"""
k.plot	
pl.colorbar()								            #plota a matriz dos k's

pl.figure("P(k)")							                            #plotando o espectro original
pl.grid(1)
#pl.loglog()
pl.yscale("log")
pl.xlabel("k")
pl.ylabel('P(k)')
pl.plot(k_r[1:], P_k[1:], label="CAMB")						            #DADOS
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
