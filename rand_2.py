"""
	Testa as funcoes para a construcao do delta_k
	v:0.1		09/12/2013
	Arthur E. M. Loureiro
"""
import numpy as np
import pylab as pl
np.random.seed(1267)
#n_points = 10000
n_points = 1
k_max = 1000
def sigma(k_):
	return np.cos(20*k_) + 1

def A_k(k_):
	return np.random.normal(0,sigma(k_), n_points)			#distribuicao gaussiana media no zero
									#retorna um array de tamanho n_points
phi_k = (np.random.random(n_points))*np.pi - np.pi/2
#print float(A_k(2))
def delta_k(k_):
	return A_k(k_)*np.exp(-1j*phi_k)
print delta_k(2)

delta_k1 = np.zeros(k_max)
delta_k2 = np.zeros(k_max)

for k in range(k_max):
	delta_k1[k] = delta_k(k)[0]
	delta_k2[k] = delta_k(k)[0]
"""
d1 = np.abs(delta_k1)**2
pl.plot(d1, '.')
"""
fft = np.fft.fft(delta_k1)
#fft2 = np.fft.fft(fft)
b = np.abs(fft)**2
pl.figure()
#pl.plot(np.real(delta_k1),np.real(delta_k2), ".")
#pl.plot(delta_k1.real,'x')
#pl.plot(delta_k2, '*')
pl.plot(b.real)
pl.plot(fft.real)

pl.show()

"""
TA TUDO ERRADO SEI LA EU O QUE TA ACONTECENDO AQUI, NEM SEI O QUE EU QUERO NESSE PROGRAMA
"""
