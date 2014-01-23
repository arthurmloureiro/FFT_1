"""
	Testando como funciona a FFT do Python
	v1.5 - tudo funciona, plota f(x), F(k) e |F(k)|^2
	v1.7 - Agora x eh uma variavel aleatoria que 
	Arthur E. M. Loureiro
	09/12/2013
"""
import numpy as np
import pylab as pl
N=100
tamanho = 40000
#x=np.linspace(0.0,np.pi*2, L)
#x=np.linspace(-10.,10., L)
x = 1./tamanho * np.arange(N)
froc = (tamanho/N)*np.arange(N)
#x=np.random.random(L)*2*np.pi
freq1 = 50.0
freq2 = 300.45
A_0 = 2.0
M= A_0*np.sin(2.*np.pi*freq1*x) + np.sin(2.*np.pi*freq2*x)		#cosseno
#M=A_0*np.exp(-(freq1*x**2)/2)			#gaussiana
#M = A_0*np.exp(1j*freq1*x)			#exponencial normal

four = np.fft.fft(M)
freq_ = np.fft.fftfreq(x.shape[-1])

fig = pl.figure()
fig.subplots_adjust(hspace=0.50)
ax1 = fig.add_subplot(311)					#faz subplots, 3-> graf na horiz, 1 -> na vertical, 1-> numero do grafico
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax2.plot(froc,four.real)
#fig.title('transformada de $10*cos(27*x)*cos(47*x)$')
ax1.set_title('$f(x) =' + str(A_0) + ' *cos(' + str(freq1)+ '*x) + cos('+str(freq2)+'*x)$')
ax3.set_title(r'$|\bar{F(k)}|^2$')
ax2.set_title(r'$Re(\bar{F(k)})$ ')
#ax2.set_xlim(0,L/2)
#ax3.set_xlim(0,L/2)
ax3.plot(np.abs(four)**2)
#ax3.plot(freq_,four.real, freq_, four.imag)
ax1.plot(M, '.')
pl.show()
