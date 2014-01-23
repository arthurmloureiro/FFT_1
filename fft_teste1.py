"""
	Testando como funciona a FFT do Python
	v1.5 - tudo funciona, plota f(x), F(k) e |F(k)|^2
	v1.7 - Agora x eh uma variavel aleatoria que 
	Arthur E. M. Loureiro
	09/12/2013
"""
import numpy as np
import pylab as pl

size = 4410
N = 1024

time = (1./size)*np.arange(-N/2,N/2)
#time = np.arange(-N/2,N/2)
freq = (size/N)*np.arange(-N/2,N/2)

a1 = 4.0
a2 = 6.0
f1 = 40.
f2 = 600.

#f = a1*np.sin(2.*np.pi*f1*time) + a2*np.sin(2.*np.pi*f2*time)
f = a1*np.exp(-(f2*2*time**2)/2)       
four = np.fft.fft(f)


fig = pl.figure()
fig.subplots_adjust(hspace=0.50)
ax1 = fig.add_subplot(311)					#faz subplots, 3-> graf na horiz, 1 -> na vertical, 1-> numero do grafico
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax2.plot(freq,np.abs(four)*(2./(N)))
#ax2.plot(freq, four)#*(1./(np.sqrt(len(four)))))
#fig.title('transformada de $10*cos(27*x)*cos(47*x)$')
#ax1.set_title('$f(x) =' + str(A_0) + ' *cos(' + str(freq1)+ '*x) + cos('+str(freq2)+'*x)$')
ax3.set_title(r'$|\bar{F(k)}|^2$')
ax2.set_title(r'$Re(\bar{F(k)})$ ')
#ax2.set_xlim(0,L/2)
#ax3.set_xlim(0,L/2)
ax3.plot(np.abs(four)**2)
#ax3.plot(freq_,four.real, freq_, four.imag)
ax1.plot(f, '.')
pl.show()
