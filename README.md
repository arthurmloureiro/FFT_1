FFT_1
=====
Esta pasta contém:

rand.py - 	
		programa feito para testar gerar um f(k) com amplitude que
	 	respeita uma distr. gaussiana e uma fase com distr. homog.
	  	plota as distr.

rand2.py - 	NÃO FUNCIONA
		tenta utilizar os dados gerados acima fazendo uma FFT com
		o pacote np.fft.fft do python mas não está dando 
		resultados agradáveis e não parece fazer muito sentido
		
fft_teste1.py -
		programa faz a fft de uma função composta por cossenos,
		plota a função, a parte real da FFT e o módulo quadrado
		dela. 
		CONTÉM: info sobre como fazer subplots.
		
grid1.py -	
		Cria um grid de vetores de onda (k) respeitando o método
		que a FFT gera os k's, pode-se plotar os k's para observar
		o comportamento. Coloca os k's em um grid 3D onde cada
		nó contém info sobre o módulo de k.
		CONTÉM: info sobre como gerar o grid fazendo uma matriz 3D
			que respeita uma certa regra de composição
			
grid3D.py - CLASSE
		Cria uma classe que é o grid de vetores de onda k.
		Tem como entrada os tamanhos do vetores k_x, k_y e k_z
		deve ser chamado como: ~x=grid3D.grid3d(n,m,l)~
		~x.size_x~ retorna o tamanho do vetor k_x
		~x.matrix~ é o grid como descrito em grid1.py
		CONTÉM: info sobre como criar classes (mas não entendi direito)
		
teste3D.py	
		plotar um exemplo em 3D ---- FALHOU
		
pk.py		
		Utiliza a classe 'grid3D.py' para gerar um contraste de densidade 
		delta(x) de um campo com espectro de potências dado por P(k) usando a
		função np.fft.ifft do python.
		Plota o mapa do contraste de densidades e a matriz do grid de k.
		CONTÉM: info sobre como usar uma classe externa, como plotar uma matriz 
		
	 
