import pandas as pd
import numpy as np

'''
	Funcion que dado un DataFrame, normaliza cada columna del mismo 
	con el metodo Z-score.
'''
def normalizar(data):
	return (data - data.mean()) / data.std()


'''
	Funcion que compara los resultados originales y los obtenidos para
	calcular lasmetricas de evaluacion como:
		- Accuracy
		- # de Falsos Positivos 
		- # de Falsos Negativos
	
'''
def metricas_eval(t_ori, t_obt, clases=["1", "-1"]):
	concat = pd.concat([t_ori.reset_index(drop=True), t_obt.reset_index(drop=True)], axis=1)
	concat.columns = ["ori", "obt"]
	acc = (concat.iloc[:,0] == concat.iloc[:,1]).astype(int).sum()/concat.shape[0]
	falsos_negativos = concat[(concat["ori"] == clases[0])&(concat["obt"] == clases[1])]["ori"].count()
	falsos_positivos = concat[(concat["ori"] == clases[1])&(concat["obt"] == clases[0])]["ori"].count()

	return acc, falsos_positivos, falsos_negativos

'''
	Clase que define el comportamiento de una Red Neuronal de Clasificacion con 
	una sola capa oculta y 	diferente cantidad de neuronas de entrada y salida.
'''
class RedNeural():
	def __init__(self, data, target, neu_intermedia, pesos=None):
		self.data = data.copy()
		self.data["bias"] = 1
		self.target = pd.get_dummies(target.copy().astype(str), prefix='', prefix_sep='')

		self.clases = self.target.columns


		self.num_feat = len(self.data.columns)
		self.num_inter = neu_intermedia 
		self.num_out = len(self.target.columns)

		if pesos:
			self.w_feats = pesos[0]
			self.w_inter = pesis[1]
		else:
			self.w_feats = np.random.randn(self.num_feat, self.num_inter)
			self.w_inter = np.random.randn(self.num_inter + 1, self.num_out)

	'''
		Funcion Sigmoidal
	'''
	def _sigmoid(self, m):
		return 1/(1+np.exp(-m))

	'''
		Derivada de la Funcion Sigmoidal
	'''
	def _der_sigmoid(self, m):
		return m * (1 - m)

	'''
	'''
	def propagate(self):
		self.activacion = self._sigmoid(np.dot(self.data, self.w_feats))
		sesgo = np.ones( (self.activacion.shape[0], 1) )
		self.activacion = np.concatenate((self.activacion, sesgo), axis=1)

		resultado = self._sigmoid(np.dot(self.activacion, self.w_inter))
		return resultado

	'''
	'''
	def backpropagate(self, lr, resultado):		
		error_salida = self.target - resultado
		delta_salida = error_salida * self._der_sigmoid(resultado)

		error_w_inter = delta_salida.dot(self.w_inter.T)
		delta_inter = error_w_inter * self._der_sigmoid(self.activacion)

		self.w_feats += lr * self.data.T.dot(delta_inter.iloc[:, :-1])
		self.w_inter += lr * self.activacion.T.dot(delta_salida)

	'''
	'''
	def train(self, epocas, lr):
		print("Epocas %d | Tasa de aprendizaje %s" % (epocas,lr))
		errores = np.zeros(epocas)
		for e in range(epocas):
			# if e % 1000 == 0: print("Epoca:", e)
			o = self.propagate()
			self.backpropagate(lr, o)

			acc, _, _ = metricas_eval(pd.DataFrame(self.target).idxmax(axis=1), self.forward(self.data))
			errores[e] = 1 - acc

		return errores

	'''
	'''
	def forward(self, X):
		X = X.copy()
		X["bias"] = 1

		activacion = self._sigmoid(np.dot(X, self.w_feats))
		sesgo = np.full( (activacion.shape[0], 1), 1 )
		activacion = np.concatenate((activacion, sesgo), axis=1)

		resultado = self._sigmoid(np.dot(activacion, self.w_inter))

		resultado = pd.DataFrame(resultado, columns=self.clases)
		resultado = resultado.idxmax(axis=1)

		return resultado


	def save_w():
		np.savetxt("w1.txt", self.w_feats, fmt="%s")
		np.savetxt("w2.txt", self.w_inter, fmt="%s")