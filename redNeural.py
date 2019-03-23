import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalizar(data):
    return (data - data.mean()) / data.std()

def sigmoid(m):
	return 1/(1+np.exp(-m))

def der_sigmoid(m):
    return m * (1 - m)

def rms(obt, esp):
	rms = 0.0
	for x in range(len(obt)):
		error = obt[x] - esp[x]
		rms += error * error 

class RedNeural():
	def __init__(self,data,target,neu_intermedia):
		self.data = data
		# print(self.data)
		self.target = target
		# print(self.target)

		self.num_feat = len(data.columns)
		self.num_inter = neu_intermedia
		self.num_out = len(target.columns)

		self.w_feats = np.random.randn(self.num_feat, self.num_inter)
		self.w_inter = np.random.randn(self.num_inter, self.num_out)
		#self.feats = np.random.uniform(-1,1,size=self.num_feat)

	def propagate(self):
		self.feats_w = np.dot(self.data,self.w_feats)
		self.activacion = sigmoid(self.feats_w)
		self.inter_w = np.dot(self.activacion, self.w_inter)
		resultado = sigmoid(self.inter_w)
		# print(resultado)
		return resultado

	def backpropagate(self,lr):
		resultado = self.propagate()
		L = (1/self.data.shape[0]) * np.sum(-self.target * np.log(resultado) - (1 - self.target) * np.log(1 - resultado))
		
		error_salida = self.target - resultado
		delta_salida = error_salida * der_sigmoid(resultado)

		error_w_inter = delta_salida.dot(self.w_inter.T)
		delta_inter = error_w_inter * der_sigmoid(self.activacion)

		self.w_feats += lr * self.data.T.dot(delta_inter)
		self.w_inter += lr * self.activacion.T.dot(delta_salida)
		# print("Perdida: %f" % (L))

	def entrenar(self,epocas,lr):
		print("Epocas %d | Tasa de aprendizaje %f" % (epocas,lr))
		errores = np.zeros(epocas)
		for e in range(epocas):
			# print("Epoca %d" % (e))
			self.backpropagate(lr)

	def accuracy(self):
		correcto = 0
		incorrecto = 0
		resultado = self.propagate()
		t_val = self.target.values
		for x in range(len(self.data)):
			max_r = np.argmax(resultado[x])
			# print(max_r)
			if abs(t_val[x,max_r] - 1.0) < 1.0e-5:
				correcto += 1
			else:
				incorrecto += 1
		return (correcto * 1.0) / (correcto + incorrecto)
			

			

data = pd.read_csv("irisTrainData.txt", sep=",", header=None)
# print(data)
target = pd.DataFrame(data.iloc[:,-3:])
data = data.drop(data.columns[-1], axis=1)
data = data.drop(data.columns[-1], axis=1)
data = data.drop(data.columns[-1], axis=1)

RN = RedNeural(data,target,2)
RN.entrenar(1000,0.05)
print("Accuracy: %f" % (RN.accuracy()))
