import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from redNeural import *

# Se lee el archivo con la entrada.
print("Leyendo Dataset...")
data = pd.read_csv("iris.data", sep=",", header=None)

# Separamos el target y lo transformamos a String.
target = pd.DataFrame(data.iloc[:,-1]).astype(str)
data = data.drop(data.columns[-1], axis=1)

# Normalizamos los datos.
data = normalizar(data)

# Para clasificador Binario
# target.replace("Iris-versicolor", "No-Iris-setosa", inplace=True)
# target.replace("Iris-virginica", "No-Iris-setosa", inplace=True)


# Todas las pruebas en una sola corrida.
ns = range(4, 11)
size_test = [0.5, 0.4, 0.3, 0.2, 0.1]

tipo = "Binario"
# tipo = "Ternario"
for n in ns:
	for st in size_test:
		print("###############################################")
		print("Numero de Neuronas:", n)
		print("Porcentaje de entrenamiento:", 1-st)
				
		# Separamos el dataset en Train y Test
		data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = st, shuffle=True)

		# Definimos la Red Neuronal con n cantidad de neuronas en la capa oculta
		RN = RedNeural(data_train, target_train, n)

		# Entrenamos la Red
		epocas = 60
		tasa = 0.02
		errores = RN.train(epocas, tasa)

		# Grafica la convergencia del Error
		fig, ax = plt.subplots()
		ax.plot(errores)
		ax.set_ylim((0, 1))
		plt.xlabel('Epocas')
		plt.ylabel('Proporción de elementos mal clasificados')
		plt.title("Historial de Errores para %s Neuronas y %s de Entrenamiento"%(n, st))
		plt.savefig("3_%s_Convergencia_n%s_ts%s.png"%(tipo, n, int((1-st)*100)))

		# Imprime las Metricas de Evaluacion de Train y Test

		acc, falsos_positivos, falsos_negativos = metricas_eval(target_train, RN.forward(data_train), ["Iris-setosa", "No-Iris-setosa"])
		print("TRAIN")
		print("--------------------------------------")
		print("Accuracy: ", acc)
		print("Falsos Positivos:", falsos_positivos)
		print("Falsos Negativos:", falsos_negativos)
		
		
		acc, falsos_positivos, falsos_negativos = metricas_eval(target_test, RN.forward(data_test), ["Iris-setosa", "No-Iris-setosa"])
		print("TEST")
		print("--------------------------------------")
		print("Accuracy: ", acc)
		print("Falsos Positivos:", falsos_positivos)
		print("Falsos Negativos:", falsos_negativos)
			
		


		