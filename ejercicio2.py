import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from redNeural import *

# Se lee el archivo con la entrada
# data = pd.read_csv("datos_P2_EM2019_N500.txt", sep="\s", header=None)
data = pd.read_csv("datos_P2_EM2019_N1000.txt", sep="\s", header=None)
# data = pd.read_csv("datos_P2_EM2019_N2000.txt", sep="\s", header=None)

# Separamos el target y lo transformamos a String.
target = pd.DataFrame(data.iloc[:,-1]).astype(str)
data = data.drop(data.columns[-1], axis=1)

# Separamos el dataset en Train y Test
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.2, shuffle=False)

# Definimos la Red Neuronal con n cantidad de neuronas en la capa oculta
n = 8
RN = RedNeural(data_train, target_train, n)

# Entrenamos la Red
epocas = 100000
tasa = 0.0001
errores = RN.train(epocas, tasa)

# Grafica la convergencia del Error
fig, ax = plt.subplots()
ax.plot(errores)
ax.set_ylim((0, 1))
plt.xlabel('Epocas')
plt.ylabel('Proporción de elementos mal clasificados')
plt.title("Historial de Errores con tasa de aprendizaje " + str(tasa))
plt.show()

# Imprime las Metricas de Evaluacion de Train y Test
print("TRAIN")
print("--------------------------------------")
acc, falsos_positivos, falsos_negativos = metricas_eval(target_train, RN.forward(data_train))
print("Accuracy: ", acc)
print("Falsos Positivos:", falsos_positivos)
print("Falsos Negativos:", falsos_negativos)

print("TEST")
print("--------------------------------------")
acc, falsos_positivos, falsos_negativos = metricas_eval(target_test, RN.forward(data_test))
print("Accuracy: ", acc)
print("Falsos Positivos:", falsos_positivos)
print("Falsos Negativos:", falsos_negativos)

# Se construye el dataset de todos los puntos del plano 10x10
probando = pd.DataFrame(columns=["X", "Y"])
k = 0
for i in np.arange(0, 10.1, 0.1):
	for j in np.arange(0, 10.1, 0.1):
		probando.loc[k] = [i, j]
		k += 1

# Se clasifican dichos puntos
probando["T"] = RN.forward(probando)

# Se grafican los puntos del plano para apreciar la evaluacion
fig, ax = plt.subplots()
ax.add_artist(plt.Circle((5,5), 4, alpha=0.3))
ax.set_xlim((0, 10))
ax.set_ylim((0, 10))
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title("Clasificación de los puntos del plano 10x10")
probando[probando["T"] == "1"].plot.scatter(x=0, y=1, c='blue', ax=ax)
probando[probando["T"] == "-1"].plot.scatter(x=0, y=1, c='red', ax=ax)
plt.show()

