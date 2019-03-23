import csv
from random import uniform, randint
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split

def leerTXT(archivoDatos):
    # Leer datos
    with open(archivoDatos, newline='') as csvfile:
        datos = csv.reader(csvfile, delimiter=' ')
        print
        datosArreglo = []
        for fila in datos:
            datosArreglo.append(fila)
    respuestasDeseadas = []
    x = []
    for elemento in datosArreglo:
        respuestasDeseadas.append(float(elemento[-1]))
        elemento.pop(-1)
        x.append(elemento)
        # Agregamos el sesgo
        elemento.insert(0,1)
    return x, respuestasDeseadas

class Perceptron:
    def __init__(self, tasaAprendizaje, cantidadParametros, funcionActivacion):
        self.tasaAprendizaje=tasaAprendizaje
        self.cantidadParametros=cantidadParametros
        self.pesos=[]
        self.funcionActivacion=funcionActivacion
        for i in range(0,cantidadParametros+1):
            self.pesos.append(0)
            #self.pesos.append(uniform(0,1))

    def funcionLogistica(self, elemento, alpha=1):
        return 1 / (1 + exp(-alpha*elemento))

    def funcionActivacionLogistica(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        salida = self.funcionLogistica(salida)
        return salida            

    def funcionActivacionLineal(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        return salida

    def activarNeurona(self, elemento):
        if self.funcionActivacion == "Logistica":
            return self.funcionActivacionLogistica(elemento)
        elif self.funcionActivacion == "Lineal":
            return self.funcionActivacionLineal(elemento)

    def derivadaFuncionLogistica(self, elemento):
        return self.funcionLogistica(elemento) * (1 - self.funcionLogistica(elemento))

# Leemos los datos y separamos las entradas
entradas, respuestasDeseadas = leerTXT("datos_P2_EM2019_N500.txt")
data_train, data_test, target_train, target_test = train_test_split(entradas, respuestasDeseadas, test_size = 0.2, shuffle=False)
print(data_test)

erroresEntrenamiento = []
erroresPrueba = []
# Cantidad de Neuronas en la capa oculta
cantNeuronas = [8]
for n in cantNeuronas:
    # Arreglo de neuronas para la capa oculta
    capaOculta = []
    for i in range(0,n):
        capaOculta.append(Perceptron(0.1, 1,"Logistica"))
    # Creamos neurona de salida con funcion de activacion lineal
    neuronaSalida = Perceptron(0.1, n, "Lineal")
    epoca = 0
    # Para graficar
    epocas = []
    errores = []

    # Condicion parada backpropagation
    while(epoca <= 700):
        if epoca % 350 == 0:
            print("Epoca: " + str(epoca))
        epoca += 1
        epocas.append(epoca)
        errorCuadraticoMedio = 0

        # Procesamos las entradas
        for i in range(0,len(data_train)): 
            # Pasamos la entrada a cada neurona de la capa oculta
            salidasCapaOculta = []
            for neurona in capaOculta:
                salidasCapaOculta.append(neurona.activarNeurona(data_train[i]))
            # Agregamos el sesgo de entrada a la capa de salida
            salidasCapaOculta.insert(0,1)
            # Pasamos las salidas de la capa oculta a la neurona de salida
            salidaFinal = neuronaSalida.activarNeurona(salidasCapaOculta)
            error = float(respuestasDeseadas[i])-float(salidaFinal)
            errorCuadraticoMedio = float(errorCuadraticoMedio) + (float(error)**2)
            gradienteSalida = error # por la derivada de la funcion linea: 1 
            # Actualizo los pesos de la neurona salida
            pesosViejos = []
            for k in range(0, len(neuronaSalida.pesos)):
                deltaW = (float(neuronaSalida.tasaAprendizaje) * gradienteSalida * float(salidasCapaOculta[k]))
                pesosViejos.append(neuronaSalida.pesos[k])
                neuronaSalida.pesos[k] = float(neuronaSalida.pesos[k]) + deltaW

                                        
            # Actualizo los pesos de las neuronas de la capa oculta
            for z in range(0, len(capaOculta)):
                # Calculamos gradienteLocal
                sumatoriaGradientesPosteriores = float(gradienteSalida) * float(pesosViejos[z])
                # Calculamos el estimulo recibido por la neurona
                estimulo = capaOculta[z].pesos[0]
                for d in range(1,capaOculta[z].cantidadParametros+1):
                    estimulo = estimulo + (float(data_train[i][d]) * float(capaOculta[z].pesos[d]))
                gradienteLocal = capaOculta[z].derivadaFuncionLogistica(estimulo) # Derivada Funcion Activacion evaluada en el estimulo
                gradienteLocal = gradienteLocal * sumatoriaGradientesPosteriores
                # Calculo deltaW y actualizo cada peso
                for k in range(0, len(capaOculta[z].pesos)):
                    deltaW = (float(capaOculta[z].tasaAprendizaje) * float(gradienteLocal) * float(data_train[i][k]))
                    capaOculta[z].pesos[k] = float(capaOculta[z].pesos[k]) + deltaW


        # Finalizar el calculo del error cuadratico medio
        errorCuadraticoMedio = errorCuadraticoMedio/len(data_train)
        errores.append(errorCuadraticoMedio)
    print("Error cuadratico medio del conjunto de entrenamiento con " + str(n) + " Neuronas: " + str(errorCuadraticoMedio) + "")
    erroresEntrenamiento.append(errorCuadraticoMedio)

    # Grafica 
    plt.plot(epocas,errores)
    plt.xlabel('Epocas')
    plt.ylabel('Error Cuadratico Medio')
    plt.title("Historial de Errores")
    plt.show()
