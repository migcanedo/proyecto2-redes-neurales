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
        datosArreglo = []
        for fila in datos:
            datosArreglo.append(fila)
    respuestasDeseadas = []
    x = []
    for elemento in datosArreglo:
        respuestasDeseada = float(elemento[-1])
        if respuestasDeseada == 1:
            respuestasDeseadas.append(1)
        else:
            respuestasDeseadas.append(0)
        elemento.pop(-1)
        x.append(elemento)
        # Agregamos el sesgo
        elemento.insert(0,1)
    return x, respuestasDeseadas

def procesarMinMax(datos):
    nuevosDatos = []
    columnaProcesadas = []
    for columna in range(1,len(datos[0])):
        valores = [float(y[columna]) for y in datos]
        minimo = min(valores)
        maximo = max(valores)
        nuevosValores = [(x-minimo)/(maximo-minimo) for x in valores]
        columnaProcesadas.append(nuevosValores)
    for x in range(len(datos)):
        nuevoDato = [1]
        for y in columnaProcesadas:
            nuevoDato.append(y[x])
        nuevosDatos.append(nuevoDato)
    return nuevosDatos

def procesarMedVar(datos):
    nuevosDatos = []
    columnaProcesadas = []
    for columna in range(1,len(datos[0])):
        valores = [float(y[columna]) for y in datos]
        media = np.mean(valores)
        varianza = np.std(valores)
        nuevosValores = [(x-varianza)/media for x in valores]
        columnaProcesadas.append(nuevosValores)
    for x in range(len(datos)):
        nuevoDato = [1]
        for y in columnaProcesadas:
            nuevoDato.append(y[x])
        nuevosDatos.append(nuevoDato)
    return nuevosDatos    

class Perceptron:
    def __init__(self, tasaAprendizaje, cantidadParametros, funcionActivacion):
        self.tasaAprendizaje=tasaAprendizaje
        self.cantidadParametros=cantidadParametros
        self.pesos=[]
        self.funcionActivacion=funcionActivacion
        for i in range(0,cantidadParametros+1):
            # Iniciamos los pesos con un numero aleatorio entre -0.5 y 0.5
            self.pesos.append(uniform(-0.5,0.5))
        self.delta = 0
        self.salida = 0
        self.error = 0

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

    def funcionActivacionSigmoidal(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        salida = self.funcionSigmoidal(salida)
        return salida

    def activarNeurona(self, elemento):
        if self.funcionActivacion == "Logistica":
            return self.funcionActivacionLogistica(elemento)
        elif self.funcionActivacion == "Lineal":
            return self.funcionActivacionLineal(elemento)
        elif self.funcionActivacion == "Sigmoidal":
            return self.funcionActivacionSigmoidal(elemento)
        elif self.funcionActivacion == "Signo":
            return self.clasificarFuncionSigno(elemento)        

    def derivadaFuncionLogistica(self, elemento):
        return self.funcionLogistica(elemento) * (1 - self.funcionLogistica(elemento))

    def funcionSigmoidal(self, elemento):
        return(1/(1+exp(-elemento)))

    def derivadaFuncionSigmoidal(self, elemento):
        return(self.funcionSigmoidal(elemento)*(1-self.funcionSigmoidal(elemento)))

    def clasificarFuncionSigno(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        if salida < 0:
            return -1
        else:
            return 1

# Leemos los datos y separamos las entradas para entrenamiento y prueba
entradas, respuestasDeseadas = leerTXT("datos_P2_EM2019_N500.txt")
# entradas = procesarMedVar(entradas)
# entradas = procesarMinMax(entradas)
data_train, data_test, target_train, target_test = train_test_split(entradas, respuestasDeseadas, test_size = 0.2, shuffle=False)

erroresEntrenamiento = []
erroresPrueba = []
# Cantidad de Neuronas en la capa oculta
n = 8
tasasAprendizaje = [0.0005]
mejorError = [1000000,10000000,1000000]     # [Error, epoca, tasa de apredizaje]
for tasaAprendizaje in tasasAprendizaje:
    # Arreglo de neuronas para la capa oculta
    capaOculta = []
    for i in range(0,n):
        capaOculta.append(Perceptron(tasaAprendizaje, 1,"Sigmoidal"))
    # Creamos una capa de salida con funcion de activacion Sigmoidal
    capaSalida = [Perceptron(tasaAprendizaje, n, "Sigmoidal"),Perceptron(tasaAprendizaje, n, "Sigmoidal")]
    neuronaSalida = Perceptron(tasaAprendizaje, n, "Sigmoidal")
    epoca = 0
    # Para graficar
    epocas = []
    erroresTrain = []
    erroresTest = []

    # Condicion parada backpropagation
    while(epoca <= 2000):
        if epoca % 350 == 0:
            print("Epoca: " + str(epoca))
        epoca += 1
        epocas.append(epoca)
        errorAcumulado = 0

        # Procesamos las entradas
        for i in range(0,len(data_train)): 
            # FEED FORWARD ########################
            # Pasamos la entrada a cada neurona de la capa oculta
            salidasCapaOculta = []
            for neurona in capaOculta:
                salida = neurona.activarNeurona(data_train[i])
                salidasCapaOculta.append(salida)
                neurona.salida = salida
            # Agregamos el sesgo de entrada a la capa de salida
            salidasCapaOculta.insert(0,1)
            # Pasamos las salidas de la capa oculta a la capa de salida
            for neurona in capaSalida:
                # Calculamos salida
                salida = neurona.activarNeurona(salidasCapaOculta)
                neurona.salida = salida
                # Calculamos error
                neurona.error = (target_train[i]-neurona.salida)
                # Calculamos delta
                neurona.delta = neurona.salida*(1-neurona.salida)*neurona.error

            # Pasamos a la capa oculta 
            for j in range(len(capaOculta)):
                # Calculamos el error
                error = 0
                for neuronaSalida in capaSalida:
                    error += neuronaSalida.pesos[j] * neuronaSalida.delta
                capaOculta[j].error = error
                # Calculamos el delta
                capaOculta[j].delta = capaOculta[j].error *  capaOculta[j].salida * (1- capaOculta[j].salida)

            # Actualizamos los pesos de la capa oculta
            for neurona in capaOculta:
                for j in range(0, len(neurona.pesos)):
                    neurona.pesos[j] += neurona.tasaAprendizaje * neurona.delta * float(data_train[i][j])
            # Actualizamos los pesos de la capa de salida
            for neurona in capaSalida:
                for j in range(0, len(neurona.pesos)):
                    neurona.pesos[j] += neurona.tasaAprendizaje * neurona.delta * float(salidasCapaOculta[j])
        
        # Verificacion con conjunto de prueba #
        # Procesamos las entradas
        errorAcumulado = 0
        for i in range(0,len(data_test)): 
            # Pasamos la entrada a cada neurona de la capa oculta
            salidasCapaOculta = []
            for neurona in capaOculta:
                salidasCapaOculta.append(neurona.activarNeurona(data_test[i]))
            # Agregamos el sesgo de entrada a la capa de salida
            salidasCapaOculta.insert(0,1)
            # Pasamos las salidas de la capa oculta a la neurona de salida
            salidaFinal = []
            for neurona in capaSalida:
                salidaFinal.append(neurona.activarNeurona(salidasCapaOculta))

        # Verificacion con conjunto de entrenamiento #
        # Procesamos las entradas
        errorAcumulado = 0
        for i in range(0,len(data_train)): 
            # Pasamos la entrada a cada neurona de la capa oculta
            salidasCapaOculta = []
            for neurona in capaOculta:
                salidasCapaOculta.append(neurona.activarNeurona(data_train[i]))
            # Agregamos el sesgo de entrada a la capa de salida
            salidasCapaOculta.insert(0,1)
            # Pasamos las salidas de la capa oculta a la neurona de salida
            salidaFinal = []
            for neurona in capaSalida:
                salidaFinal.append(neurona.activarNeurona(salidasCapaOculta))
            if int(target_train[i]) != int(salidaFinal.index(max(salidaFinal))):
                errorAcumulado += 1
        
        erroresTrain.append(errorAcumulado/len(data_train))

        # Verificacion con conjunto de prueba #
        # Procesamos las entradas
        errorAcumulado = 0
        for i in range(0,len(data_test)): 
            # Pasamos la entrada a cada neurona de la capa oculta
            salidasCapaOculta = []
            for neurona in capaOculta:
                salidasCapaOculta.append(neurona.activarNeurona(data_test[i]))
            # Agregamos el sesgo de entrada a la capa de salida
            salidasCapaOculta.insert(0,1)
            # Pasamos las salidas de la capa oculta a la neurona de salida
            salidaFinal = []
            for neurona in capaSalida:
                salidaFinal.append(neurona.activarNeurona(salidasCapaOculta))
            if int(target_test[i]) != int(salidaFinal.index(max(salidaFinal))):
                errorAcumulado += 1
        
        errorRelativo = errorAcumulado/len(data_test)        
        erroresTest.append(errorRelativo)

        if mejorError[0] > errorRelativo:
            mejorError = [errorRelativo, epoca, tasaAprendizaje]

    print("Mejor error obtenido en prueba: " + str(mejorError[0]) + ", en la epoca " + str(mejorError[1]) + ", con tasa de aprendizaje " + str(mejorError[2]))

    # Grafica 
    plt.plot(epocas,erroresTrain, label="Entrenamiento")
    plt.plot(epocas,erroresTest, label="Prueba")
    plt.xlabel('Epocas')
    plt.ylabel('Proporcion de elementos mal clasificados')
    plt.title("Historial de Errores con tasa de aprendizaje " + str(tasaAprendizaje))
    plt.legend()
    plt.show()

