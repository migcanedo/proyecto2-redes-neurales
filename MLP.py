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

entradas, respuestasDeseadas = leerTXT("datos_P2_EM2019_N500.txt")
data_train, data_test, target_train, target_test = train_test_split(entradas, respuestasDeseadas, test_size = 0.2, shuffle=False)
print(data_test)

