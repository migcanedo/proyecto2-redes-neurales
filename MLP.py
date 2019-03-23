import csv
from random import uniform, randint
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
        respuestasDeseadas.append(float(elemento[-1]))
        elemento.pop(-1)
        x.append(elemento)
        # Agregamos el sesgo
        elemento.insert(0,1)

    return x, respuestasDeseadas


entradas, salidas = leerTXT("datos_P2_EM2019_N500.txt")
for i in  range(len(entradas)):
    print("Entrada: " + str(entradas[i]) + " Salidas: " + str(salidas[i]))
