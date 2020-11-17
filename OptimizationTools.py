import random
import numpy as np
import pandas as pd
import keras
from keras.callbacks import CSVLogger
import Models

def optimizeNN(param_dict,n_particles, iter,x_train,y_train,epochs):

    def trainProcess_min(loss_history):
        """

        :param loss_history: data frame de funciones de costo
        :return: mínimos
        """
        min_list = []
        for col in loss_history.columns:
            min = loss_history[col].min()
            min_list.append(min)
        return min_list

    def x1p_update(velocidad, c1, x1_pg, x1p, c2, x1_pL):
        """

        :param velocidad: velocidad de convergencia
        :param c1:
        :param x1_pg: mínimo global hasta el momento
        :param x1p:
        :param c2:
        :param x1_pL: mínimo local
        :return:
        """
        x1p_update = list()
        for i in range(0, len(x1p)):
            x1p_update.append(x1p[i] + (velocidad[i] + c1 * np.random.rand() * (x1_pg - x1p[i])
                                        + c2 * np.random.rand() * (x1_pL[i] - x1p[i])))
        return x1p_update

    def getRandomParams(start, stop, step, scale, n_samples):
        """

        :param start: valor mínimo que puede tomar el hiperparámetro dado
        :param stop: valor máximo que puede tomar el hiperparámetro dado
        :param step: tamaño de paso
        :param scale: número sobre el cual se dividirán las muestras aleatorias
        :param n_samples: número de muestras a generar
        :return: arreglo de muestras aleatorias sin reemplazo
        """
        number_list = list(i for i in range(start, stop + 1, step))
        sample = random.sample(number_list, n_samples)
        sample = list(float(i / scale) for i in sample)
        return sample

    def createParamRanges(optimizer_dict, n_particles):
        """

        :param optimizer_dict: diccionario de hiperparámetros a optimizar
        :param n_particles: númerode particulas a utilizar
        :return: arreglos aleatorios con muestro sin reemplazo con las especifiaciones dadas por el diccionario
        """
        range_dict = {}
        for i in optimizer_dict.keys():
            range_dict[i] = list(
                getRandomParams(start=optimizer_dict[i]["start"], stop=optimizer_dict[i]["stop"],
                                step=optimizer_dict[i]["step"], scale=optimizer_dict[i]["scale"],
                                n_samples=n_particles))

        return range_dict


    def paramspeed_update(x1p, velocidad, c1, local_x, c2, x1_pL):
        """

        :param x1p: particulas para cada parámetro
        :param velocidad: velocidad de convergencia
        :param c1:
        :param local_x: arreglo de mínimos locales encontrados hasta el momento
        :param c2:
        :param x1_pL:
        :return:
        """
        x1p_updated = {}
        for i in x1p.keys():
            list_position = 0
            if type(i) != int:
                x1p[i] = x1p[i]
                x1p_updated[i] = x1p_update(velocidad, c1, local_x[list_position], x1p[i], c2, x1_pL[i])
            else:
                continue
            list_position += 1
        return x1p_updated


    c1, c2 = 0.85, 0.85 #velocidad de las particulas

    x1p = createParamRanges(param_dict, n_particles)  # diccionario con los valores para cada parámetro
    x1pL = x1p #mínimo local = mínimo global en la primer iteración
    velocidad_x1 = np.zeros(n_particles)
    #mínimos globales
    x1_pg = 0
    x2_pg = 0
    x3_pg = 0
    x4_pg = 0
    x5_pg = 0
    fx_pg = 1
    fx_pL = np.ones(n_particles) * fx_pg
    history = pd.DataFrame()
    param_history = list()
    for i in range(0, iter):

        for j in range(0, n_particles):
            if x1p["learning_rate"][j]>0:

                model = Models.createNN(lr=float(x1p["learning_rate"][j]), neuron_pctg=float(x1p["neuron percentage"][j])
                                        , layer_pctg=float(x1p["layer percentage"][j]),
                                        dropout=float(x1p["dropout"][j])) #creación del modelo
            else : # el modelo no puede tener 0 exacto de tasa de aprendizaje
                model = Models.createNN(lr=float(x1p["learning_rate"][j])+.1,
                                        neuron_pctg=float(x1p["neuron percentage"][j])
                                        , layer_pctg=float(x1p["layer percentage"][j]))


            csv_logger = CSVLogger('log' + str(j) + '.csv', append=False,
                                   separator=';') # instancia para guardar historal de función de costo
            model.fit(x_train, y_train, epochs=epochs, batch_size=int(x1p["batch size"][j]), verbose=1,
                      callbacks=[csv_logger], shuffle=False) #entrenamiento del modelo
            to_read = 'log' + str(j) + '.csv' #nombre del archivo con función de costo de cada epoch en cada combinación
            fx = (pd.read_csv("C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/" + to_read,
                              sep=';', usecols=["loss"])) #almacenamiento de la función de costo
            fx = fx.rename(columns={'loss': 'loss' + str(j)})
            if j == 0:
                history = fx
            else:

                history = pd.concat([history, fx], axis=1, sort=False)
        fx = pd.DataFrame(trainProcess_min(history)) # mínimo encotrada para esa iteración
        [val, idx] = fx.min(), fx.idxmin()[0]  # índice y valores que minimizan

        if val.values < float(fx_pg): #actualización de mínimos
            fx_pg = val
            x1_pg = x1p["learning_rate"][idx]
            x2_pg = x1p['neuron percentage'][idx]
            x3_pg = x1p['batch size'][idx]
            x4_pg = x1p['layer percentage'][idx]
            x5_pg = x1p["dropout"][idx]

        for k in range(0, n_particles):
            if fx[0][k] < fx_pL[k]:
                fx_pL[k] = fx[0][k]
                x1pL['learning_rate'][k] = x1p["learning_rate"][k]  # diccionario de parámetros
                x1pL['neuron percentage'][k] = x1p["neuron percentage"][k]
                x1pL['batch size'][k] = x1p['batch size'][k]
                x1pL['layer percentage'][k] = x1p["layer percentage"][k]
                x1pL["dropout"][k] = x1p["dropout"][k]

        x1p = paramspeed_update(x1p, velocidad_x1, c1, [x1_pg, x2_pg, x3_pg, x4_pg,x5_pg],
                                c2, x1pL) #actualización de velocidad


    parametros = {'learning_rate': x1_pg, #diccionario de salida
                  'neuron_percentage': x2_pg,
                  'batch size': x3_pg,
                  'layer percentage': x4_pg,
                  'dropout':x5_pg,
                  'funcion de costo': fx_pg
                    }

    return parametros