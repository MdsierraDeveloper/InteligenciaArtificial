import math
import time
import timeit
import statistics
import numpy as np, pandas as pd


from aco import ACO, Graph
from plot import plot
import matplotlib.pyplot as plt


def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def main():
    cities = []
    points = []
    with open('./data/coord.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities.append(dict(index=int(city[0]), x=float(city[1]), y=float(city[2])))
            points.append((float(city[1]), float(city[2])))
    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)
    
    
    
    res = []
    end_times = [] 
    n_gens = []
    
    nejecuciones = 20
    
    for i in range(nejecuciones):
        
       start_time = timeit.default_timer() #Para tomar el tiempo de ejecución
       #ant_countt, generations, alpha, beta, rho, q, sategy)
       aco = ACO(20, 600, 1, 2, 0.5, 1, 3)
       #aco = ACO(20, 300, 1, 2, 0.8, 1, 3)
       graph = Graph(cost_matrix, rank)

       path, cost, sln, n_gen = aco.solve(graph)

       stop_time = timeit.default_timer()
       end_time = stop_time - start_time
       end_times.append(end_time)
       res.append(cost)
       n_gens.append(n_gen)
       
       print('cost: {}, path: {}'.format(cost, path))
       print("running_time: ",format(end_time, '.8f'))
       print("Encontró la solución en %d iteraciones" % n_gen)
    #plot(points, path)
       plt.plot(sln)
       plt.ylabel("Fitness")
       plt.xlabel("Iteration")
       plt.show()


    print("Costo total promedio =", round(statistics.mean(res), 2))
    varianza = statistics.variance(res)
    print("Varianza: ", round(varianza,3))
    print("Mejor distancia total encontrada en %d correidas: %f" % (nejecuciones, min(res)))

    df = pd.DataFrame(res, columns =['Costo_total'], dtype = float)
    df['Time'] = end_times
    df['Iteracion'] = n_gens
    print(df.head())
    df.to_excel('resultado3.xlsx')



if __name__ == '__main__':
    main()
