import numpy as np
import matplotlib.pyplot as plt

# PSO estándar de Guofei trabaja en espacios continuos (números reales), mientras que el TSP es un problema discreto (orden de ciudades). 
# Para resolver esto sin usar la clase adaptada, aplicamos la técnica de Valores de Orden de Magnitud (ROV):
#    	El PSO optimiza un vector de números reales (ej: [0.5, 1.2, 0.1]).
#	Usamos np.argsort() para convertir esos reales en una permutación de índices 
# 	(ej: el menor es el índice 2, luego 0, luego 1 --> ruta [2, 0, 1]).
# De esta forma podemos adaptar el proceso para trabajar con valores discretos (posiciones de las ciudades). 
# En lugar de que las partículas busquen una secuencia de ciudades, buscan un punto en un espacio de N dimensiones.
# El Espacio de Búsqueda: Cada partícula tiene una posición X = (x_1, x_2, ..., x_n).
# La Transformación: La función np.argsort(x) es el puente. Si una partícula se mueve de x_1=0.5 a x_1=0.2, su posición 
# en el ranking de la ruta cambia, alterando la distancia total calculada.
# Ventaja: Puedes usar todos los parámetros estándar del PSO (inercia w, coeficientes c_1 y c_2) 
# para ajustar el comportamiento del enjambre.
# Inercia: Un valor alto (0.9) ayuda a explorar nuevas rutas; uno bajo (0.4) refina la ruta actual.

from sko.PSO import PSO
# 1. Configuración de ciudades
num_points = 15
np.random.seed(42) # Para que sea reproducible
points_coordinate = np.random.rand(num_points, 2)

# 2. Función de fitness con manejo de dimensiones
def objective_function(p):
    # p es un vector continuo de tamaño (n_dim,)
    # Usamos argsort para convertir el vector real en una ruta (0, 1, 2...)
    route = np.argsort(p)
    
    # Calcular distancia total
    vec_c1 = points_coordinate[route]
    vec_c2 = points_coordinate[np.roll(route, -1)] # Siguiente ciudad (circular)
    dist = np.sum(np.sqrt(np.sum((vec_c1 - vec_c2)**2, axis=1)))
    return dist

# 3. Configuración del PSO
# Aumentamos pop a 100 y max_iter a 500 para compensar la dificultad del espacio continuo
pso = PSO(func=objective_function, 
          n_dim=num_points, 
          pop=100,           # Más partículas ayudan a cubrir el espacio de permutaciones
          max_iter=500, 
          lb=[0] * num_points, 
          ub=[100] * num_points, 
          w=0.9,             # Inercia alta para explorar más al principio
          c1=0.8,            # Coeficiente cognitivo
          c2=0.8)            # Coeficiente social

# 4. Ejecución
pso.run()

# 5. Extraer y visualizar
best_route = np.argsort(pso.gbest_x)
print(f"Mejor ruta: {best_route}")
print(f"Distancia: {pso.gbest_y[0]}")

# Gráfico
plt.figure(figsize=(10, 5))
plot_route = np.append(best_route, best_route[0])
plt.plot(points_coordinate[plot_route, 0], points_coordinate[plot_route, 1], 'o-b', mfc='r')
for i, (x, y) in enumerate(points_coordinate):
    plt.text(x, y, f'  {i}', fontsize=12)
plt.title(f"TSP con PSO (Ranking Method)\nDistancia final: {pso.gbest_y[0]:.4f}")
plt.grid(True)
plt.show()