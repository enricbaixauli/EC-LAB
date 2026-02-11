import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    '''
    x1, x2 = p
    part1 = np.square(x1) - np.square(x2)
    part2 = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(part1)) - 0.5) / np.square(1 + 0.001 * part2)


# %%
from sko.GA import GA
from sko.operators import selection

# 3. DEFINIR: Definim la teva classe AQUÍ mateix (no la importis, crea-la)
class GA_t(GA):
    def __init__(self, func, n_dim, size_pop=50, max_iter=100, prob_mut=0.01, 
                 lb=-1, ub=1, constraint_eq=tuple(), constraint_ueq=tuple(), 
                 precision=0.01, early_stop=None, n_processes=0, 
                 tournament_size=3): 
        
        # Inicialitzem el pare (GA original)
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, lb, ub, 
                         constraint_eq, constraint_ueq, precision, early_stop, 
                         n_processes)
        
        self.tournament_size = tournament_size

    def selection(self):
        return selection.selection_tournament_faster(self, tourn_size=self.tournament_size)

# sweep of parameters:
# 3 experiments, 1 per a cada paràmetre (pop size, mutation vals, tournament) modifiquem el valor de dit paràmetre 
# i mantenim els valors estàndard (els del mig) pals demès, 30 voltes cada
#  or grid search with these values
best_fitnesses=[]
size_pop_vals=[20, 50, 1000]
prob_mut_vals=[0.25, 0.01, 0.001]
tournament_vals=[1, 3, 10] # modified in selection.py or we can create a way to do it directly from here
precision_vals=[0.2, 0.01, 0.001]

#do a loop for averaging the same combination of parameters average of the best fitness and average of the average
for j in range(3):
    for i in range(30):
        ga = GA_t(func=schaffer, n_dim=2, size_pop=size_pop_vals[j], max_iter=100, prob_mut=prob_mut_vals[1], lb=[-1, -1], ub=[1, 1], precision=precision_vals[1], tournament_size=tournament_vals[1])

        best_x, best_y = ga.run()
        print('Config ', size_pop_vals[j], ' ', 'best_x:', best_x, '\n', 'best_y:', best_y)
        best_fitnesses.append(best_y)



avg_fitness=np.mean(best_fitnesses)
std_fitness=np.std(best_fitnesses)
print('avg_fitness:', avg_fitness, '\n', 'std_fitness:', std_fitness)

# %% Plot the result
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
