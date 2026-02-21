import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
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

def rosenbrock(p):
    sum = 0
    print('p:', p)

    for i in range(len(p)-1):
        xi = p[i]
        xii = p[i+1]
        part = 100*(xii-xi**2)**2 + (xi-1)**2
        sum+=part
    return sum

# %%
from sko.GA import GA
from sko.GA import RCGA
from sko.operators import selection


class GA_t(GA):
    def __init__(self, func, n_dim, size_pop=50, max_iter=100, prob_mut=0.01, 
                 lb=-1, ub=1, constraint_eq=tuple(), constraint_ueq=tuple(), 
                 precision=0.01, early_stop=None, n_processes=0, 
                 tournament_size=3): 
        

        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, lb, ub, 
                         constraint_eq, constraint_ueq, precision, early_stop, 
                         n_processes)
        
        self.tournament_size = tournament_size

    def selection(self):
        return selection.selection_tournament_faster(self, tourn_size=self.tournament_size)

class RCGA_t(RCGA):
    # Inicialitzador adaptat per a RCGA afegint tournament_size
    def __init__(self, func, n_dim, size_pop=50, max_iter=200,
                 prob_mut=0.001, prob_cros=0.9, lb=-1, ub=1,
                 n_processes=0, tournament_size=3):
        
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, prob_cros, lb, ub, n_processes)
        self.tournament_size = tournament_size

    def selection(self):
        return selection.selection_tournament_faster(self, tourn_size=self.tournament_size)
# sweep of parameters:
# 3 experiments, 1 per a cada paràmetre (pop size, mutation vals, tournament) modifiquem el valor de dit paràmetre 
# i mantenim els valors estàndard (els del mig) pals demès, 30 voltes cada
#  or grid search with these values
#do a loop for averaging the same combination of parameters average of the best fitness and average of the average



# best_fitnesses=[]
# size_pop_vals=[20, 50, 1000]
# prob_mut_vals=[0.25, 0.01, 0.001]
# tournament_vals=[1, 3, 10] # modified in selection.py or we can create a way to do it directly from here
# precision_vals=[0.2, 0.01, 0.001]

# for j in range(3):
#     for i in range(30):
#         ga = GA_t(func=schaffer, n_dim=2, size_pop=size_pop_vals[j], max_iter=100, prob_mut=prob_mut_vals[1], lb=[-1, -1], ub=[1, 1], precision=precision_vals[1], tournament_size=tournament_vals[1])

#         best_x, best_y = ga.run()
#         print('Config ', size_pop_vals[j], ' ', 'best_x:', best_x, '\n', 'best_y:', best_y)
#         best_fitnesses.append(best_y)



# avg_fitness=np.mean(best_fitnesses)
# std_fitness=np.std(best_fitnesses)
# print('avg_fitness:', avg_fitness, '\n', 'std_fitness:', std_fitness)


# Valors a testejar
default_pop = 50
default_mut = 0.01
default_tourn = 3
default_prec = 0.01

experiments = {
    "size_pop": [20, 50, 1000],
    "prob_mut": [0.1, 0.01, 0.001],
    "tournament_size": [1, 3, 10],
    "precision": [0.1, 0.01, 0.001]
}

funcions_disponibles = {
    "schaffer": schaffer,
    "rosenbrock": rosenbrock
}

experiment_config = {
    "func_name": "rosenbrock", # "rosenbrock" or "schaffer"
    "n_dim": 4,                # 2 for Schaffer, 4 for Rosenbrock
    "coding_type": "binary",   # "binary" or "real"
    "bounds": [-5, 5]          # inferior and superior bounds for the search space
}

max_iter_experiment = 100 
results = []

for param_name, values in experiments.items():
    
    # Ens saltem 'precision' si estem fent Real Coding
    if experiment_config["coding_type"] == "real" and param_name == "precision":
        continue
        
    print(f"\n--- Analitzant: {param_name} ---")
    
    for val in values:
        current_params = {
            "size_pop": default_pop,
            "prob_mut": default_mut,
            "tournament_size": default_tourn,
            "precision": default_prec
        }
        
        current_params[param_name] = val
        
        best_fitnesses = []
        
        # --- 1. PREPARACIÓ PEL PLOT: Matrius buides per guardar l'historial ---
        # Forma: 30 files (runs) x 100 columnes (generacions)
        all_runs_best_y = np.zeros((30, max_iter_experiment))
        all_runs_avg_y = np.zeros((30, max_iter_experiment))
        # ----------------------------------------------------------------------
        
        for i in range(30):
            # Creem els límits de forma dinàmica segons les dimensions (n_dim)
            lb_dinamic = [experiment_config["bounds"][0]] * experiment_config["n_dim"]
            ub_dinamic = [experiment_config["bounds"][1]] * experiment_config["n_dim"]
            
            # Obtenim la funció de Python (no el text)
            funci_a_optimitzar = funcions_disponibles[experiment_config["func_name"]]
            
            if experiment_config["coding_type"] == "binary":
                ga = GA_t(func=funci_a_optimitzar, 
                          n_dim=experiment_config["n_dim"], 
                          max_iter=max_iter_experiment, 
                          lb=lb_dinamic, 
                          ub=ub_dinamic, 
                          **current_params)
            else:
                params_reals = current_params.copy()
                params_reals.pop("precision", None) 
                
                ga = RCGA_t(func=funci_a_optimitzar, 
                          n_dim=experiment_config["n_dim"],  
                          max_iter=max_iter_experiment, 
                          lb=lb_dinamic, 
                          ub=ub_dinamic, 
                          **params_reals)    
                
            best_x, best_y = ga.run()
            
            # --- 2. EXTRACCIÓ DE DADES PER A LA TAULA ---
            if isinstance(best_y, (list, np.ndarray)):
                best_y_val = best_y[0]
            else:
                best_y_val = best_y
                
            best_fitnesses.append(best_y_val)
            
            # --- 3. EXTRACCIÓ DE DADES PEL PLOT (Historial de la generació) ---
            # Aplanem el best_Y (perquè scikit-opt ho guarda com una llista d'arrays)
            best_y_history = np.array([v[0] if isinstance(v, (list, np.ndarray)) else v 
                                       for v in ga.generation_best_Y])
            
            # Calculem la mitjana de TOTA la població en cada generació
            avg_y_history = np.array([np.mean(pop_Y) for pop_Y in ga.all_history_Y])
            
            # Guardem aquestes dades a la matriu general
            actual_iters = len(best_y_history)
            all_runs_best_y[i, :actual_iters] = best_y_history
            all_runs_avg_y[i, :actual_iters] = avg_y_history
            
            # Si per algun motiu l'algorisme para abans d'hora, omplim la resta amb l'últim valor conegut
            if actual_iters < max_iter_experiment:
                all_runs_best_y[i, actual_iters:] = best_y_history[-1] if actual_iters > 0 else 0
                all_runs_avg_y[i, actual_iters:] = avg_y_history[-1] if actual_iters > 0 else 0
            # ------------------------------------------------------------------

        # Càlculs per a la taula final
        avg_fitness = np.mean(best_fitnesses)
        std_fitness = np.std(best_fitnesses)
        
        print(f"Config: {param_name}={val} -> Avg final: {avg_fitness:.6f}, Std final: {std_fitness:.6f}")
        
        results.append({
            "Function": experiment_config["func_name"],
            "Coding": experiment_config["coding_type"],
            "Parameter": param_name,
            "Value": val,
            "Average Best Fitness": avg_fitness,
            "Std Dev": std_fitness
        })

        # --- 4. GENERAR I GUARDAR EL PLOT D'AQUESTA CONFIGURACIÓ ---
        # Calculem l'estadística generació a generació (verticalment al llarg de les 30 runs)
        mean_of_best = np.mean(all_runs_best_y, axis=0) 
        std_of_best = np.std(all_runs_best_y, axis=0)   
        mean_of_avg = np.mean(all_runs_avg_y, axis=0)   
        
        generations = np.arange(1, max_iter_experiment + 1)
        
        plt.figure(figsize=(9, 5))
        
        # Línia 1: Average best fitness
        plt.plot(generations, mean_of_best, label='Average Best Fitness (elit)', color='blue', linewidth=2)
        
        # Ombra: Desviació estàndard
        plt.fill_between(generations, mean_of_best - std_of_best, mean_of_best + std_of_best, 
                         color='blue', alpha=0.2, label='Std Dev (best fitness)')
        
        # Línia 2: Average of population average
        plt.plot(generations, mean_of_avg, label='Average of Population Avg', color='red', linestyle='--', linewidth=2)
        
        # Disseny i etiquetes
        exp_title = f"{experiment_config['func_name'].capitalize()} ({experiment_config['coding_type']}) - {param_name}={val}"
        plt.title(f'Convergence: {exp_title}', fontsize=14)
        plt.xlabel('Generations', fontsize=12)
        plt.ylabel('Fitness Score', fontsize=12)
        
        # Si és rosenbrock, fiquem escala logarítmica perquè es vegi bé la baixada
        if experiment_config["func_name"] == "rosenbrock":
            plt.yscale('log')
            
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        # Guardem com a imatge png (et crearà un fitxer a la mateixa carpeta per a cada paràmetre testejat)
        filename = f"plot_{experiment_config['func_name']}_{experiment_config['coding_type']}_{param_name}_{val}.png"
        plt.savefig(filename)
        plt.close() # Tanquem la figura per no saturar la memòria
        # -------------------------------------------------------------------

# Convertir a DataFrame per veure la taula final
df_results = pd.DataFrame(results)
print("\n" + "="*50)
print("TAULA FINAL DE RESULTATS:")
print("="*50)
print(df_results)