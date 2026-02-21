import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Configuració de quines imatges volem unir (adapta-ho si has fet Schaffer o Real)
func_name = "rosenbrock"
coding_type = "binary"

# Aquest és l'ordre de les columnes
parametres = ["size_pop", "prob_mut", "tournament_size", "precision"]

# Aquests són els valors (cada valor serà una fila)
valors_per_parametre = {
    "size_pop": [20, 50, 1000],
    "prob_mut": [0.1, 0.01, 0.001],
    "tournament_size": [1, 3, 10],
    "precision": [0.1, 0.01, 0.001]
}

# Creem una figura gran (4 columnes x 3 files)
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 15))
# fig.suptitle(f'Resum de Tots els Paràmetres - {func_name.capitalize()} ({coding_type})', fontsize=24, y=0.98)

for col_idx, param in enumerate(parametres):
    valors = valors_per_parametre[param]
    
    for row_idx, val in enumerate(valors):
        ax = axes[row_idx, col_idx]
        
        # Construïm el nom exacte de l'arxiu que va generar el teu codi
        filename = f"plot_{func_name}_{coding_type}_{param}_{val}.png"
        
        if os.path.exists(filename):
            # Llegim i mostrem la imatge
            img = mpimg.imread(filename)
            ax.imshow(img)
            ax.axis('off') # Amaguem els eixos perquè la imatge ja els té
        else:
            # Si falta alguna imatge, deixem el buit i avisem
            ax.text(0.5, 0.5, f"Falta:\n{filename}", ha='center', va='center', fontsize=12)
            ax.axis('off')

# Ajustem els marges perquè quedi ben atapeït i maco
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Guardem el resultat final
nom_sortida = f"RESUM_{func_name}_{coding_type}.png"
plt.savefig(nom_sortida, dpi=300, bbox_inches='tight')
print(f"Iupi! S'ha guardat la imatge combinada com a: {nom_sortida}")