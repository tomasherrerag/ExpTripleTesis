import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def obtener_top_trials(db_path, study_name, porcentaje=0.10):

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}"
    )

    trials = [
        t for t in study.trials
        if t.value is not None
    ]

    trials.sort(key=lambda t: t.value)

    n = max(1, int(len(trials) * porcentaje))

    mejores = trials[:n]

    filas = []

    for t in mejores:

        fila = {
            "score": t.value
        }

        fila.update(t.params)

        filas.append(fila)

    return pd.DataFrame(filas)






#bloque MAIN

#Extraccion de los mejores trials de cada experimento (10% mejores)
exp1 = obtener_top_trials(
    "exp1.db",
    "experimento1",
    porcentaje=0.10
)

exp2 = obtener_top_trials(
    "exp2.db",
    "experimento2",
    porcentaje=0.10
)

exp3 = obtener_top_trials(
    "exp3.db",
    "experimento3",
    porcentaje=0.10
)


# extraccion de parametros y graficado de KDE (para buscar rangos de valores optimos)
parametros = [
    c for c in exp1.columns
    if c != "score"
]

for parametro in parametros:

    plt.figure(figsize=(8,5))

    sns.kdeplot(exp1[parametro], label="Exp1")
    sns.kdeplot(exp2[parametro], label="Exp2")
    sns.kdeplot(exp3[parametro], label="Exp3")

    plt.title(parametro)

    plt.legend()

    plt.tight_layout()

    plt.savefig(f"kde_{parametro}.png")

    plt.close()

    resumen = []

for parametro in parametros:

    print(parametro)

    print(exp1[parametro].mean())
    print(exp2[parametro].mean())
    print(exp3[parametro].mean())

    minimo_comun = max(
        exp1[parametro].min(),
        exp2[parametro].min(),
        exp3[parametro].min()
    )

    maximo_comun = min(
        exp1[parametro].max(),
        exp2[parametro].max(),
        exp3[parametro].max()
    )

    resumen.append({
        "parametro": parametro,
        "min": minimo_comun,
        "max": maximo_comun
    })

rangos = pd.DataFrame(resumen)

print(rangos)