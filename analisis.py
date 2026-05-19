import os

import optuna
import pandas as pd
import matplotlib.pyplot as plt

from optuna.importance import get_param_importances


# =========================
# CARGAR ESTUDIO
# =========================

study = optuna.load_study(
    study_name="tesis",
    storage="sqlite:///tesis.db"
)

df = study.trials_dataframe()


# =========================
# TOP 10 SCORES
# =========================

top10 = df.sort_values(
    by="value",
    ascending=False
).head(10)

print("\nTOP 10 SCORES:")
print(top10[[
    "number",
    "value"
]])

top10.to_csv("top10_scores.csv", index=False)


# =========================
# LISTA DE PARÁMETROS
# =========================

parametros = [
    "params_tiempoExp",
    "params_umbralActividad",
    "params_ruidoPM",
    "params_ruidoTalamo",
    "params_ruidoSNr",
    "params_ruidoGPe",
    "params_ruidoStrD2",
    "params_ruidoStrD1",
    "params_ruidoSTN",
    "params_ruidoTonos",
    "params_ruidoObjetivos",
    "params_tauObjStrD2",
    "params_tauObjStrD1",
    "params_tauTonesStrD2",
    "params_tauTonesStrD1",
    "params_tauTonesSTN",
    "params_tauStrD2GPe",
    "params_tauGPeSNr",
    "params_tauStrD1SNr",
    "params_tauSTNSNr",
    "params_tauSNrVA",
    "params_tauVAPM"
]


# =========================
# CREAR CARPETAS
# =========================

os.makedirs("graficos/histogramas", exist_ok=True)
os.makedirs("graficos/scatter", exist_ok=True)


# =========================
# HISTOGRAMAS
# =========================

print("\nGenerando histogramas...")

for param in parametros:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[param],
        bins=20
    )

    plt.xlabel(param)
    plt.ylabel("Frecuencia")

    plt.title(f"Distribución de {param}")

    plt.tight_layout()

    nombre = param.replace("params_", "")

    plt.savefig(
        f"graficos/histogramas/{nombre}_hist.png"
    )

    plt.close()


# =========================
# SCATTER PLOTS
# =========================

print("\nGenerando scatter plots...")

for param in parametros:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[param],
        df["value"]
    )

    plt.xlabel(param)
    plt.ylabel("Score")

    plt.title(f"{param} vs Score")

    plt.tight_layout()

    nombre = param.replace("params_", "")

    plt.savefig(
        f"graficos/scatter/{nombre}_scatter.png"
    )

    plt.close()

'''
# =========================
# IMPORTANCIA DE PARÁMETROS
# =========================

print("\nCalculando importancia de parámetros...\n")

importance = get_param_importances(study)

for k, v in importance.items():
    print(f"{k}: {v:.4f}")
'''