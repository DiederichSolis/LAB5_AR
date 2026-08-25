# Laboratorio 5 - Aproximación de funciones

Entrega parcial del curso CC3104 - Aprendizaje por Refuerzo. Compara SARSA
semi-gradiente con aproximación lineal y Q-Learning tabular en una intersección
de tráfico simplificada.

## Entregables

- `Lab5_entrega_parcial.pdf`: informe con Tasks 1 y 2, gráficas y análisis.
- `Lab5_entrega_parcial.ipynb`: implementación documentada y resultados.

## Estructura

- `lab5_experiment.py`: entorno Gymnasium, agentes, entrenamiento y gráficas.
- `Lab5_entrega_parcial.ipynb`: implementación ejecutada de Task 2.
- `Lab5_entrega_parcial.pdf`: informe final con Task 1, Task 2 y conclusiones.
- `resultados/`: métricas y cuatro figuras producidas por la corrida reproducible.

## Reproducción

Se recomienda Python 3.10 o posterior.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python lab5_experiment.py
```

La corrida principal usa cinco semillas, 600 episodios por semilla y 100 pasos
por episodio. El entorno se valida con `gymnasium.utils.env_checker.check_env`.
El notebook entregado también puede abrirse y ejecutarse de principio a fin con
un kernel que tenga instaladas las dependencias anteriores.
