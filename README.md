# Particle Swarm Optimization

Implementacion base de PSO con foco en arquitectura modular para extender a versiones paralelas.

## Estado actual

- V0 secuencial implementado.
- Funciones benchmark incluidas: `sphere`, `rosenbrock`, `rastrigin`, `ackley`.
- Restricciones de caja con politica explicita: `ClampBoundsPolicy` (clip de posicion y velocidad a 0 en componentes que chocan con el limite).
- Criterios de parada: maximo de iteraciones, tolerancia y estancamiento.
- Logging por iteracion con estructura JSON.

## Estructura

```
pso/
  core/
    particle.py
    swarm.py
    pso.py
  objectives/
    benchmarks.py
parallel/
  base.py
  sequential.py
app.py
```

## Ejecucion

```bash
python app.py
```

Otra ejecucion de ejemplo:

```bash
python app.py
```

## API secuencial (V0)

- Objetivo: funcion que recibe `np.ndarray` y devuelve `float`.
- Motor: `PSO(objective, bounds, config)`.
- Resultado: `OptimizationResult` con mejor solucion, historial por iteracion y tiempos.

## Siguientes pasos

- `experiments/`: runners reproducibles y grid search.
- `io/`: persistencia estructurada de resultados.
- `viz/`: curvas y animaciones para `d=2`/`d=3`.
- `parallel/`: V1-V4 reutilizando el mismo core (V0 secuencial ya integrado).
