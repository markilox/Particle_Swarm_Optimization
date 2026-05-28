# Design Document — PSO Parallelism Benchmark

## 1. Visión general

Este proyecto implementa Particle Swarm Optimization (PSO) canónico como banco de pruebas para comparar estrategias de paralelismo y concurrencia en Python. El objetivo no es solo optimizar funciones, sino medir con rigor el impacto real de threading, multiprocessing y la versión secuencial sobre el tiempo de evaluación y el overhead de coordinación.

---

## 2. Arquitectura

El diseño sigue una separación estricta entre el **motor PSO** y las **estrategias intercambiables**, permitiendo añadir nuevos evaluadores, topologías o políticas de límites sin tocar el núcleo.

```
pso/
├── core/           # Motor PSO: estado, actualización, criterios de parada
│   ├── pso.py          PSO principal (bucle de optimización)
│   ├── particle.py     Estado individual de cada partícula
│   ├── swarm.py        Colección de partículas + global best
│   ├── bounds.py       Política de límites (ClampBoundsPolicy)
│   ├── topology.py     Topología (GlobalBestTopology)
│   └── types.py        Dataclasses compartidas (PSOConfig, BoxBounds, etc.)
├── objectives/     # Funciones benchmark
│   └── benchmarks.py   Sphere, Rastrigin, Rosenbrock, Ackley
├── parallel/       # Estrategias de evaluación de fitness
│   ├── base.py         Interfaz abstracta FitnessEvaluator
│   ├── sequential.py   V0: bucle secuencial
│   ├── threading_evaluator.py     V1: ThreadPoolExecutor
│   └── multiprocessing_evaluator.py  V2: ProcessPoolExecutor
├── experiments/    # Orquestación
│   ├── runner.py       ExperimentConfig + run_experiment()
│   └── grid_search.py  GridSearchConfig + run_grid_search()
├── io/             # Persistencia
│   └── persistence.py  save_result(), load_metadata(), load_history()
└── viz/            # Visualización
    ├── recorder.py     SwarmRecorder (callback on_iteration)
    ├── animation.py    animate_swarm() — GIF/MP4 para d=2 y d=3
    └── plots.py        plot_convergence() — PNG estático
```

### Principio de diseño clave

El `PSO` no sabe nada del evaluador concreto. Llama a `evaluator.evaluate(swarm, objective)` y el evaluador actualiza los `best_value` / `best_position` de cada partícula. Esto permite cambiar la estrategia de paralelismo inyectando un objeto distinto sin modificar el bucle principal.

---

## 3. Interfaces y abstracciones

### 3.1 FitnessEvaluator

```python
class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, swarm: Swarm, objective) -> None: ...
```

Todas las variantes (sequential, threading, multiprocessing) implementan esta interfaz. La función objetivo es un `callable(np.ndarray) -> float` envuelto en `ObjectiveSpec`.

### 3.2 Política de límites (ClampBoundsPolicy)

Estrategia elegida: **clamp + zeroing de velocidad**.

- Si una partícula sale del dominio en la dimensión `i`, su posición se recorta a `[lower_i, upper_i]` y su velocidad en esa dimensión se pone a 0.
- **Motivación**: Es la estrategia más sencilla y la más usada en literatura. Evita acumulación de partículas en los bordes (problema de absorbing walls) mejor que el clamp puro sin zeroing, y es más estable numéricamente que reflect (que puede producir oscilaciones) o penalty (que altera la función objetivo).
- **Limitación**: Las partículas tienden a "pegarse" brevemente al borde antes de moverse hacia el interior. En dimensiones altas esto no supone un problema significativo.

### 3.3 Topología (GlobalBestTopology)

Se implementa únicamente **global best** (gbest): todos los vecinos de cada partícula son el enjambre completo, y el atractor social es siempre la mejor posición global encontrada hasta el momento.

- **Ventaja**: convergencia rápida.
- **Limitación**: mayor riesgo de convergencia prematura en funciones multimodales (e.g., Rastrigin en d=30). Una topología de anillo (lbest) reduciría este riesgo a costa de convergencia más lenta.

La interfaz `GlobalBestTopology.social_best(swarm, particle_index)` está diseñada para que añadir nuevas topologías sea un cambio de una clase, sin modificar el PSO.

---

## 4. Estrategias de paralelismo

### V0 — Secuencial (baseline)

Evaluación y actualización en bucle Python puro. Es el punto de referencia para calcular speedup de las demás versiones.

### V1 — Threading (ThreadPoolExecutor)

Paraleliza la evaluación de fitness de todas las partículas usando un pool de hilos.

**Ciclo de vida del pool**: el `ThreadPoolExecutor` se crea una vez en `__init__` y se reutiliza en cada llamada a `evaluate()` durante toda la ejecución. Al terminar, `run_experiment()` llama a `close()` → `shutdown()` en un bloque `finally`. Esto elimina el overhead de crear y destruir hilos en cada iteración.

**¿Por qué no mejora en Python?**
El GIL (Global Interpreter Lock) impide que dos hilos ejecuten bytecode Python simultáneamente. Para funciones como Sphere o Rastrigin, que son operaciones NumPy puras, el overhead de sincronización de hilos supera el tiempo de evaluación por partícula. Los resultados experimentales muestran speedups de ~0.5x (2x más lento que secuencial) con el pool reutilizado.

**Número de workers**: con `max_workers=None`, Python crea `min(32, cpu_count + 4)` hilos. Para un enjambre de 30 partículas en una máquina de 20 cores, esto significa 24 hilos con ~1.25 partículas/hilo — el overhead de sincronización domina completamente. Se recomienda fijar `max_workers=4` (~7.5 partículas/worker) en `configs/benchmarks.yaml` para un ratio más razonable.

**Cuándo sería útil**: si la función objetivo implicara I/O (consultas a base de datos, llamadas HTTP) o código C externo que libera el GIL durante un tiempo significativo.

### V2 — Multiprocessing (ProcessPoolExecutor)

Paraleliza la evaluación usando procesos del sistema operativo, evitando el GIL.

**Ciclo de vida del pool**: el `ProcessPoolExecutor` se crea una vez en `__init__` y se reutiliza en cada iteración, igual que en threading. Esto evita el coste de arrancar y destruir el pool de procesos 200 veces por run (que antes dominaba el tiempo total). `run_experiment()` llama a `close()` en un `finally`.

**Coste de IPC residual**: aunque el pool se reutiliza, cada llamada a `executor.map()` sigue requiriendo serializar (pickle) las posiciones y el objetivo, enviarlos a los workers y recoger los resultados. Para funciones de benchmark (microsegundos por evaluación), este coste por iteración sigue siendo mayor que el cómputo. Speedup experimental: ~0.07x.

**Número de workers**: con `max_workers=None`, Python crea `cpu_count` procesos. En una máquina de 20 cores, esto son 20 procesos con 1.5 partículas/proceso — más overhead de IPC del necesario. Con `max_workers=4`, cada proceso recibe ~7.5 partículas y el ratio trabajo/coste mejora significativamente. Regla general: `swarm_size / max_workers ≥ 4–8`.

**Optimización con `chunksize`**: se envían varios argumentos por tarea para reducir el número de mensajes IPC. Mejora para enjambres grandes.

**Cuándo sería útil**: funciones objetivo computacionalmente costosas (simulaciones, modelos físicos, ejecutables externos) donde el tiempo de evaluación supera el overhead de IPC (regla empírica: > 10 ms por evaluación).

### V3 — Asyncio (concurrencia cooperativa)

Paraleliza la evaluación de fitness usando corrutinas con `asyncio`, sin crear hilos ni procesos adicionales.

**Caso de uso diseñado**: asyncio es concurrencia cooperativa, no paralelismo real. No acelera cómputo CPU-bound puro. Su utilidad aparece cuando la evaluación del fitness implica **esperas asíncronas**: consultas a un servicio externo, lectura de archivos, latencias de red, o cualquier función que simule evaluaciones con tiempos variables (ej. simuladores que tardan distinto según la partícula). El `AsyncioEvaluator` implementa esto inyectando `latency_s` y `jitter` configurables.

**Ciclo de vida del event loop**: se crea un `asyncio.new_event_loop()` en `__init__` y se reutiliza en todas las iteraciones mediante `loop.run_until_complete(...)`. Esto evita el overhead de crear un nuevo event loop en cada iteración.

**Cuatro estrategias implementadas**:

| Estrategia | Mecanismo | Cuándo preferirla |
|---|---|---|
| `gather` | `asyncio.gather(*coros)` — todas las corrutinas en paralelo | Caso general; recomendada por sencillez |
| `create_task` | `loop.create_task()` + `await` por tarea | Cuando se necesita control explícito de las tareas |
| `as_completed` | `asyncio.as_completed(coros)` | Cuando se quiere procesar resultados según llegan (útil con tiempos muy asimétricos) |
| `queue` | Producer-consumer con `asyncio.Queue` y `n_workers` consumidores | Cuando se quiere limitar la concurrencia máxima |

**Relación con el GIL**: asyncio no evita el GIL. Todas las corrutinas se ejecutan en el mismo hilo Python, alternando su ejecución en los puntos de `await`. Sin embargo, durante el `await asyncio.sleep()` (o una llamada I/O real), el hilo está libre y otras corrutinas pueden avanzar. Para funciones benchmark puras (sin latencia real), el overhead del event loop hace que sea más lento que el secuencial.

**Cuándo sería útil con latencias reales**: si cada partícula consulta un endpoint REST (p. ej. 50 ms de latencia × 30 partículas = 1500 ms secuencial vs ~50 ms con gather), asyncio ofrece un speedup teórico igual al número de partículas. El evaluador puede probarse con `latency_s=0.05, jitter=0.01` para simular este escenario.

### Resumen experimental

| Evaluador | Speedup vs secuencial | Condición | Caso de uso |
|---|---|---|---|
| Sequential | 1.0x (baseline) | — | Siempre para funciones baratas (< 1 ms/eval) |
| Threading | ~0.5x (más lento) | CPU-bound NumPy | I/O-bound o código C que libera el GIL |
| Multiprocessing | ~0.07x (más lento) | CPU-bound NumPy | Funciones costosas (> ~10 ms/eval) |
| Asyncio | ~0.3x sin latencia; speedup ≈ N con latencias I/O | Latencia simulada | Evaluaciones con I/O o latencias variables |

**Conclusión**: para las funciones de benchmark estándar, el evaluador secuencial es siempre el más eficiente. Las variantes paralelas/concurrentes son relevantes únicamente cuando el coste de evaluación lo justifica: threading y multiprocessing para cómputo intensivo que libera el GIL o supera el overhead de IPC; asyncio para evaluaciones I/O-bound o con latencias variables.

---

## 5. Persistencia

Formato elegido: **JSON + CSV**.

- `metadata.json`: configuración completa (parámetros PSO, función objetivo, evaluador, seed, commit de git, info de hardware). JSON es legible, sin dependencias y soporta estructuras anidadas.
- `history.csv`: métricas por iteración (best_fitness, eval_time_s, update_time_s). CSV es eficiente para series temporales y compatible directamente con pandas.

Cada ejecución crea su propio directorio `results/runs/{run_id}/` con ambos archivos, garantizando que los resultados no se sobreescriban.

---

## 6. Configuración

Todos los scripts leen de archivos YAML en `configs/`:

| Script | Config |
|---|---|
| `run_pso.py` | `configs/pso.yaml` |
| `run_benchmarks.py` | `configs/benchmarks.yaml` |
| `run_grid_search.py` | `configs/grid_search.yaml` |
| `make_viz.py` | `configs/viz.yaml` |

---

## 7. Limitaciones conocidas

- **GIL**: threading y asyncio no ofrecen paralelismo real para código CPU-bound en Python puro o NumPy. Es una limitación del intérprete CPython, no del diseño.
- **IPC overhead**: multiprocessing es contraproducente para funciones ligeras. Requiere funciones objetivo serializables (picklables), lo que excluye lambdas y closures complejos.
- **Asyncio sin I/O real**: el `AsyncioEvaluator` simula latencia con `asyncio.sleep`. En producción, la ganancia real solo aparece si la función objetivo hace I/O (HTTP, BD, ficheros) o llama a código que libera el GIL durante la espera.
- **Topología única**: solo se implementa global best. Funciones multimodales en alta dimensión (Rastrigin d=30) podrían beneficiarse de topologías locales.
- **Sin paralelismo en actualización**: la fase de actualización de velocidades/posiciones es siempre secuencial.
- **Reproducibilidad con multiprocessing**: el orden de evaluación es determinista (executor.map preserva orden), pero el seed del RNG principal no se propaga a los workers (no lo necesitan, ya que solo evalúan la función).
- **Reproducibilidad con asyncio**: con `latency_s=0` (sin latencia), el orden de `gather` es determinista y los resultados son idénticos a los del secuencial. Con latencia y jitter activados, el orden de finalización puede variar entre ejecuciones; los resultados de optimización son igualmente correctos porque el swarm update se aplica una vez todas las corrutinas han terminado.
