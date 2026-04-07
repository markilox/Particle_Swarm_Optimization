# Particle Swarm Optimization

Implementación completa de PSO (Particle Swarm Optimization) en Python, diseñada como banco de pruebas para comparar estrategias de paralelismo y concurrencia. Incluye benchmarks, visualización, grid search de hiperparámetros y persistencia estructurada de resultados.

## Contenido

- [Instalación](#instalación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Configuración](#configuración)
- [Comandos](#comandos)
- [Estrategias de paralelismo](#estrategias-de-paralelismo)
- [Reproducibilidad](#reproducibilidad)
- [Análisis de resultados](#análisis-de-resultados)

---

## Instalación

```bash
git clone https://github.com/markilox/Particle_Swarm_Optimization
cd Particle_Swarm_Optimization
pip install -r requirements.txt
```

Dependencias principales: `numpy`, `matplotlib`, `pandas`, `pyyaml`, `prettytable`, `pillow` (GIF), `ffmpeg` (MP4, instalación del sistema).

---

## Estructura del proyecto

```
pso/
├── core/           # Motor PSO (bucle, partículas, enjambre, límites, topología)
├── objectives/     # Funciones benchmark (Sphere, Rastrigin, Rosenbrock, Ackley)
├── parallel/       # Estrategias de evaluación: sequential, threading, multiprocessing
├── experiments/    # Runner de experimentos y grid search
├── io/             # Persistencia de resultados (JSON + CSV)
└── viz/            # Animaciones y gráficos de convergencia

configs/            # Archivos YAML de configuración
docs/               # Documento de diseño y notebook de análisis
results/            # Resultados generados (runs, benchmarks, grid_search, viz)
tests/              # Tests unitarios
```

---

## Configuración

Cada script lee su configuración de un archivo YAML en `configs/`. Edita el YAML correspondiente antes de ejecutar.

| Script | Config | Descripción |
|---|---|---|
| `run_pso.py` | `configs/pso.yaml` | Una ejecución PSO individual |
| `run_benchmarks.py` | `configs/benchmarks.yaml` | Suite completa de benchmarks |
| `run_grid_search.py` | `configs/grid_search.yaml` | Grid search de hiperparámetros |
| `make_viz.py` | `configs/viz.yaml` | Animación y gráfico de convergencia |

**Límites por defecto de cada función:**

| Función | lower | upper |
|---|---|---|
| sphere | -5.12 | 5.12 |
| rastrigin | -5.12 | 5.12 |
| rosenbrock | -2.048 | 2.048 |
| ackley | -32.768 | 32.768 |

Si no se especifican `lower_bound` / `upper_bound` en el YAML, se usan estos valores automáticamente.

---

## Comandos

### Ejecución individual

```bash
python run_pso.py
```

Lee `configs/pso.yaml`, corre el PSO y muestra resultados. Al terminar pregunta si generar visualización (usa `configs/viz.yaml`).

### Suite de benchmarks

```bash
python run_benchmarks.py
```

Ejecuta todas las combinaciones de (función × dimensión × evaluador × seed) definidas en `configs/benchmarks.yaml`. Guarda `summary_<ts>.csv` e `history_<ts>.csv` en `results/benchmarks/`.

### Grid search de hiperparámetros

```bash
python run_grid_search.py
```

Producto cartesiano de (w × c1 × c2 × swarm_size × seeds) según `configs/grid_search.yaml`. Guarda CSV con ranking de configuraciones en `results/grid_search/`.

### Visualización

```bash
python make_viz.py
```

Genera animación GIF/MP4 y gráfico de convergencia PNG según `configs/viz.yaml`. Para `d=2`: contorno de la función + posiciones del enjambre. Para `d=3`: scatter 3D. Para `d>3`: solo curva de convergencia.

### Tests

```bash
python -m pytest tests/ -v
```

---

## Estrategias de paralelismo

El núcleo PSO es idéntico en todas las variantes. Solo cambia el **evaluador de fitness**, inyectado como dependencia.

| Versión | Implementación | Cuándo conviene |
|---|---|---|
| **V0 Sequential** | Bucle Python puro | Siempre para funciones baratas (< 1 ms/eval) |
| **V1 Threading** | `ThreadPoolExecutor` (pool reutilizado) | Funciones I/O-bound o código C que libera el GIL |
| **V2 Multiprocessing** | `ProcessPoolExecutor` (pool reutilizado) + batching | Funciones costosas (> ~10 ms/eval) |

**Ciclo de vida del pool**: ambos evaluadores crean el pool de workers una vez en `__init__` y lo reutilizan en cada iteración. `run_experiment()` llama a `close()` en un bloque `finally` al terminar la ejecución.

**Nota sobre el GIL**: threading no ofrece paralelismo real para código CPU-bound en CPython. Para las funciones de benchmark (NumPy puro), el overhead de coordinación supera el tiempo de evaluación. Los resultados experimentales muestran speedups de ~0.5x (threading) y ~0.07x (multiprocessing) respecto al secuencial para estas funciones.

Para seleccionar el evaluador, edita el campo `evaluator` en el YAML correspondiente:

```yaml
evaluator: sequential   # sequential | threading | multiprocessing
```

---

## Reproducibilidad

Todas las ejecuciones aceptan `seed` en el YAML. Los resultados guardados en `results/` incluyen:

- Parámetros completos de configuración
- Commit de git activo
- Información de hardware (plataforma, versión de Python, número de cores)

Para reproducir exactamente una ejecución anterior, consulta el `metadata.json` del run correspondiente en `results/runs/<run_id>/`.

---

## Análisis de resultados

Abre el notebook de análisis desde la raíz del proyecto (importante para que las rutas funcionen):

```bash
jupyter notebook docs/analysis.ipynb
```

El notebook carga automáticamente el CSV más reciente de `results/benchmarks/` y genera:

- Curvas de convergencia por función, dimensión y evaluador
- Speedup vs secuencial
- Boxplots de fitness final por evaluador
- Desglose de tiempos (evaluación vs actualización)
- Tablas resumen de calidad y tiempos

Para el documento de diseño con decisiones de arquitectura, trade-offs y limitaciones: [`docs/DESIGN.md`](docs/DESIGN.md).
