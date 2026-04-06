"""make_viz.py — Genera animaciones y plots de convergencia del PSO.

Para d=2: animación con contorno de la función + posiciones del enjambre + curva de convergencia.
Para d=3: animación 3D con scatter del enjambre + curva de convergencia.
Para d>3: solo curva de convergencia animada.

Además siempre guarda un PNG estático con la curva de convergencia.

Uso:
  python make_viz.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pso.core.types import PSOConfig, StopCriteria, BoxBounds
from pso.core.pso import PSO
from pso.objectives.benchmarks import get_objective, list_objectives
from pso.viz import SwarmRecorder, animate_swarm, plot_convergence


def ask_one(prompt: str, options: list[str], default: str) -> str:
    print(f"  Opciones: {', '.join(options)}")
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    if raw not in options:
        print(f"  Opción no válida. Se usará '{default}'.")
        return default
    return raw


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("  Valor no válido. Se usará el valor por defecto.")
        return default


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("  Valor no válido. Se usará el valor por defecto.")
        return default


def ask_str(prompt: str, default: str) -> str:
    raw = input(f"  {prompt} [{default}]: ").strip()
    return raw if raw else default


def prompt_params() -> dict:
    print("\n=== Configuración de la visualización ===\n")

    print("Función objetivo:")
    objective = ask_one("Selecciona", list_objectives(), "sphere")

    print("\nEspacio de búsqueda:")
    dimension = ask_int("Dimensión (2 o 3 para animación completa)", 2)

    print("\nParámetros del PSO:")
    swarm_size = ask_int("Tamaño del enjambre", 30)
    iterations = ask_int("Número de iteraciones", 100)
    inertia    = ask_float("Inercia w", 0.7)
    cognitive  = ask_float("Cognitivo c1", 1.5)
    social     = ask_float("Social c2", 1.5)
    seed       = ask_int("Semilla aleatoria", 42)

    print("\nAnimación:")
    fmt    = ask_one("Formato (gif requiere pillow, mp4 requiere ffmpeg)", ["gif", "mp4"], "gif")
    fps    = ask_int("Fotogramas por segundo", 10)
    every_n = ask_int("Guardar frame cada N iteraciones", 1)
    resolution = ask_int("Resolución del contorno para d=2", 150)

    print("\nSalida:")
    output_dir = ask_str("Directorio de resultados", "results/viz")

    return {
        "objective": objective,
        "dimension": dimension,
        "swarm_size": swarm_size,
        "iterations": iterations,
        "inertia": inertia,
        "cognitive": cognitive,
        "social": social,
        "seed": seed,
        "format": fmt,
        "fps": fps,
        "every_n": every_n,
        "resolution": resolution,
        "output_dir": output_dir,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    p = prompt_params()

    objective = get_objective(p["objective"])
    lower, upper = objective.bounds(p["dimension"])
    bounds = BoxBounds(lower=lower, upper=upper)

    config = PSOConfig(
        dimension=p["dimension"],
        swarm_size=p["swarm_size"],
        inertia_weight=p["inertia"],
        cognitive_weight=p["cognitive"],
        social_weight=p["social"],
        stop=StopCriteria(max_iterations=p["iterations"]),
        seed=p["seed"],
    )

    pso = PSO(objective=objective, bounds=bounds, config=config)
    recorder = SwarmRecorder(every_n=p["every_n"])

    print(f"\nEjecutando PSO: {p['objective']} d={p['dimension']}, {p['iterations']} iteraciones...")
    result = pso.optimize(on_iteration=recorder)
    print(f"  Mejor valor: {result.best_value:.6e}  |  frames grabados: {len(recorder)}")

    output_dir = Path(p["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{p['objective']}_d{p['dimension']}_seed{p['seed']}"

    # --- animación ---
    anim_path = output_dir / f"{base}.{p['format']}"
    print(f"Generando animación → {anim_path}")
    try:
        animate_swarm(
            frames=recorder.frames,
            history=result.history,
            dimension=p["dimension"],
            output_path=anim_path,
            objective=objective,
            bounds=bounds,
            fps=p["fps"],
            resolution=p["resolution"],
        )
        print(f"  Animación guardada: {anim_path}")
    except Exception as exc:
        print(f"  ERROR al generar animación: {exc}", file=sys.stderr)
        print("  Asegúrate de tener 'pillow' instalado para GIF o 'ffmpeg' para MP4.", file=sys.stderr)

    # --- convergencia estática ---
    conv_path = output_dir / f"{base}_convergence.png"
    print(f"Generando curva de convergencia → {conv_path}")
    fig = plot_convergence(
        history=result.history,
        title=f"Convergencia — {p['objective']}  d={p['dimension']}  seed={p['seed']}",
    )
    fig.savefig(conv_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {conv_path}")

    print("\nListo.")


if __name__ == "__main__":
    main()
