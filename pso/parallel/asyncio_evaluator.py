import asyncio
import random

from pso.parallel.base import FitnessEvaluator


class AsyncioEvaluator(FitnessEvaluator):
    """Evaluador de fitness basado en asyncio (V3 – concurrencia cooperativa).

    Usa asyncio.gather() para despachar todas las evaluaciones a la vez.
    El beneficio real aparece cuando la función objetivo es I/O-bound
    (latencia de red, simulador externo): el tiempo por iteración pasa de
    N × latencia  a  max(latencias individuales).

    Args:
        latency_s:  Latencia base simulada por partícula (segundos).
                    0.0 → sin simulación de I/O (overhead puro de corutinas).
        jitter:     Variación aleatoria ±jitter sobre latency_s.
                    Modela tiempos de respuesta asimétricos (caso realista).
        seed:       Semilla del PRNG para el jitter (no afecta al PSO).
    """

    def __init__(
        self,
        latency_s: float = 0.0,
        jitter: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._latency_s = latency_s
        self._jitter = jitter
        self._rng = random.Random(seed)

        # Bucle de eventos reutilizado entre iteraciones para evitar el coste
        # de crear/destruir el loop en cada llamada.
        self._loop = asyncio.new_event_loop()



    def evaluate(self, swarm, objective) -> None:
        """Punto de entrada síncrono.

        Bloquea el hilo llamante hasta que todas las corutinas han completado,
        usando loop.run_until_complete().
        """
        self._loop.run_until_complete(self._gather(swarm, objective))

    def close(self) -> None:
        """Cierra el bucle de eventos y libera recursos."""
        if not self._loop.is_closed():
            self._loop.close()



    async def _gather(self, swarm, objective) -> None:
        """Evalúa todas las partículas con asyncio.gather().

        gather() convierte las corutinas en Tasks implícitamente y devuelve
        los resultados en el mismo orden de la secuencia original.
        Tiempo total ≈ max(latencias individuales).
        """
        coroutines = [self._eval_particle(p, objective) for p in swarm.particles]
        results = await asyncio.gather(*coroutines)
        self._apply_results(results)



    async def _eval_particle(self, particle, objective) -> tuple:
        """Evalúa una partícula de forma asíncrona.

        await asyncio.sleep(delay) simula la espera I/O liberando el event
        loop para ejecutar otras corutinas mientras esta partícula espera.
        """
        if self._latency_s > 0.0 or self._jitter > 0.0:
            delay = max(
                0.0,
                self._latency_s + self._rng.uniform(-self._jitter, self._jitter),
            )
            await asyncio.sleep(delay)

        value = float(objective(particle.position))
        return particle, value



    def _apply_results(self, results: list[tuple]) -> None:
        for particle, value in results:
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()
