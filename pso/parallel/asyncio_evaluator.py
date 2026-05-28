import asyncio
import random
from typing import Literal

from pso.parallel.base import FitnessEvaluator

Strategy = Literal["gather", "tasks", "as_completed", "queue"]


class AsyncioEvaluator(FitnessEvaluator):
    """Evaluador de fitness basado en asyncio (V3 – concurrencia cooperativa).

    Args:
        latency_s:  Latencia base simulada por partícula (segundos).
                    0.0 → sin simulación de I/O (overhead puro de corutinas).
        jitter:     Variación aleatoria ±jitter sobre latency_s.
                    Modela tiempos de respuesta asimétricos (caso realista).
        strategy:   Primitiva asyncio utilizada para la concurrencia:
                    'gather' | 'tasks' | 'as_completed' | 'queue'.
        n_workers:  Número de consumidores para la estrategia 'queue'.
        seed:       Semilla del PRNG para el jitter (no afecta al PSO).
    """

    def __init__(
        self,
        latency_s: float = 0.0,
        jitter: float = 0.0,
        strategy: Strategy = "gather",
        n_workers: int = 4,
        seed: int | None = None,
    ) -> None:
        self._latency_s = latency_s
        self._jitter = jitter
        self._strategy = strategy
        self._n_workers = n_workers
        # PRNG independiente para el jitter; asyncio es hilo único → seguro.
        self._rng = random.Random(seed)

        # ── Bucle de eventos explícito (Tema 4, pág. 22-23) ──────────────
        # Se crea una vez y se reutiliza en todas las iteraciones del PSO
        # para evitar el coste de crear/destruir el loop en cada llamada.
        self._loop = asyncio.new_event_loop()

    # ------------------------------------------------------------------ #
    # Interfaz síncrona requerida por FitnessEvaluator                    #
    # ------------------------------------------------------------------ #

    def evaluate(self, swarm, objective) -> None:
        """Punto de entrada síncrono.

        Bloquea el hilo llamante hasta que todas las corutinas de evaluación
        han completado, usando ``loop.run_until_complete()`` (pág. 22).
        """
        dispatch = {
            "gather":       self._strategy_gather,
            "tasks":        self._strategy_tasks,
            "as_completed": self._strategy_as_completed,
            "queue":        self._strategy_queue,
        }
        coro = dispatch[self._strategy](swarm, objective)
        self._loop.run_until_complete(coro)

    def close(self) -> None:
        """Cierra el bucle de eventos y libera recursos (pág. 22-23)."""
        if not self._loop.is_closed():
            self._loop.close()

    # ------------------------------------------------------------------ #
    # Estrategia 1 – asyncio.gather() (Tema 4, pág. 16)                  #
    # ─────────────────────────────────────────────────                   #
    # Opción recomendada del temario.  gather() convierte las corutinas   #
    # en Tasks implícitamente y devuelve resultados en orden de secuencia #
    # ------------------------------------------------------------------ #

    async def _strategy_gather(self, swarm, objective) -> None:
        """Evalúa todas las partículas con asyncio.gather().

        Las corutinas se pasan directamente; gather() las promociona a Tasks
        internamente.  Tiempo total ≈ max(latencias individuales).
        """
        coroutines = [
            self._eval_particle(p, objective)
            for p in swarm.particles
        ]
        # gather() devuelve la lista de resultados en el mismo orden
        results = await asyncio.gather(*coroutines)
        self._apply_results(results)

    # ------------------------------------------------------------------ #
    # Estrategia 2 – asyncio.create_task() explícito (Tema 4, pág. 12)   #
    # ─────────────────────────────────────────────────────────────────── #
    # Creación explícita de un Task por partícula.  Equivalent a gather() #
    # pero expone el objeto Task (nombre, cancel, done, etc.).            #
    # ------------------------------------------------------------------ #

    async def _strategy_tasks(self, swarm, objective) -> None:
        """Evalúa cada partícula creando un Task explícito por partícula.

        asyncio.create_task() programa la corutina en el event loop actual
        y devuelve un objeto Task que puede inspeccionarse o cancelarse.
        """
        tasks = [
            asyncio.create_task(
                self._eval_particle(p, objective),
                name=f"particle-{i}",          # identificador opcional (pág. 15)
            )
            for i, p in enumerate(swarm.particles)
        ]
        # await de cada Task en orden; todos ya están en marcha desde create_task
        results = [await t for t in tasks]
        self._apply_results(results)

    # ------------------------------------------------------------------ #
    # Estrategia 3 – asyncio.as_completed() (Tema 4, pág. 25)            #
    # ─────────────────────────────────────────────────────────────────── #
    # Procesa partículas en el orden en que terminan, no en el orden      #
    # original.  Útil cuando se quiere actualizar el mejor global tan     #
    # pronto como llega el primer resultado (early stopping).             #
    # ------------------------------------------------------------------ #

    async def _strategy_as_completed(self, swarm, objective) -> None:
        """Evalúa partículas y procesa cada resultado en cuanto está listo.

        asyncio.as_completed() devuelve un iterador de corutinas en el orden
        de finalización: la partícula más rápida se procesa primero.
        """
        coroutines = [
            self._eval_particle(p, objective)
            for p in swarm.particles
        ]
        results = []
        # El iterador devuelve corutinas envueltas; await cada una para
        # obtener su resultado en el orden de llegada (no de la secuencia).
        for coro in asyncio.as_completed(coroutines):
            result = await coro
            results.append(result)
        self._apply_results(results)

    # ------------------------------------------------------------------ #
    # Estrategia 4 – asyncio.Queue productor-consumidor (pág. 26-27)     #
    # ─────────────────────────────────────────────────────────────────── #
    # Productores depositan partículas en la cola; consumidores las       #
    # retiran, evalúan y depositan el resultado en otra cola.             #
    # No hay condiciones de carrera: asyncio es hilo único.               #
    # ------------------------------------------------------------------ #

    async def _strategy_queue(self, swarm, objective) -> None:
        """Modelo productor-consumidor con asyncio.Queue.

        work_q:   cola de partículas pendientes de evaluar.
        result_q: cola de (partícula, valor) ya evaluados.

        Se carga work_q de forma síncrona (put_nowait) antes de lanzar los
        consumidores, evitando la necesidad de corutinas productoras separadas.
        Al final de work_q se insertan n_workers centinelas (None) para que
        cada consumidor pueda detectar que no hay más trabajo y terminar.

        Los consumidores usan ``await work_q.get()`` (corutina awaitable,
        pág. 27) para suspenderse cooperativamente si la cola estuviera vacía.
        Cuando hay latencia (latency_s > 0) el ``await asyncio.sleep(delay)``
        permite que otros consumidores arranquen durante la espera.  Sin
        latencia se usa ``await asyncio.sleep(0)`` como punto de cesión
        explícito (pág. 7), garantizando que n_workers corutinas realmente
        se interleaven en lugar de que una acapare todos los elementos.
        """
        work_q: asyncio.Queue = asyncio.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        # Carga síncrona: todos los items están en la cola antes de lanzar
        # consumidores; así get() nunca bloquea realmente (siempre hay items).
        for p in swarm.particles:
            work_q.put_nowait(p)
        # Un centinela None por consumidor para señalizar fin de cola
        for _ in range(self._n_workers):
            work_q.put_nowait(None)

        # ── Consumidores: retiran, evalúan y depositan en result_q ───
        async def consumer() -> None:
            while True:
                # await get() es el punto de suspensión cooperativa (pág. 27):
                # cede el control al event loop si la cola estuviera vacía.
                particle = await work_q.get()
                work_q.task_done()
                if particle is None:   # centinela → este consumidor termina
                    break
                if self._latency_s > 0.0 or self._jitter > 0.0:
                    delay = max(
                        0.0,
                        self._latency_s + self._rng.uniform(-self._jitter, self._jitter),
                    )
                    await asyncio.sleep(delay)   # simula I/O asimétrico
                else:
                    # Sin latencia no hay await natural; sleep(0) cede el hilo
                    # al event loop para que otros consumidores puedan avanzar
                    # (pág. 7: "yield to the event loop").
                    await asyncio.sleep(0)
                value = float(objective(particle.position))
                await result_q.put((particle, value))

        # Lanzar n_workers consumidores concurrentemente
        await asyncio.gather(*[consumer() for _ in range(self._n_workers)])

        # Recoger todos los resultados de result_q
        results = []
        while not result_q.empty():
            results.append(result_q.get_nowait())

        self._apply_results(results)

    # ------------------------------------------------------------------ #
    # Corutina auxiliar compartida por todas las estrategias              #
    # ------------------------------------------------------------------ #

    async def _eval_particle(self, particle, objective) -> tuple:
        """Evalúa una partícula de forma asíncrona.

        1. ``await asyncio.sleep(delay)`` simula la espera de I/O liberando
           el event loop para ejecutar otras corutinas (Tema 4, pág. 7).
        2. Calcula el fitness de forma síncrona (adecuado para funciones
           rápidas o extensiones C que liberan el GIL, como NumPy).

        Returns:
            Tupla (Particle, float): partícula y su valor de fitness.
        """
        if self._latency_s > 0.0 or self._jitter > 0.0:
            delay = max(
                0.0,
                self._latency_s + self._rng.uniform(-self._jitter, self._jitter),
            )
            # await suspende esta corutina; el event loop ejecuta las demás
            await asyncio.sleep(delay)

        value = float(objective(particle.position))
        return particle, value

    # ------------------------------------------------------------------ #
    # Actualización del mejor personal                                    #
    # ------------------------------------------------------------------ #

    def _apply_results(self, results: list[tuple]) -> None:
        """Actualiza el mejor personal de cada partícula con los resultados.

        Se ejecuta en el hilo principal, tras completar todas las corutinas.
        Mismo patrón que MultiprocessingEvaluator para consistencia del core.
        """
        for particle, value in results:
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()
