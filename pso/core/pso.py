import numpy as np

# traer la fábrica de objetivos y la clase de partícula explícitamente
from pso.objectives.benchmarks import get_objective
from pso.core.particle import particle

class PSO:
    def __init__(self, objective_name: str, num_particles: int, num_iterations: int):
        self.objective_function = get_objective(objective_name)
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.particles = []
        self.global_best_position = None
        self.global_best_value = float('inf')
    
    def initialize_particles(self):
        for _ in range(self.num_particles):
            position = np.random.uniform(self.objective_function.lower_bound, self.objective_function.upper_bound, size=10)
            velocity = np.random.uniform(-1, 1, size=10)
            p = particle(position, velocity)
            p.evaluate(self.objective_function)
            self.particles.append(p)
            if p.best_value < self.global_best_value:
                self.global_best_value = p.best_value
                self.global_best_position = p.best_position.copy()
    
    def optimize(self):
        self.initialize_particles()
        
        for iteration in range(self.num_iterations):
            for p in self.particles:
                p.update_velocity(self.global_best_position)
                p.update_position(self.objective_function.lower_bound, self.objective_function.upper_bound)
                p.evaluate(self.objective_function)
                
                if p.best_value < self.global_best_value:
                    self.global_best_value = p.best_value
                    self.global_best_position = p.best_position.copy()
            
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Global Best Value: {self.global_best_value:.6f}")