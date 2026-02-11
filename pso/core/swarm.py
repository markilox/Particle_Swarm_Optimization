class swarm:
    def __init__(self, particles):
        self.particles = particles
        self.global_best_position = None
        self.global_best_value = float('inf')
    
    def update_global_best(self):
        for particle in self.particles:
            if particle.best_value < self.global_best_value:
                self.global_best_value = particle.best_value
                self.global_best_position = particle.best_position.copy()
    
    def __str__(self):
        return f"Global Best Position: {self.global_best_position}, Global Best Value: {self.global_best_value}"