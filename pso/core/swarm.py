class swarm:
    def __init__(self, particles):
        self.particles = particles
        self.global_best_position = None
        self.global_best_value = float('inf')