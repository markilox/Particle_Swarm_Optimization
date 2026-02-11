import numpy as np
import random
class particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_value = float('inf')
    
    def update_velocity(self, global_best_position, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0):
        r1 = np.random.rand(*self.position.shape)
        r2 = np.random.rand(*self.position.shape)
        
        cognitive_component = cognitive_weight * r1 * (self.best_position - self.position)
        social_component = social_weight * r2 * (global_best_position - self.position)
        
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component
    
    def update_position(self, lower_bound, upper_bound):
        self.position += self.velocity
        self.position = np.clip(self.position, lower_bound, upper_bound)
    
    def evaluate(self, objective_function):
        value = objective_function(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()
    
    def __str__(self):
        return f"Position: {self.position}, Velocity: {self.velocity}, Best Position: {self.best_position}, Best Value: {self.best_value}"