from amuse.datamodel import Particles
from amuse.units import units


class NoStellarEvolution():
    def __init__(self, **options):
        self.particles = Particles()
        self.model_time = 0.0 | units.Myr

    def evolve_model(self, end_time=None, keep_synchronous=None):
        self.model_time = end_time
