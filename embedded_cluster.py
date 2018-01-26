# -*- coding: utf-8 -*-
"""
Simultaneously evolves gas and stars in a cluster.

The stars and gas particles evolve in their respective codes.
The force of the gas particles on the star particles is calculated by a
separate code.
The force of the star particles on the gas particles is calculated by the star
code.
"""
from __future__ import (
        print_function,
        division,
        )
import os
import logging
import time as clock
import numpy as np
import matplotlib
matplotlib.use('Agg')
from amuse.units import units
from amuse.datamodel import AbstractParticleSet
from amuse.couple.bridge import Bridge
from amuse.support.console import set_printing_strategy
from amuse.io import (
        write_set_to_file,
        read_set_from_file,
        )

from plotting_fresco import gas_stars_plot
from parameters import Parameters
from setup_codes import (
        code_stars_field,
        code_sph_field,
        new_code_field_gravity,
        new_code_stars_gravity,
        new_code_sph,
        new_code_stellarevolution,
        )
from argumentparser import new_simulation_argument_parser


def find_outliers(
        particles,
        cutoff_radius,
        ):
    particles.r = particles.position.lengths()
    outliers = particles.select(
            lambda r: r > cutoff_radius, "r",
            )
    return outliers


class EmbeddedCluster(object):
    """
    Calculates a cluster with stars and gas
    """
    def velocity_dispersion(particles):
        N = len(particles)
        dv = particles.velocity - particles.center_of_mass_velocity()
        squarevelocities = dv*dv
        sigma = np.sqrt(squarevelocities.sum()/(N-1))
        return sigma

    AbstractParticleSet.add_global_function_attribute(
            "velocity_dispersion",
            velocity_dispersion,
            )

    def __init__(
            self,
            p,
            ):

        self.p = p

        self.time = self.p.time_start
        self.model_time = self.p.time_start

        self.plot_n = 0

        self.log_output = logging.getLogger(__name__)
        self.log_output.setLevel("INFO")
        self.log_file = open(self.p.log_output, 'w')
        self.log_energy_file = open(self.p.log_energy, 'w')

        self.unit_time = units.Myr
        self.unit_energy = units.erg
        self.unit_length = units.parsec
        self.unit_speed = units.kms
        self.unit_mass = units.MSun

        logheader = "#Time Energy Virial_radius "
        logheader += "Core_radius Core_density Bound_star_mass Sigma "
        logheader += "LR10 LR50 LR100 "
        logheader += "E_kin E_pot GAS r_virial r_core core_density "
        logheader += "bound_gas_mass Sigma LR10 LR50 LR100 E_kin E_pot\n"
        logheader += "#Units in [ %s ] [ %s ] [ %s ] [ %s ] [ %s ]\n" % (
                self.unit_time,
                self.unit_energy,
                self.unit_length,
                self.unit_speed,
                self.unit_mass,
                )
        self.log_file.write(logheader)
        self.log_file.flush()

    def setup(self):
        self.initialize_data()
        self.create_codes()
        self.create_bridge()
        self.store_encounter(
                setup=True,
                )
        self.log_energy()
        self.next_log_energy_time = self.model_time+p.timestep_log_energy
        self.log_diagnostic()
        self.next_log_diagnostic_time = (
                self.model_time
                + p.timestep_log_diagnostic)
        self.plot_model()
        self.next_plot_model_time = self.model_time+p.timestep_plot_model
        self.save_to_hdf()
        self.next_save_backup_time = self.model_time+p.timestep_save_backup

        self.current_clock = clock.time() | units.s

    def initialize_data(self):
        self.stars = read_set_from_file(
                p.dir_initialconditions + self.p.stars_initial_file,
                'amuse',
                )
        self.p.stars_n = len(self.stars)
        stars_mass = self.stars.mass.sum()

        self.log_output.info(
                "Read %i star particles, total mass %s" % (
                    self.p.stars_n,
                    stars_mass,
                    )
                )

        self.stars.radius = p.stars_interaction_radius

        all_gas_particles = read_set_from_file(
                p.dir_initialconditions + self.p.gas_initial_file,
                'amuse',
                )
        self.gas_particles = all_gas_particles[
                :int(len(all_gas_particles) / self.p.gas_subsample)
                ]
        self.gas_particles.mass *= self.p.gas_subsample
        self.p.gas_n = len(self.gas_particles)
        gas_mass = self.gas_particles.mass.sum()

        self.log_output.info(
                "Read %i gas particles, using %i, total mass %s" % (
                    len(all_gas_particles),
                    self.p.gas_n,
                    gas_mass,
                    ),
                )

        self.converter = p.converter

        self.stars_epsilon = \
            self.converter.to_si(self.p.stars_smoothing_fraction)
        self.gas_epsilon = self.converter.to_si(self.p.gas_smoothing_fraction)

        self.gas_particles.epsilon = self.gas_epsilon

    def create_codes(self):
        self.sph_code = new_code_sph(
                self.converter,
                self.gas_epsilon,
                p,
                )
        self.stars_evolution_code = new_code_stellarevolution(
                p,
                )
        self.stars_gravity_code = new_code_stars_gravity(
                self.converter,
                self.stars_epsilon,
                p,
                )
        self.collision_detection = \
            self.stars_gravity_code.stopping_conditions.collision_detection

        if p.codes_collision_detection:
            self.collision_detection.enable()

        field_gravity_code = new_code_field_gravity(
                self.converter,
                self.stars_epsilon,
                p,
                )

        self.stars_to_gas = code_stars_field(
                self.stars_gravity_code,
                field_gravity_code,
                )
        self.gas_to_stars = code_sph_field(
                self.sph_code,
                field_gravity_code,
                )

        self.stars_gravity_code.particles.add_particles(self.stars)
        self.stars_evolution_code.particles.add_particles(self.stars)
        self.sph_code.gas_particles.add_particles(self.gas_particles)

        self.channel_from_evolution_to_memory_for_stars = \
            self.stars_evolution_code.particles.new_channel_to(self.stars)
        self.channel_from_memory_to_gravity_for_stars = \
            self.stars.new_channel_to(self.stars_gravity_code.particles)
        self.channel_from_gravity_to_memory_for_stars = \
            self.stars_gravity_code.particles.new_channel_to(self.stars)
        self.channel_from_sphcode_to_memory_for_gas = \
            self.sph_code.gas_particles.new_channel_to(self.gas_particles)

    def create_bridge(self):
        timestep_interaction = \
            self.converter.to_si(self.p.timestep_interaction)
        self.bridge = Bridge(
            timestep=timestep_interaction,
            use_threading=False,
        )

        self.bridge.add_system(
            self.sph_code,
            (self.stars_to_gas,),
        )
        self.bridge.add_system(
            self.stars_gravity_code,
            (self.gas_to_stars,),
        )

    def save_to_hdf(self):
        self.sync_model()
        self.log_output.debug(
                "Saving backup at time %s", self.model_time
                )
        starfile = 'StarsDump%010.4f.hdf5' % (
                self.model_time.value_in(units.Myr),
                )
        gasfile = 'GasDump%010.4f.hdf5' % (
                self.model_time.value_in(units.Myr),
                )
        write_set_to_file(
                self.stars,
                p.dir_simulation + starfile,
                'amuse',
                )
        write_set_to_file(
                self.gas_particles,
                p.dir_simulation + gasfile,
                'amuse',
                )

    def save_to_text(self):
        self.sync_model()
        self.log_output.debug(
                "Saving txt backup at time %s", self.model_time
                )
        write_set_to_file(
                self.stars,
                p.dir_simulation + 'StarsLatestDump.txt',
                'txt',
                )
        write_set_to_file(
                self.gas_particles,
                p.dir_simulation + 'GasLatestDump.txt',
                'txt',
                )

    def stop(self):
        self.stars_gravity_code.stop()
        self.stars_evolution_code.stop()
        self.sph_code.stop()

    def plot_model(self):
        self.sync_model()
        self.log_output.debug(
                "Making plot at time %s", self.model_time
                )
        # axes        = ["xy", "xz", "yz"]
        # plot_types  = ["all", "stars", "gas"]
        axes = ["xy"]
        plot_types = ["all"]
        for ax in axes:
            for plot_type in plot_types:
                self.p.plot_axes = ax
                tmp = gas_stars_plot(
                        self.plot_n,
                        self.model_time,
                        self.sph_code.particles,
                        self.stars,
                        self.p,
                        plot_type=plot_type,
                        )
        self.plot_n = tmp

    def evolve_model(self, time_end):

        while self.model_time < time_end:
            dt = self.p.timestep
            if dt < self.bridge.timestep:
                dt = self.bridge.timestep
                self.log_output.debug(
                        "Adjusting timestep to %s", dt
                        )

            # FIXME use a dummy evo code instead of this flag
            if p.stars_evo_enabled:
                self.stars_evolution_code.evolve_model(self.time+dt)

            self.bridge.evolve_model(self.time+dt)

            # Are we detecting collisions, and is one taking place now?
            # If so: resolve the collision. Also, don't run analysis etc yet,
            # we're not at the right timestep.
            if (
                    self.collision_detection.is_set() and
                    (self.stars_gravity_code.model_time < self.bridge.time)
                    # model_time+dt)
                    ):
                # self.resolve_encounter()
                self.store_encounter()
                self.stars_gravity_code.evolve_model(self.bridge.time)
            elif self.bridge.time >= self.time+(0.5*dt):
                self.previous_clock = self.current_clock
                self.current_clock = clock.time() | units.s

                self.time = self.bridge.time

                self.log_output.info(
                        "Evolved to time %s in %s: a speedup of %s" % (
                            self.time,
                            self.current_clock - self.previous_clock,
                            dt/(self.current_clock-self.previous_clock),
                            ),
                        )

                if self.time > self.next_log_energy_time:
                    self.log_energy()
                    self.next_log_energy_time += self.p.timestep_log_energy

                if self.time > self.next_log_diagnostic_time:
                    self.log_diagnostic()
                    self.next_log_diagnostic_time += \
                        self.p.timestep_log_diagnostic

                if self.time > self.next_plot_model_time:
                    self.plot_model()
                    self.next_plot_model_time += self.p.timestep_plot_model

                if self.time > self.next_save_backup_time:
                    self.save_to_hdf()
                    self.next_save_backup_time += self.p.timestep_save_backup

        self.sync_model()

    def sync_model(self):
        if self.model_time == self.bridge.time:
            return

        self.channel_from_evolution_to_memory_for_stars.copy_attributes(
                ["mass"])
        self.channel_from_memory_to_gravity_for_stars.copy_attributes(
                ["mass"])
        self.channel_from_gravity_to_memory_for_stars.copy()
        self.channel_from_sphcode_to_memory_for_gas.copy()
        self.model_time = self.bridge.time

        self.gas_particles.collection_attributes.model_time = self.model_time
        self.gas_particles.collection_attributes.sphtime = \
            self.sph_code.model_time
        self.stars.collection_attributes.gravitytime = \
            self.stars_gravity_code.model_time
        self.stars.collection_attributes.evolutiontime = \
            self.stars_evolution_code.model_time
        self.stars.collection_attributes.model_time = self.model_time

    def log_energy(self):
        self.sync_model()
        self.log_output.debug(
                "Writing energy to logfile",
                )
        self.log_energy_file.write(
                "%s %s %s %s %s %s\n" % (
                    self.model_time,
                    self.bridge.kinetic_energy.value_in(self.unit_energy),
                    self.bridge.potential_energy.value_in(self.unit_energy),
                    self.bridge.kick_energy.value_in(self.unit_energy),
                    self.bridge.thermal_energy.value_in(self.unit_energy),
                    self.unit_energy,
                    ),
                )
        self.log_energy_file.flush()

    def log_diagnostic(self):
        self.sync_model()
        incode_stars = self.stars_gravity_code.particles
        incode_gas = self.sph_code.gas_particles
        self.log_output.debug(
                "Writing diagnostics to logfile",
                )
        time = self.model_time
        energy = (
                self.bridge.kinetic_energy +
                self.bridge.potential_energy +
                self.bridge.thermal_energy
                )

        stars_kinetic_energy = (
                self.stars_gravity_code.kinetic_energy
                )
        stars_potential_energy = (
                self.stars_gravity_code.potential_energy
                )
        gas_kinetic_energy = (
                self.sph_code.kinetic_energy
                )
        gas_potential_energy = (
                self.sph_code.potential_energy
                )

        stars_virial_radius = \
            incode_stars.virial_radius()
        gas_virial_radius = \
            incode_gas.virial_radius()

        DensityCenter, CoreRadius, CoreDensity = (
                incode_stars.densitycentre_coreradius_coredens(
                    unit_converter=self.converter,
                    )
                )
        gas_DensityCenter, gas_CoreRadius, gas_CoreDensity = (
                incode_gas.densitycentre_coreradius_coredens(
                    unit_converter=self.converter,
                    )
                )

        LagrangianRadii, MassFraction = (
            incode_stars.LagrangianRadii(
                cm=DensityCenter,
                unit_converter=self.converter,
                )
            )
        gas_LagrangianRadii, MassFraction = (
            incode_gas.LagrangianRadii(
                cm=gas_DensityCenter,
                unit_converter=self.converter,
                )
            )

        stars_bound = (
            incode_stars.bound_subset(
                    unit_converter=self.converter,
                    )
            )
        stars_bound_mass = (
            stars_bound.total_mass()
            )
        gas_bound = (
            incode_gas.bound_subset(
                unit_converter=self.converter,
                )
            )
        gas_bound_mass = (
            gas_bound.total_mass()
            )

        stars_velocity_dispersion = (
            incode_stars.velocity_dispersion()
            )
        gas_velocity_dispersion = (
            incode_gas.velocity_dispersion()
            )

        log = ""
        log += "%s " % time.value_in(self.unit_time)
        log += "%s " % energy.value_in(self.unit_energy)
        log += "%s " % stars_virial_radius.value_in(self.unit_length)
        log += "%s " % CoreRadius.value_in(self.unit_length)
        log += "%s " % CoreDensity.value_in(
                self.unit_mass * self.unit_length**-3)
        log += "%s " % stars_bound_mass.value_in(self.unit_mass)
        log += "%s " % stars_velocity_dispersion.value_in(self.unit_speed)
        log += "%s " % LagrangianRadii[3].value_in(self.unit_length)
        log += "%s " % LagrangianRadii[5].value_in(self.unit_length)
        log += "%s " % LagrangianRadii[8].value_in(self.unit_length)
        log += "%s " % stars_kinetic_energy.value_in(self.unit_energy)
        log += "%s " % stars_potential_energy.value_in(self.unit_energy)
        log += "%s " % gas_virial_radius.value_in(self.unit_length)
        log += "%s " % gas_CoreRadius.value_in(self.unit_length)
        log += "%s " % gas_CoreDensity.value_in(
                self.unit_mass * self.unit_length**-3)
        log += "%s " % gas_bound_mass.value_in(self.unit_mass)
        log += "%s " % gas_velocity_dispersion.value_in(self.unit_speed)
        log += "%s " % gas_LagrangianRadii[3].value_in(self.unit_length)
        log += "%s " % gas_LagrangianRadii[5].value_in(self.unit_length)
        log += "%s " % gas_LagrangianRadii[8].value_in(self.unit_length)
        log += "%s " % gas_kinetic_energy.value_in(self.unit_energy)
        log += "%s " % gas_potential_energy.value_in(self.unit_energy)
        log += "\n"

        self.log_file.write(log)
        self.log_file.flush()

    def resolve_encounter(self):
        self.log_output.debug("Handling encounter")

        self.store_encounter()
        # TODO: use encounters module for this

        particles_in_encounter = self.collision_detection.particles
        for particle in particles_in_encounter:
            print(particle)
        # particles_in_field      = ?
        # particles_in_multiples  = ?

        # x = TryHandleEncounter(
        #         G=constants.G,
        #         kepler_code=new_kepler(),
        #         )

        # x.particles_in_encounter.add_particles(particles_in_encounter)

        # x.particles_in_field.add_particles(particles_in_field)
        # x.particles_in_multiples.add_particles(particles_in_multiples)

        # x.execute()

        self.stars_gravity_code.evolve_model(self.time)

    def store_encounter(
            self,
            setup=False,
            unit_length=units.AU,
            fileformat="hdf5",
            # fileformat="txt",
            # fileformat=p.encounters_file_format,
            ):
        if fileformat == "txt":
            self.store_encounter_txt(
                    setup=setup,
                    unit_length=unit_length,
                    )
        elif fileformat == "hdf5":
            self.store_encounter_hdf5(
                    setup=setup,
                    unit_length=unit_length,
                    )

    def store_encounter_txt(
            self,
            setup=False,
            unit_length=units.AU,
            ):
        if setup:
            self.log_encounters_file = open(
                    self.p.log_encounters,
                    "w",
                    )
            self.log_encounters_file.write(
                    "#time key0 mass0 key1 mass1 dr   dx   dy   dz   dvx  dvy\
                    dvz\n"
                    )
            self.log_encounters_file.write(
                    "#units: %s %s %s %s\n" % (
                        self.unit_time,
                        self.unit_mass,
                        self.unit_length,
                        self.unit_speed,
                        )
                    )
            self.log_encounters_file.flush()
        else:
            self.log_output.debug(
                    "Storing encounter at time %s / %s" % (
                        self.stars_gravity_code.model_time,
                        self.time,
                        )
                    )
            for i in range(len(self.collision_detection.particles(0))):
                p0 = self.collision_detection.particles(0)[i]
                p1 = self.collision_detection.particles(1)[i]
                if p1.mass > p0.mass:
                    temp = p1
                    p1 = p0
                    p0 = temp

                encounter = \
                    "%09.6f %i %05.2f %i %05.2f %07.2f %07.2f %07.2f %07.2f\
                    %06.4f %06.4f %06.4f\n" % (
                        self.stars_gravity_code.model_time.value_in(
                            self.unit_time),
                        p0.key,
                        p0.mass.value_in(self.unit_mass),
                        p1.key,
                        p1.mass.value_in(self.unit_mass),
                        (p0.position - p1.position).length().value_in(
                            unit_length),
                        (p0.x - p1.x).value_in(unit_length),
                        (p0.y - p1.y).value_in(unit_length),
                        (p0.z - p1.z).value_in(unit_length),
                        (p0.vx - p1.vx).value_in(self.unit_speed),
                        (p0.vy - p1.vy).value_in(self.unit_speed),
                        (p0.vz - p1.vz).value_in(self.unit_speed),
                        )
                self.log_encounters_file.write(encounter)
                self.log_encounters_file.flush()

    def store_encounter_hdf5(
            self,
            setup=False,
            unit_length=units.AU,
            ):
        if setup:
            import h5py
            self.log_encounters_file = h5py.File(
                    self.p.log_encounters,
                    "w",
                    )
            dtype = np.dtype(
                    [
                        ('time', np.float32, 1),
                        ('star', np.uint64, 1),
                        ('mass', np.float32, 1),
                        ('position', np.float32, (3,)),
                        ('velocity', np.float32, (3,)),
                        ('other', np.uint64, 1),
                        ])
            self.log_encounters_encounters = \
                self.log_encounters_file.create_dataset(
                    "encounters",
                    (0,),
                    dtype=dtype,
                    compression="gzip",
                    maxshape=(None,),
                    chunks=True,
                    )
            # FIXME: just use SI basic units here, like Amuse hdf files do
            self.log_encounters_encounters.attrs['unit_time'] =\
                str(self.unit_time)
            self.log_encounters_encounters.attrs['unit_mass'] =\
                str(self.unit_mass)
            self.log_encounters_encounters.attrs['unit_length'] =\
                str(self.unit_length)
            self.log_encounters_encounters.attrs['unit_speed'] =\
                str(self.unit_speed)
        else:
            self.log_output.debug(
                    "Storing encounter at time %s / %s" % (
                        self.stars_gravity_code.model_time,
                        self.time,
                        )
                    )
            for i in range(len(self.collision_detection.particles(0))):
                self.log_encounters_encounters.resize(
                        (self.log_encounters_encounters.size+2,)
                        )
                for j in range(2):
                    p = self.collision_detection.particles(j)[i]
                    p2 = self.collision_detection.particles((j+1) % 2)[i]
                    self.log_encounters_encounters[-2+j] = (
                            self.stars_gravity_code.model_time.value_in(
                                self.unit_time),
                            p.key,
                            p.mass.value_in(self.unit_mass),
                            p.position.value_in(unit_length),
                            p.velocity.value_in(self.unit_speed),
                            p2.key,
                            )
            self.log_encounters_file.flush()


if __name__ == "__main__":
    p = Parameters()
    p = new_simulation_argument_parser(p)

    if not os.path.exists(p.dir_simulation):
        os.makedirs(p.dir_simulation)
    if not os.path.exists(p.dir_logs):
        os.makedirs(p.dir_logs)
    if not os.path.exists(p.dir_codelogs):
        os.makedirs(p.dir_codelogs)
    if not os.path.exists(p.dir_plots):
        os.makedirs(p.dir_plots)

    log_output = logging.getLogger(__name__)

    logging.basicConfig(
            filename=p.dir_logs + "EmbeddedCluster.log",
            filemode="w",
            level=logging.DEBUG,
            format="%(asctime)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z"
            )

    set_printing_strategy(
            "custom",
            preferred_units=p.log_units_preferred,
            precision=p.log_units_precision,
            )

    system = EmbeddedCluster(
            p
    )

    system.setup()
    system.evolve_model(p.time_end)

    system.log_diagnostic()
