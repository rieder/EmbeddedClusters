# -*- encoding: utf8 -*-
"""
set up initial conditions for a MYStIX cluster, from real stellar positions and
masses guesses for velocities & velocity dispersions

"""
from __future__ import print_function, division

import os
import time
import numpy as np
import numpy.random as rnd
from numpy import pi, sqrt

from amuse.datamodel import (
        AbstractParticleSet,
        Particles)
from amuse.units import units, nbody_system
from amuse.units.trigo import (
        cos, sin, arccos,
        )
from amuse.io import write_set_to_file
from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.support.console import set_printing_strategy
from amuse.ext.spherical_model import (
        new_uniform_spherical_particle_distribution,
        )

from gasplummer import new_plummer_gas_model
from plummer import new_plummer_model
from plotting_fresco import gas_stars_plot
from parameters import Parameters
from argumentparser import new_IC_argument_parser


def new_xy_for_velocity(p):
    number_of_selected_items = 0
    selected_values_for_x = np.zeros(0)
    selected_values_for_y = np.zeros(0)
    while (number_of_selected_items < p.n_for_velocities):
        x = rnd.uniform(
                0.,
                1.0,
                (p.n_for_velocities - number_of_selected_items),
                )
        y = rnd.uniform(
                0.,
                0.1,
                (p.n_for_velocities - number_of_selected_items),
                )
        g = (x**2) * np.power(
                1.0 - x**2,
                3.5,
                )
        compare = y <= g

        selected_values_for_x = np.concatenate(
                (
                    selected_values_for_x,
                    x.compress(compare),
                    )
                )
        selected_values_for_y = np.concatenate(
                (
                    selected_values_for_x,
                    y.compress(compare),
                    )
                )
        number_of_selected_items = len(selected_values_for_x)
    return selected_values_for_x, selected_values_for_y


def new_velocities_spherical_coordinates(
        p,
        radius,
        ):
    pi2 = pi * 2
    x, y = new_xy_for_velocity(p)
    velocity = x * sqrt(2.0) * np.power(
            1.0 + radius * radius,
            -0.25,)
    theta = arccos(
            rnd.uniform(
                -1.0,
                1.0,
                p.n_for_velocities,
                )
            )
    phi = rnd.uniform(
            0.0,
            pi2,
            p.n_for_velocities,
            )
    return(velocity, theta, phi)


def coordinates_from_spherical(
        p,
        radius,
        theta,
        phi,
        ):
    x = radius * sin(theta) * cos(phi)
    y = radius * sin(theta) * sin(phi)
    z = radius * cos(theta)
    return (x, y, z)


def velocity_dispersion(particles):
    N = len(particles)
    dv = (
            particles.velocity -
            particles.center_of_mass_velocity()
            )
    squarevelocities = dv * dv
    sigma = (
            (
                squarevelocities.sum() /
                (N - 1)
                ).sqrt()
            )
    return sigma


class RealClusterIC (object):

    def __init__(
            self,
            p,
            ):
        if p.seed is not None:
            rnd.seed(p.seed)

        self.virial_ratio = p.stars_virial_ratio

        self.initialize_stars()
        self.initialize_gas()

    AbstractParticleSet.add_global_function_attribute(
            "velocity_dispersion",
            velocity_dispersion,
            )

    def initialize_stars(self):
        # Reads stars' observed positions from datafile
        ra, dec = np.loadtxt(
                p.cluster_stars_file,
                skiprows=1,
                unpack=True,
                )

        nobsstars = len(ra)

        rastars = ra | units.deg
        decstars = dec | units.deg

        # Reads subcluster position
        (
                racentre,
                deccentre,
                rcmajorangle,
                rcminorangle,
                paellipse,
                distance,
                nstarstot,
                ) = np.genfromtxt(
                        p.cluster_file,
                        skip_header=p.cluster_number,
                        skip_footer=(
                            p.cluster_ntotal -
                            p.cluster_number
                            ),
                        unpack=True,
                        )
        # Add units
        racentre = racentre | units.deg
        deccentre = deccentre | units.deg
        distance = distance | units.parsec
        paellipse = paellipse | units.deg

        # rcmajor and rcminor are in arcminutes and are the axes of an ellipse
        # 4 times the size of the fitted core.
        rcmajorangle = 0.25 * rcmajorangle | units.arcmin
        rcminorangle = 0.25 * rcminorangle | units.arcmin

        # Want these to actually be the axes of the core, in parsec.
        rcmajor = rcmajorangle.value_in(units.rad) * distance
        rcminor = rcminorangle.value_in(units.rad) * distance

        p.stars_n = nstarstot

        # core radius is the harmonic mean of rcmajor and rc minor;
        # rscale = virial radius = 16 sqrt(2)/3 pi *rcore for a
        # Plummer sphere.

        p.rscale = (
                2.0 /
                (
                    1.0 / rcmajor +
                    1.0 / rcminor
                    ) *
                16. * sqrt(2) / 3. / pi
                )
        p.variable_refresh()

        # Make converter, based on 1 MSun stars. It's ok.
        self.converter = nbody_system.nbody_to_si(
                p.stars_n | units.MSun,
                p.rscale,
                )

        self.paellipse = paellipse

        self.ellipticity = (rcmajor - rcminor) / rcmajor
        print(self.ellipticity, rcmajor, rcminor)
        self.xellip = p.xellip*self.ellipticity
        self.yellip = p.yellip*self.ellipticity
        self.zellip = p.zellip*self.ellipticity

        self.obsstars = Particles(nobsstars)

        imf_masses = new_salpeter_mass_distribution(
                nobsstars,
                mass_max=p.cluster_observed_mass_max,
                mass_min=p.cluster_observed_mass_min,
                alpha=-2.3
                )
        self.stellar_mass = imf_masses.sum()
        self.obsstars.mass = imf_masses

        # Shift right ascension to center on zero
        dra = rastars - racentre

        # Shift declination to center around zero
        # keep into account that deformation occurs!
        cosdec = cos(deccentre)  # cosine of central declination
        sindec = sin(deccentre)  # sine of central declination

        x = (
                cos(decstars) * sin(dra) /
                (
                    cosdec * cos(decstars) * cos(dra) +
                    sindec * sin(decstars)
                    )
                )
        y = (
                (
                    cosdec * sin(decstars) -
                    sindec * cos(decstars) * cos(dra)
                    ) /
                (
                    sindec * sin(decstars) +
                    cosdec * cos(decstars) * cos(dra)
                    )
                )

        # Spherical trig -- everything is an angle
        self.obsstars.x = x * distance
        self.obsstars.y = y * distance
        self.obsstars.z = rnd.uniform(
                -p.obsstars_zfactor,
                p.obsstars_zfactor,
                nobsstars,
                ) * rcminor
        self.obsstars.vx = rnd.uniform(
                0.,
                1.0,
                nobsstars,
                ) | units.kms
        self.obsstars.vy = rnd.uniform(
                0.,
                1.0,
                nobsstars,
                ) | units.kms
        self.obsstars.vz = rnd.uniform(
                0.,
                1.0,
                nobsstars,
                ) | units.kms
        self.obsstars.radius = p.stars_interaction_radius

        # now do the background of lower-mass stars
        nmorestars = int(p.stars_n - nobsstars)
        imf_masses2 = new_broken_power_law_mass_distribution(
                nmorestars,
                mass_boundaries=p.stars_mass_boundaries,
                mass_max=p.stars_mass_max,
                alphas=p.stars_mass_alphas,
                random=True,
                )
        othermass = imf_masses2.sum()
        self.otherstars = new_plummer_model(
                nmorestars,
                xellip=self.xellip,
                yellip=self.yellip,
                zellip=self.zellip,
                convert_nbody=self.converter,
                )
        self.otherstars.mass = imf_masses2
        self.otherstars.move_to_center()

        # rotate the ellipse so that it lines up with the stars.
        # paellipse is measured from north (+y) axis towards the east
        # (+x axis)

        xnew = (
                self.otherstars.x * cos(self.paellipse)
                + self.otherstars.y * sin(self.paellipse)
                )
        ynew = (
                self.otherstars.y * cos(self.paellipse)
                - self.otherstars.x * sin(self.paellipse)
                )

        self.otherstars.x = xnew
        self.otherstars.y = ynew
        self.otherstars.radius = p.stars_interaction_radius

        # put it all together
        self.stellar_mass += othermass

        self.stars = Particles()
        self.stars.add_particles(self.obsstars)
        self.stars.add_particles(self.otherstars)
        # DO NOT scale things, we have actual positions!
        # self.stars.move_to_center()
        # self.stars.scale_to_standard(
        #         convert_nbody=self.converter,
        #         )
        new_scale_factor = sqrt(self.virial_ratio / 0.5)
        self.stars.velocity *= new_scale_factor

    def initialize_gas(self):
        gas_mass = self.stellar_mass * p.gas_fraction

        p.n_for_velocities = int(gas_mass / p.gas_particle_mass)

        self.gas = new_plummer_gas_model(
                p.n_for_velocities,
                xellip=self.xellip,
                yellip=self.yellip,
                zellip=self.zellip,
                convert_nbody=self.converter,
                )
        self.gas.h_smooth = self.converter.to_si(
                p.gas_smoothing_fraction
                )
        self.gas.u = p.gas_u
        self.gas.mass = p.gas_particle_mass
        self.gas.move_to_center()

        # rotate the ellipse so that it lines up with the stars.
        # paellipse is measured from north (+y) axis towards the east
        # (+x axis)
        xnew = (
                self.gas.x * cos(self.paellipse)
                + self.gas.y * sin(self.paellipse)
                )
        ynew = (
                self.gas.y * cos(self.paellipse)
                - self.gas.x * sin(self.paellipse)
                )

        self.gas.x = xnew
        self.gas.y = ynew

        x = self.gas.x.value_in(
                p.rscale.to_unit()
                )
        y = self.gas.y.value_in(
                p.rscale.to_unit()
                )
        z = self.gas.z.value_in(
                p.rscale.to_unit()
                )
        radius = sqrt(x*x + y*y + z*z)
        vel, theta, phi = new_velocities_spherical_coordinates(
                p,
                radius,
                )
        vx, vy, vz = coordinates_from_spherical(
                p,
                vel,
                theta,
                phi,
                )
        vx1 = nbody_system.speed.new_quantity(vx)
        vy1 = nbody_system.speed.new_quantity(vy)
        vz1 = nbody_system.speed.new_quantity(vz)

        self.gas.vx = self.converter.to_si(vx1)
        self.gas.vy = self.converter.to_si(vy1)
        self.gas.vz = self.converter.to_si(vz1)

        gas_velocity_dispersion = self.gas.velocity_dispersion()
        scale_factor = p.gas_sigma/gas_velocity_dispersion
        self.gas.velocity *= scale_factor

    def write_data(self):
        write_set_to_file(
                self.stars,
                p.dir_initialconditions + p.stars_initial_file,
                'amuse',
                )
        write_set_to_file(
                self.gas,
                p.dir_initialconditions + p.gas_initial_file,
                'amuse',
                )


class AllClusterIC (object):

    def __init__(
            self,
            p,
            stars_sets,
            gas_sets,
            ):
        self.stars_sets = stars_sets
        self.gas_sets = gas_sets

        (
                racentre,
                deccentre,
                rcmajorangle,
                rcminorangle,
                paellipse,
                distance,
                nstarstot,
                ) = np.loadtxt(
                        p.cluster_file,
                        skiprows=1,
                        unpack=True,
                        )
        rcmajorangle = rcmajorangle | units.deg
        rcminorangle = rcminorangle | units.deg
        self.racentre = racentre | units.deg
        self.deccentre = deccentre | units.deg
        distance = distance | units.parsec
        # Verify that all distances are the same.
        if distance.std() == 0.0 | units.parsec:
            self.dcluster = distance[0]
        else:
            print("Error: multiple distances found!")
        self.allstars = Particles()
        self.allgas = Particles()

    def read_combine_clusters(
            self,
            ):

        cosdec = cos(p.cluster_dec_avg)
        sindec = sin(p.cluster_dec_avg)
        dra = self.racentre - p.cluster_ra_avg
        dx = (
                cos(self.deccentre) * sin(dra) /
                (
                    cosdec * cos(self.deccentre) * cos(dra)
                    + sindec * sin(self.deccentre)
                    )
                )
        dy = (
                (
                    cosdec * sin(self.deccentre)
                    - cos(self.deccentre) * sindec * cos(dra)
                    ) /
                (
                    sindec * sin(self.deccentre)
                    + cosdec * cos(self.deccentre) * cos(dra)
                    )
                )
        dx *= self.dcluster
        dy *= self.dcluster

# include a random variation in z as well. p.cluster_zmax gives the extent of
# the z variation, such that the centre of each cluster will be chosen to be
# between -1.0*p.cluster_zmax and p.cluster_zmax (in parsec)

        dz = rnd.uniform(
            -1.0*p.cluster_zmax.value_in(units.parsec),
            p.cluster_zmax.value_in(units.parsec),
            len(self.stars_sets),
            ) | units.parsec

# Give each stellar subcluster a velocity relative to each other. y velocity
# (along the filament) must be pointed towards the centre of the system; x and
# z velocities are scaled by the ellipticity of the filament so that on
# average, they will be smaller. p.cluster_velocity_target is the requested
# total velocity of the star cluster. Currently taken to be the same for each
# cluster. To be more sophisticated, we might want to give the subclusters
# their own velocity dispersion.
# Also need to give each gas subcluster the associated velocity of the stellar
# subcluster.

        dvx = (1.0-p.filament_xellip)*rnd.uniform(
            -1.0,
            1.0,
            len(self.stars_sets),
            ) | units.kms

        dvy = (1.0-p.filament_yellip)*rnd.uniform(
            -1.0,
            1.0,
            len(self.stars_sets),
            ) | units.kms

# make sure the y velocity is going in the right direction -- opposite to the
# position vector
        for i in range(0, len(self.stars_sets)):
            if (
                    np.sign(
                        dvy[i].value_in(units.kms)
                        ) ==
                    np.sign(
                        dy[i].value_in(units.parsec)
                        )
                    ):
                dvy[i] *= -1.0

        dvz = (1.0-p.filament_zellip)*rnd.uniform(
            -1.0,
            1.0,
            len(self.stars_sets),
            ) | units.kms

# normalize so that we get the requested total velocity for each subcluster

        vmag = sqrt(
                dvx.value_in(units.kms)*dvx.value_in(units.kms)
                + dvy.value_in(units.kms)*dvy.value_in(units.kms)
                + dvz.value_in(units.kms)*dvz.value_in(units.kms)
                ) | units.kms
        print(vmag, p.cluster_velocity_target)
        dvx *= p.cluster_velocity_target/vmag
        dvy *= p.cluster_velocity_target/vmag
        dvz *= p.cluster_velocity_target/vmag

        # dv = sqrt(
        #         dvx.value_in(units.kms)*dvx.value_in(units.kms)
        #         + dvy.value_in(units.kms)*dvy.value_in(units.kms)
        #         + dvz.value_in(units.kms)*dvz.value_in(units.kms)
        #         )

        print (dvx, dvy, dvz, dx, dy)

        self.gas_mass = 0.0 | units.MSun

        for i in range(0, len(self.stars_sets)):
            self.stars = self.stars_sets[i]
            self.stars.x += dx[i]
            self.stars.y += dy[i]
            self.stars.z += dz[i]
            self.stars.vx += dvx[i]
            self.stars.vy += dvy[i]
            self.stars.vz += dvz[i]
            self.allstars.add_particles(self.stars)

            self.gas = self.gas_sets[i]
            self.gas.x += dx[i]
            self.gas.y += dy[i]
            self.gas.z += dz[i]
            self.gas.vx += dvx[i]
            self.gas.vy += dvy[i]
            self.gas.vz += dvz[i]
            if i == 0:
                self.gasu = self.gas.u[0]
                self.gash = self.gas.h_smooth[0]
                print(self.gasu, self.gash)

            self.gas_mass += self.gas.mass.sum()
            print(i, self.gas_mass)
            self.allgas.add_particles(self.gas)

        self.nclustered = len(self.allstars.x)
# This 'move to center' causes different offsets between gas & stars. Actually
# don't want to move the stars (because then the unclustered stars are not on
# the same scale/coordinate system as the clustered stars. In fact, don't move
# anything and let's see what happens.
#        self.allstars.move_to_center()
#        self.allgas.move_to_center()

    def add_other_stars(self):
        ra, dec = np.loadtxt(
                p.cluster_stars_u,
                skiprows=1,
                unpack=True,
                )
        rastars = ra | units.deg
        decstars = dec | units.deg
        nstars = len(ra)
        nunclustered = int(nstars)
        unclustered_stars = Particles(nunclustered)
        imf_masses = new_salpeter_mass_distribution(
                nunclustered,
                mass_max=p.cluster_observed_mass_max,
                mass_min=p.cluster_observed_mass_min,
                alpha=-2.3
                )
        unclustered_stars.mass = imf_masses
        unclustered_stellar_mass = imf_masses.sum()

        dra = rastars - p.cluster_ra_avg
        cosdec = cos(p.cluster_dec_avg)
        sindec = sin(p.cluster_dec_avg)

        unclustered_converter = nbody_system.nbody_to_si(
                unclustered_stellar_mass,
                p.rscale,
                )

        x = (
                cos(decstars) * sin(dra) /
                (
                    cosdec * cos(decstars) * cos(dra) +
                    sindec * sin(decstars)
                    )
                )
        y = (
                (
                    cosdec * sin(decstars) -
                    sindec * cos(decstars) * cos(dra)
                    ) /
                (
                    sindec * sin(decstars) +
                    cosdec * cos(decstars) * cos(dra)
                    )
                )

        unclustered_stars.x = x * self.dcluster
        unclustered_stars.y = y * self.dcluster
        unclustered_stars.z = rnd.uniform(
                -1.0*p.unclustered_zdistance.value_in(units.parsec),
                p.unclustered_zdistance.value_in(units.parsec),
                nstars,
                ) | units.parsec

        x = unclustered_stars.x.value_in(
                        p.rscale.to_unit()
                        )
        y = unclustered_stars.y.value_in(
                        p.rscale.to_unit()
                        )
        z = unclustered_stars.z.value_in(
                        p.rscale.to_unit()
                        )

        radius = sqrt(x*x + y*y + z*z)
        p.n_for_velocities = nunclustered

        vel, theta, phi = new_velocities_spherical_coordinates(
                        p,
                        radius,
                        )
        vx, vy, vz = coordinates_from_spherical(
                        p,
                        vel,
                        theta, phi)

        vx1 = nbody_system.speed.new_quantity(vx)
        vy1 = nbody_system.speed.new_quantity(vy)
        vz1 = nbody_system.speed.new_quantity(vz)

        unclustered_stars.vx = unclustered_converter.to_si(vx1)
        unclustered_stars.vy = unclustered_converter.to_si(vy1)
        unclustered_stars.vz = unclustered_converter.to_si(vz1)

        stars_velocity_dispersion = unclustered_stars.velocity_dispersion()
        scale_factor = p.stars_sigma/stars_velocity_dispersion
        unclustered_stars.velocity *= scale_factor

        unclustered_stars.radius = p.stars_interaction_radius

        self.allstars.add_particles(unclustered_stars)

        ra, dec = np.loadtxt(
                p.cluster_stars_x,
                skiprows=1,
                unpack=True,
                )
        rastars = ra | units.deg
        decstars = dec | units.deg
        nstars = len(ra)
        nunknown = int(nstars)
        unknown_stars = Particles(nunknown)
        imf_masses = new_salpeter_mass_distribution(
                nunknown,
                mass_max=p.cluster_observed_mass_max,
                mass_min=p.cluster_observed_mass_min,
                alpha=-2.3
                )
        unknown_stars.mass = imf_masses
        unknown_stellar_mass = imf_masses.sum()

        dra = rastars - p.cluster_ra_avg
        cosdec = cos(p.cluster_dec_avg)
        sindec = sin(p.cluster_dec_avg)

        unknown_converter = nbody_system.nbody_to_si(
                        unknown_stellar_mass,
                        p.rscale,
                        )

        x = (
                cos(decstars) * sin(dra)
                / (
                    cosdec * cos(decstars) * cos(dra)
                    + sindec * sin(decstars)
                    )
                )
        y = (
                (
                    cosdec * sin(decstars)
                    - sindec * cos(decstars) * cos(dra)
                    )
                / (
                    sindec * sin(decstars)
                    + cosdec * cos(decstars) * cos(dra)
                    )
                )

        unknown_stars.x = x * self.dcluster
        unknown_stars.y = y * self.dcluster
        unknown_stars.z = rnd.uniform(
                -1.0*p.unclustered_zdistance.value_in(units.parsec),
                p.unclustered_zdistance.value_in(units.parsec),
                nunknown,
                ) | units.parsec
        x = unknown_stars.x.value_in(
                        p.rscale.to_unit()
                        )
        y = unknown_stars.y.value_in(
                        p.rscale.to_unit()
                        )
        z = unknown_stars.z.value_in(
                        p.rscale.to_unit()
                        )

        radius = sqrt(x*x + y*y + z*z)
        p.n_for_velocities = nunknown

        vel, theta, phi = new_velocities_spherical_coordinates(
                        p,
                        radius,
                        )
        vx, vy, vz = coordinates_from_spherical(
                        p,
                        vel,
                        theta, phi)

        vx2 = nbody_system.speed.new_quantity(vx)
        vy2 = nbody_system.speed.new_quantity(vy)
        vz2 = nbody_system.speed.new_quantity(vz)

        unknown_stars.vx = unknown_converter.to_si(vx2)
        unknown_stars.vy = unknown_converter.to_si(vy2)
        unknown_stars.vz = unknown_converter.to_si(vz2)

        stars_velocity_dispersion = unknown_stars.velocity_dispersion()
        scale_factor = p.stars_sigma/stars_velocity_dispersion
        unknown_stars.velocity *= scale_factor

        unknown_stars.radius = p.stars_interaction_radius

        self.allstars.add_particles(unknown_stars)

# Number of low-mass unclustered/unknown stars calculated from extending the
# IMF (assuming broken power law with alpha = 1.3 between 0.2 and 0.5 Msun; and
# alpha = 2.3 between 0.5 and 100 Msun)
# notes from 13 June 2016

        nlow = int(
                nunclustered * (
                    5.838  # FIXME: Origin of this number?
                    - p.cluster_observed_mass_min.value_in(units.MSun)**(-1.3)
                    )
                / (
                    p.cluster_observed_mass_min.value_in(units.MSun)**(-1.3)
                    - p.cluster_observed_mass_max.value_in(units.MSun)**(-1.3)
                    )
                )
        lowmass_stars = Particles(nlow)
        imf_masses = new_broken_power_law_mass_distribution(
                nlow,
                mass_boundaries=p.stars_mass_boundaries,
                mass_max=p.stars_mass_max,
                alphas=p.stars_mass_alphas,
                random=True,
                )
        othermass = imf_masses.sum()

        lowmass_converter = nbody_system.nbody_to_si(
                        othermass,
                        p.rscale,
                        )

        lowmass_stars = new_uniform_spherical_particle_distribution(
                nlow,
                p.unclustered_zdistance,
                othermass,
                type="random",
                )

        lowmass_stars.mass = imf_masses

        x = lowmass_stars.x.value_in(
                p.rscale.to_unit()
                )
        y = lowmass_stars.y.value_in(
                p.rscale.to_unit()
                )
        z = lowmass_stars.z.value_in(
                p.rscale.to_unit()
                )
        radius = sqrt(x*x + y*y + z*z)
        p.n_for_velocities = nlow

        vel, theta, phi = new_velocities_spherical_coordinates(
                p,
                radius,
                )
        vx, vy, vz = coordinates_from_spherical(
                p,
                vel,
                theta, phi,)

        vx3 = nbody_system.speed.new_quantity(vx)
        vy3 = nbody_system.speed.new_quantity(vy)
        vz3 = nbody_system.speed.new_quantity(vz)

        lowmass_stars.vx = lowmass_converter.to_si(vx3)
        lowmass_stars.vy = lowmass_converter.to_si(vy3)
        lowmass_stars.vz = lowmass_converter.to_si(vz3)

        stars_velocity_dispersion = lowmass_stars.velocity_dispersion()
        scale_factor = p.stars_sigma/stars_velocity_dispersion
        lowmass_stars.velocity *= scale_factor

        lowmass_stars.radius = p.stars_interaction_radius
        self.allstars.add_particles(lowmass_stars)

    def add_filament(self):

        p.gas_filament_n = int(p.gas_filament_mass / p.gas_particle_mass)

        gas_converter = nbody_system.nbody_to_si(
                self.gas_mass,
                p.filament_rscale,
                )

        filament = new_plummer_gas_model(
                p.gas_filament_n,
                xellip=p.filament_xellip,
                yellip=p.filament_yellip,
                zellip=p.filament_zellip,
                convert_nbody=gas_converter
                )
        filament.h_smooth = self.gash
        filament.mass = p.gas_particle_mass
        filament.u = self.gasu

        paellipse = p.filament_paellipse

        xnew = (
                filament.x * cos(paellipse)
                + filament.y * sin(paellipse)
                )
        ynew = (
                filament.y * cos(paellipse)
                - filament.x * sin(paellipse)
                )
        filament.x = xnew
        filament.y = ynew

        x = filament.x.value_in(
                p.rscale.to_unit()
                )
        y = filament.y.value_in(
                p.rscale.to_unit()
                )
        z = filament.z.value_in(
                p.rscale.to_unit()
                )
        radius = sqrt(x*x + y*y + z*z)
        p.n_for_velocities = p.gas_filament_n

        vel, theta, phi = new_velocities_spherical_coordinates(
                p,
                radius,
                )
        vx, vy, vz = coordinates_from_spherical(
                p,
                vel,
                theta, phi,)

        vx1 = nbody_system.speed.new_quantity(vx)
        vy1 = nbody_system.speed.new_quantity(vy)
        vz1 = nbody_system.speed.new_quantity(vz)

        filament.vx = gas_converter.to_si(vx1)
        filament.vy = gas_converter.to_si(vy1)
        filament.vz = gas_converter.to_si(vz1)

        gas_velocity_dispersion = filament.velocity_dispersion()
        scale_factor = p.gas_sigma/gas_velocity_dispersion
        filament.velocity *= scale_factor

        self.allgas.add_particles(filament)

    def write_all(self):
        write_set_to_file(
                self.allstars,
                p.dir_initialconditions + p.stars_initial_file,
                'amuse',
                )
        write_set_to_file(
                self.allgas,
                p.dir_initialconditions + p.gas_initial_file,
                'amuse',
                )


if __name__ in("__main__"):
    p = Parameters()
    p = new_IC_argument_parser(p)

    set_printing_strategy(
            "custom",
            preferred_units=p.log_units_preferred,
            precision=p.log_units_precision,
            )

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    stars_sets = []
    gas_sets = []

    for n in range(1, p.cluster_ntotal+1):
        p.cluster_number = n
        print("Creating initial conditions for %s cluster %s...  " % (
                p.cluster_datarelease,
                p.cluster_label[p.cluster_number-1],
                ), end=' ')
        p.cluster_stars_file = p.dir_input + "/%s%sstars.txt" % (
                p.cluster_datarelease,
                p.cluster_label[p.cluster_number-1],
                )

        p.dir_simulation = "./Results-%s/Run-%s/" % (
                timestamp,
                p.cluster_label[p.cluster_number-1],
                )

        p.variable_refresh()

        if not os.path.exists(p.dir_simulation):
            os.makedirs(p.dir_simulation)
            os.makedirs(p.dir_initialconditions)
            os.makedirs(p.dir_plots)

        system = RealClusterIC(
                p,
                )

        system.write_data()
        p.plot_axes_x = "x"
        p.plot_axes_y = "y"
        i = gas_stars_plot(
            0,
            0.0 | units.Myr,
            system.gas,
            system.stars,
            p
            )
        p.plot_axes_x = "x"
        p.plot_axes_y = "z"
        j = gas_stars_plot(
            0,
            0.0 | units.Myr,
            system.gas,
            system.stars,
            p
            )
        p.plot_axes_x = "y"
        p.plot_axes_y = "z"
        k = gas_stars_plot(
            0,
            0.0 | units.Myr,
            system.gas,
            system.stars,
            p
            )
        print("Done!")

        stars_sets.append(system.stars)
        gas_sets.append(system.gas)

    # From here, we build the full set
    p.dir_simulation = "./Results-%s/All/" % (
            timestamp,
            )

    p.variable_refresh()

    if not os.path.exists(p.dir_simulation):
        os.makedirs(p.dir_simulation)
        os.makedirs(p.dir_initialconditions)
        os.makedirs(p.dir_plots)

    full_system = AllClusterIC(
            p,
            stars_sets,
            gas_sets,
    )

    full_system.read_combine_clusters()
    full_system.add_other_stars()
    full_system.add_filament()
    full_system.write_all()

    p.plot_minx = -4.0 | units.parsec
    p.plot_maxx = 4.0 | units.parsec
    p.plot_miny = -4.0 | units.parsec
    p.plot_maxy = 4.0 | units.parsec

    p.plot_axes_x = "x"
    p.plot_axes_y = "y"
    i = gas_stars_plot(
        0,
        0.0 | units.Myr,
        full_system.allgas,
        full_system.allstars,
        p
        )

    p.plot_axes_x = "x"
    p.plot_axes_y = "z"
    j = gas_stars_plot(
        0,
        0.0 | units.Myr,
        full_system.allgas,
        full_system.allstars,
        p
        )

    p.plot_axes_x = "y"
    p.plot_axes_y = "z"
    k = gas_stars_plot(
        0,
        0.0 | units.Myr,
        full_system.allgas,
        full_system.allstars,
        p
        )
