# -*- encoding: utf-8

from amuse.units import units, nbody_system, constants
from amuse.support.console import set_printing_strategy


class Parameters(object):

    def __init__(self):
        # General #####################
        self.seed = 42

        # Converter ###################
        self.scale_M = 1000 | units.MSun
        self.scale_R = 1.0 | units.parsec
        self.converter = nbody_system.nbody_to_si(
                self.scale_M,
                self.scale_R,
                )

        # Directories #################
        self.dir_input = "./mystix_data/"
        self.dir_simulation = "./"

        # Cluster #####################
        # These values need to be set for each specific cluster simulation
        self.cluster_datarelease = "DR21"
        self.cluster_stars_file = ""
        self.cluster_number = 0
        self.cluster_ntotal = 9
        self.cluster_label = (
                [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    ]
                )
        self.cluster_ra_avg = 309.752083 | units.deg
        self.cluster_dec_avg = 42.345092 | units.deg
        self.cluster_observed_mass_min = 0.83 | units.MSun
        self.cluster_observed_mass_max = 3.5 | units.MSun

        # Codes and code settings #####
        self.codes_stars_gravity = "ph4"
        self.codes_stars_gravity_gpu = True
        self.codes_stars_gravity_nworkers = 1
        self.codes_bridge_feedback = "FastKick"
        self.codes_bridge_feedback_gpu = True
        self.codes_stars_evolution = "SeBa"
        self.codes_stars_evolution_log = False
        self.codes_sph = "gadget"
        self.codes_sph_nworkers = 1
        self.codes_collision_detection = False

        # Stars #######################
        self.stars_initial_file = "StarInitialConditions.hdf5"

        # stars_n used only for making ICs, for running the simulation stars_n
        # is read from stars_initial_file
        self.stars_n = 0

        self.stars_smoothing_fraction = 200 | units.AU
        self.stars_evo_enabled = True
        self.stars_metallicity = 0.02
        self.stars_interaction_radius = 100 | units.AU
        self.stars_virial_ratio = 0.04

        # used for the IMF
        self.stars_mass_boundaries = [0.2, 0.5, 100.0] | units.MSun
        self.stars_mass_max = (
                self.cluster_observed_mass_min  # 0.83 | units.MSun
                )
        self.stars_mass_alphas = [-1.3, -2.3]

        # Gas #########################
        self.gas_initial_file = "GasInitialConditions.hdf5"

        # use only part of the gas if gas_subsample > 1
        self.gas_subsample = 1

        # Values used only for making ICs, for running the simulation these are
        # read from gas_initial_file
        self.gas_n = 10000
        self.gas_filament_n = 30000
        self.gas_particle_mass = 0.05 | units.MSun
        self.gas_temperature = 10 | units.K
        self.gas_sigma = 2.0 | units.kms
        self.gas_fraction = 1.0
        self.gas_expand = 2.0
        #
        self.gas_mean_molecular_weight = (
                constants.proton_mass
                * 4.0 / (1.0 + 3.0 * 0.76)
                # FIXME: add origin for these numbers
                )

        self.gas_smoothing_fraction = 200 | units.AU
        self.gas_isothermal = False
        self.gas_gamma = 1.0
        # self.gas_cutoff_radius = 100 | units.parsec
        self.gas_box_size = 500 | units.parsec

        # Timesteps ###################
        self.timesteps = None
        # timestep_interaction: used for Bridge and other codes
        self.timestep_interaction = 0.01 | nbody_system.time
        # timestep: loop timesteps (interactions / energy output)
        # Note: this will always be larger than or the same as
        # timestep_interaction

        self.timestep = \
            self.converter.to_si(self.timestep_interaction)
        self.timestep_log_energy = 1*self.timestep
        self.timestep_plot_model = 20*self.timestep
        self.timestep_log_diagnostic = 40*self.timestep
        self.timestep_save_backup = 200*self.timestep

        # Simulation duration #########
        self.time_start = 0.0 | units.Myr
        self.time_end = 10.0 | units.Myr

        # Plotting ####################
        self.plot_bins = 256
        self.plot_dpi = 300
        self.plot_figsize = (6, 5)
        self.plot_minx = -5.0 | units.parsec
        self.plot_maxx = 5.0 | units.parsec
        self.plot_miny = -5.0 | units.parsec
        self.plot_maxy = 5.0 | units.parsec
        self.plot_bgcolor = "black"
        self.plot_axes_x = "x"
        self.plot_axes_y = "y"
        self.plot_colormap = "pault_rainbow"
        self.plot_vmin = 1.2
        self.plot_vmax = 4.0

        # Logging #####################
        self.log_units_preferred = [
                units.MSun,
                units.parsec,
                units.Myr,
                units.kms,
                ]
        self.log_units_precision = 5
        set_printing_strategy(
                "custom",
                preferred_units=self.log_units_preferred,
                precision=self.log_units_precision,
                )

        self.variable_refresh()

    def variable_refresh(self):
        # (Re-)initialise parameters that depend on another parameter
        # This function should be called after any of these parameters
        # changes!

        # Directories #################
        self.dir_initialconditions = self.dir_simulation + "ICs/"
        self.dir_logs = self.dir_simulation + "logs/"
        self.dir_codelogs = self.dir_simulation + "code_logs/"
        self.dir_plots = self.dir_simulation + "plots/"

        # Cluster #####################
        self.cluster_file = self.dir_input + "/DR21all.txt"
        self.cluster_stars_u = self.dir_input + "/DR21Ustars.txt"
        self.cluster_stars_x = self.dir_input + "/DR21Xstars.txt"

        # Gas #########################
        self.gas_u = (
                3.0 * constants.kB * self.gas_temperature
                / (2.0 * self.gas_mean_molecular_weight)
                )

        # Logging #####################
        self.log_output = self.dir_logs + "output.log"
        self.log_energy = self.dir_logs + "energy.log"
        self.log_encounters = self.dir_logs + "encounters.hdf5"
