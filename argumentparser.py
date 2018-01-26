import argparse

# from amuse.units import units, nbody_system, constants


def new_IC_argument_parser(p):
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--ngas',
            dest='gas_n',
            type=int,
            default=p.gas_n,
            help="Number of gas particles (default: %i)" % (
                p.gas_n,
                ),
            )
    parser.add_argument(
            '--temperature',
            dest='gas_temperature',
            type=float,
            default=p.gas_temperature.value_in(
                p.gas_temperature.unit
                ),
            help="Gas temperature (default: %s)" % (
                p.gas_temperature,
                ),
            )
    parser.add_argument(
            '--virial',
            dest='stars_virial_ratio',
            type=float,
            default=p.stars_virial_ratio,
            help="Virial ratio used for the stars (default: %s)" % (
                p.stars_virial_ratio,
                ),
            )
    args = parser.parse_args()

    # FIXME: we should somehow just loop over all args and set the
    # corresponding parameter
    p.gas_n = args.gas_n
    p.gas_temperature = args.gas_temperature | p.gas_temperature.unit
    p.stars_virial_ratio = args.stars_virial_ratio
    p.variable_refresh()
    return p


def new_simulation_argument_parser(p):
    parser = argparse.ArgumentParser()
    if p.codes_collision_detection:
        parser.add_argument(
                '--nocollision',
                dest='codes_collision_detection',
                action='store_false',
                default=p.codes_collision_detection,
                help="Explicitly set collision detection to %s (default: %s)"
                     % (
                        False,
                        p.codes_collision_detection,
                     ),
                )
    else:
        parser.add_argument(
                '--collision',
                dest='codes_collision_detection',
                action='store_true',
                default=p.codes_collision_detection,
                help="Explicitly set collision detection to %s (default: %s)"
                     % (
                         True,
                         p.codes_collision_detection,
                     ),
                )
    parser.add_argument(
            '--collisionradius',
            dest='stars_interaction_radius',
            type=float,
            default=p.stars_interaction_radius.value_in(
                p.stars_interaction_radius.unit
                ),
            help="Set radius within which encounters are recorded\
            (default: %s)" % (
                p.stars_interaction_radius,
                ),
            )
    parser.add_argument(
            '--gravity',
            dest='codes_stars_gravity',
            type=str,
            default=p.codes_stars_gravity,
            help="Set gravity code used for stars (default: %s)" % (
                p.codes_stars_gravity,
                ),
            )
    parser.add_argument(
            '--sph',
            dest='codes_sph',
            type=str,
            default=p.codes_sph,
            help="Set sph code used for gas (default: %s)" % (
                p.codes_sph,
                ),
            )
    parser.add_argument(
            '--sphworkers',
            dest='codes_sph_nworkers',
            type=int,
            default=p.codes_sph_nworkers,
            help="Set nr of sph code workers (default: %i)" % (
                p.codes_sph_nworkers,
                ),
            )
    parser.add_argument(
            '--epsilonstars',
            dest='stars_smoothing_fraction',
            type=float,
            default=p.stars_smoothing_fraction.value_in(
                p.stars_smoothing_fraction.unit),
            help="Softening length used for stars (default: %s)" % (
                p.stars_smoothing_fraction,
                ),
            )
    parser.add_argument(
            '--epsilongas',
            dest='gas_smoothing_fraction',
            type=float,
            default=p.gas_smoothing_fraction.value_in(
                p.gas_smoothing_fraction.unit
                ),
            help="Softening length used for gas (default: %s)" % (
                p.gas_smoothing_fraction,
                ),
            )
    parser.add_argument(
            '--dir',
            dest='dir_simulation',
            type=str,
            default=p.dir_simulation,
            help="Simulation directory (default: %s)" % (
                p.dir_simulation,
                ),
            )
    parser.add_argument(
            '--stars',
            dest='stars_initial_file',
            type=str,
            default=p.stars_initial_file,
            help="Initial conditions file for stars (default: %s)" % (
                p.stars_initial_file,
                ),
            )
    parser.add_argument(
            '--gas',
            dest='gas_initial_file',
            type=str,
            default=p.gas_initial_file,
            help="Initial conditions file for gas (default: %s)" % (
                p.gas_initial_file,
                ),
            )
    args = parser.parse_args()

    # FIXME: we should somehow just loop over all args and set the
    # corresponding parameter
    p.codes_collision_detection = args.codes_collision_detection
    p.stars_interaction_radius = args.stars_interaction_radius\
        | p.stars_interaction_radius.unit
    p.codes_stars_gravity = args.codes_stars_gravity
    p.codes_sph = args.codes_sph
    p.codes_sph_nworkers = args.codes_sph_nworkers
    p.stars_smoothing_fraction = args.stars_smoothing_fraction\
        | p.stars_smoothing_fraction.unit
    p.gas_smoothing_fraction = args.gas_smoothing_fraction\
        | p.gas_smoothing_fraction.unit
    p.stars_initial_file = args.stars_initial_file
    p.gas_initial_file = args.gas_initial_file
    p.dir_simulation = args.dir_simulation

    p.variable_refresh()
    return p
