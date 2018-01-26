import logging

from amuse.units import (
        # nbody_system,
        units,
        # constants,
        )

from amuse.couple.bridge import CalculateFieldForCodesUsingReinitialize

from amuse.community.hermite0.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.fi.interface import Fi
from amuse.community.gadget2.interface import Gadget2
from amuse.community.fastkick.interface import FastKick
from amuse.community.seba.interface import SeBa

from dummy_codes import NoStellarEvolution

logger = logging.getLogger(__name__)


def new_code_stellarevolution(
        p,
        ):
    if p.codes_stars_evolution == "SeBa":
        se = new_code_stellarevolution_seba(p)
    elif p.codes_stars_evolution is None:
        se = new_code_stellarevolution_none()
    else:
        logger.error(
                "No Stellar Evolution code %s known" % (
                    p.codes_stars_evolution),
                )
        exit()
    logger.info(
            "Stellar Evolution code %s added" % (
                p.codes_stars_evolution,
                ),
            )
    return se


def new_code_stellarevolution_seba(
        p,
        ):
    if p.codes_stars_evolution_log:
        result = SeBa(
                redirection="file",
                redirect_file=(
                    p.dir_codelogs + "/se_code.log"
                    )
                )
    else:
        result = SeBa()
    result.parameters.metallicity = p.stars_metallicity
    return result


def new_code_stellarevolution_none():
    result = NoStellarEvolution()
    return result


def new_code_sph(
        converter,
        gas_epsilon,
        p,
        ):
    if p.codes_sph == "gadget":
        sph = new_code_sph_gadget(
                converter,
                gas_epsilon,
                p,
                )
    elif p.codes_sph == "fi":
        sph = new_code_sph_fi(
                converter,
                gas_epsilon,
                p,
                )
    else:
        logger.error(
                "No SPH code %s known" % (
                    p.codes_sph),
                )
        exit()
    logger.info(
            "SPH code %s added" % (
                p.codes_sph,
                ),
            )
    return sph


def new_code_sph_gadget(
        converter,
        gas_epsilon,
        p,
        ):
    result = Gadget2(
            converter,
            number_of_workers=p.codes_sph_nworkers,
            redirection="file",
            redirect_file=(
                p.dir_codelogs + "/sph_code.log"
                )
            )
    result.parameters.time_limit_cpu = 1.0 | units.yr
    result.parameters.time_max = 1.1 * p.time_end
    result.gas_epsilon = gas_epsilon
    result.gamma = p.gas_gamma
    if p.gas_isothermal:
        if not result.parameters.isothermal_flag:
            logger.warning(
                    "Gadget2 not compiled for isothermal gas"
                    )
    else:
        if result.parameters.isothermal_flag:
            logger.warning(
                    "Gadget2 compiled for isothermal gas"
                    )
    return result


def new_code_sph_fi(
        converter,
        epsilon,
        p,
        ):
    result = Fi(
            converter,
            mode="openmp",
            redirection="file",
            redirect_file=(
                p.dir_codelogs + "/gas_code.log"
                )
            )
    result.parameters.self_gravity_flag = True
    result.parameters.use_hydro_flag = True
    result.parameters.integrate_entropy_flag = False
    result.parameters.isothermal_flag = p.gas_isothermal
    result.parameters.gamma = p.gas_gamma
    result.parameters.timestep = p.timestep_interaction/2.
    result.parameters.periodic_box_size = p.gas_box_size
    result.parameters.verbosity = 1
    return result


def new_code_stars_gravity(
        converter,
        epsilon,
        p,
        ):
    if p.codes_stars_gravity == "ph4":
        star = new_code_stars_gravity_ph4(
                converter,
                epsilon,
                p,
                )
    elif p.codes_stars_gravity == "hermite":
        star = new_code_stars_gravity_hermite(
                converter,
                epsilon,
                p,
                )
    else:
        logger.error(
                "No gravity code %s known" % (
                    p.codes_stars_gravity,
                    ),
                )
        exit()
    logger.info(
            "Stellar Gravity code %s added" % (
                p.codes_stars_gravity,
                ),
            )
    return star


def new_code_stars_gravity_ph4(
        converter,
        epsilon,
        p,
        ):
    result = ph4(
            converter,
            mode="gpu" if p.codes_stars_gravity_gpu else "cpu",
            redirection="file",
            redirect_file=(
                p.dir_codelogs + "/stars_gravity_code.log"
                )
            )
    result.initialize_code()
    result.parameters.epsilon_squared = epsilon**2
    return result


def new_code_stars_gravity_hermite(
        converter,
        epsilon,
        p,
        ):
    result = Hermite(
            converter,
            redirection="file",
            redirect_file=(
                p.dir_codelogs + "/stars_gravity_code.log"
                )
            )
    result.parameters.epsilon_squared = epsilon**2
    result.parameters.end_time_accuracy_factor = 0
    return result


def new_code_field_gravity(
        converter,
        epsilon,
        p,
        ):
    # if p.codes_bridge_feedback == "FastKick":
    result = FastKick(
            converter,
            redirection="file",
            redirect_file=(
                p.dir_codelogs + "/field_gravity_code.log"
                ),
            mode="gpu" if p.codes_bridge_feedback_gpu else "normal",
            )
    result.parameters.epsilon_squared = (epsilon)**2
    return result


def code_stars_field(
        stars_gravity_code,
        field_gravity_code,
        ):
    result = CalculateFieldForCodesUsingReinitialize(
            field_gravity_code,
            [stars_gravity_code],
            required_attributes=[
                'mass', 'radius',
                'x', 'y', 'z',
                'vx', 'vy', 'vz',
                ]
            )
    return result


def code_sph_field(
        sph_code,
        field_gravity_code,
        ):
    result = CalculateFieldForCodesUsingReinitialize(
            field_gravity_code,
            [sph_code],
            required_attributes=[
                'mass', 'u',
                'x', 'y', 'z',
                'vx', 'vy', 'vz',
                ]
            )
    return result
