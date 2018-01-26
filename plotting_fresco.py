# Drop-in replacement for Fresco

from fresco.fresco.core import initialise_image as init_fresco_image
from fresco.fresco.core import make_image as make_fresco_image
from fresco.fresco.core import evolve_to_age
try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def gas_stars_plot(
        i,
        time,
        gas,
        stars,
        p,
        plot_type="all",
        ):
    if not HAS_MATPLOTLIB:
        return -1
    image_size = [1024, 1024]
    se_code = "SeBa"
    age = time
    evolve_to_age(stars, age, se=se_code)
    try:
        fig = p.plot_figure
        fig = init_fresco_image(fig)
    except:
        fig = init_fresco_image(
                dpi=p.plot_dpi,
                image_size=image_size,
                image_width=(p.plot_maxx-p.plot_minx),
                plot_axes=False,
                )
        p.plot_figure = fig
    ax = fig.get_axes()[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    plot_name = 'fresco-%05i.png' % i

    try:
        vmax = p.vmax
    except:
        vmax = None
    image, vmax = make_fresco_image(
            stars,
            gas,
            mode=["stars", "gas"],
            image_size=image_size,
            vmax=vmax,
            return_vmax=True,
            sourcebands="ubvri",
            percentile=0.9995,
            # psf_type="gaussian",
            # psf_sigma=1,
            )
    p.vmax = vmax
    ax.imshow(
            image,
            origin='lower',
            extent=[
                xmin,
                xmax,
                ymin,
                ymax,
                ],
            )
    plt.savefig(
            p.dir_plots + plot_name,
            dpi=p.plot_dpi,
            )
    return i+1
