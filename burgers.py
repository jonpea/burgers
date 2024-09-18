"""
Solution of the inviscid Burgers equation on a cyclic domain.

Acknowledgements:
* Thanks to Matthew Clay for the clear comments on the minmod slope limiter:
  https://github.com/mpclay/MUSCL1D/blob/master/src/Flux.F90
* Thanks to Michael Zingale for the Riemann problem formulae in flux_update():
  https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html
"""

import dataclasses
from typing import Callable, TypeAlias
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from numpy.typing import NDArray
import scipy
import scipy.interpolate
import tqdm

Array: TypeAlias = NDArray[np.float64]
OdeSolution: TypeAlias = scipy.integrate._ivp.common.OdeSolution


@dataclasses.dataclass(frozen=True, slots=True)
class Burgers:
    """
    A callable representing the ODE right-hand side of the
    inviscid Burgers' equation on a cyclic/periodic uniform grid.
    """

    x: Array  # points at control volume centers
    dx: float  # grid spacing

    @staticmethod
    def flux(u: Array) -> Array:
        """Burgers flux F(u) in d/dt u + d/dx F(u) = 0."""
        return 0.5 * u**2

    @staticmethod
    def reconstruct_at_faces(u: Array) -> tuple[Array, Array]:
        """
        Reconstruct values at the left- and right side of each
        cell interface, given values at cell centers.
        """
        # Arrays:
        #       u:     [ u0 , u1 , u2 , u3 ]     @ cell centers, size N
        #  u_wrap: [u3 | u0 | u1 | u2 | u3 | u0] @ cell centers, size N + 2
        #  u_left:   [u30, u01, u12, u23, u30]   @ cell facets, size N + 1
        u_wrap: Array = np.pad(u, 1, mode="wrap")
        # Apply minimum modulus (minmod) slope limiter to spatial differences
        du_left: Array = u - u_wrap[:-2]
        du_right: Array = u_wrap[2:] - u
        minmod: Array = np.minimum(np.abs(du_left), np.abs(du_right))
        du: Array = np.where(du_left * du_right > 0.0, minmod * np.sign(du_left), 0.0)
        du_wrap: Array = np.pad(du, 1, mode="wrap")
        # Reconstruction of values beside each cell face (relative to face, not center)
        u_left: Array = u_wrap[:-1] + 0.5 * du_wrap[:-1]
        u_right: Array = u_wrap[1:] - 0.5 * du_wrap[1:]
        # Simpler initial implementation:
        # Piecewise constant reconstruction of values beside each face
        # u_left: Array = u_wrap[:-1]
        # u_right: Array = u_wrap[1:]
        return u_left, u_right

    @staticmethod
    def evaluate_flux(u_left: Array, u_right: Array) -> Array:
        """Approximate flux at each cell interface."""

        # Adapted from formulae in flux_update() by Michael Zingale:
        # https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html
        u_mean: Array = 0.5 * (u_left + u_right)
        u_mean_abs: Array = np.abs(u_mean)
        # For use where u_left exceeds u_right (shock/compression)
        u_shock: Array = np.where(
            u_mean_abs < 1e-12 * np.max(u_mean_abs),
            0.0,
            np.where(
                u_mean > 0.0,
                u_left,  # for "u_left > u_right" where shock interface moves right
                u_right,  # for "u_left > u_right" where shock interface moves left
            ),
        )
        # For use where u_right exceeds u_left (rarefaction/decompression)
        u_rarefaction: Array = np.where(
            u_left >= 0.0,
            u_left,  # for "0.0 <= u_left <= u_right"
            np.where(
                u_right <= 0.0,
                u_right,  # for "u_left <= u_right <= 0"
                0.0,
            ),
        )

        u_interface: Array = np.where(u_left > u_right, u_shock, u_rarefaction)
        return Burgers.flux(u_interface)

    def __call__(
        self, t: np.float64, u: Array, progress: Callable[[np.float64], None]
    ) -> Array:
        """Evaluates ODE right-hand side function."""
        progress(t)
        u_left, u_right = Burgers.reconstruct_at_faces(u)
        assert u_left.size == u_right.size == u.size + 1
        flux: Array = Burgers.evaluate_flux(u_left, u_right)
        assert flux.size == u.size + 1
        # Evaluates (d/dt u) in
        #             d/dt u + d/dx F(u) == 0
        # from
        #   int(d/dt u, dx) + net_outflux == 0
        # on each control volume, where
        #   net_outflux := [n*F(u)]_left + [n*F(u)]_right
        #         and n := unit outward normal
        net_outflux: Array = -1.0 * flux[:-1] + 1.0 * flux[1:]
        assert net_outflux.size == u.size
        return net_outflux / -self.dx


def pyramid(x_: Array) -> Array:
    """Unit pyramid on [-1, 1]."""
    u = np.zeros_like(x_)
    i_left = np.logical_and(-1.0 <= x_, x_ < 0.0)
    i_right = np.logical_and(0.0 <= x_, x_ <= 1.0)
    u[i_left] = x_[i_left] + 1.0
    u[i_right] = 1.0 - x_[i_right]
    return u


# Parameters
def propagate(
    rhs,
    t_initial: float,
    t_final: float,
    u_initial: Array,
    **options,
) -> OdeSolution:
    """Integrates the semi-discrete inviscid Burgers system through [t_initial, t_final]."""
    with tqdm.tqdm(total=t_final - t_initial) as progress_bar:

        def update_progress_bar(t: np.float64) -> None:
            progress_bar.update(t - progress_bar.n)

        return scipy.integrate.solve_ivp(
            fun=rhs,
            args=(update_progress_bar,),
            y0=u_initial,
            t_span=(t_initial, t_final),
            **options,
        )


def visualize(
    results: OdeSolution,
    t_initial: float,
    t_final: float,
    num_frames: int,
    filename: str,
    suptitle: str,
) -> None:

    # Actual step sizes for CFL evaluation
    t_all = results.t[:-1]
    u_all = results.y[:, :-1]
    dt_all = np.diff(results.t, n=1)
    cfl_all = u_all * dt_all / dx

    # Visualization
    fig, ax_u = plt.subplots()
    fig.suptitle(suptitle.format(t_initial))
    ax_u.set_xlabel("x")
    ax_cfl = ax_u.twinx()

    (plot_u,) = ax_u.plot(x, u_initial, "-", label="u", color="blue")
    ax_u.axis("equal")
    ax_u.set_ylabel("u")

    (plot_cfl,) = ax_cfl.plot(x, cfl_all[:, 0], "--", label="CFL", color="green")
    ax_cfl.set_ylabel("CFL")
    ax_cfl.set_yscale("log")

    plots = [plot_u, plot_cfl]
    ax_u.legend(plots, [p.get_label() for p in plots], loc="lower center")

    plt.show(block=False)

    animation_fps = (num_frames - 1) / (t_final - t_initial)

    # Uniform temporal grid for animation frames
    t_frame = np.linspace(t_initial, t_final, num_frames)
    u_frame = results.sol(t_frame)
    cfl_frame = scipy.interpolate.interp1d(
        t_all, cfl_all, axis=1, fill_value="extrapolate"
    )(t_frame)

    with tqdm.tqdm(total=num_frames) as progress_bar:

        def update_plots(frame_index: int) -> None:
            """Updates plot for next animation frame."""
            progress_bar.update()
            fig.suptitle(suptitle.format(t_frame[frame_index]))
            plot_u.set_ydata(u_frame[:, frame_index])
            plot_cfl.set_ydata(cfl_frame[:, frame_index])
            ax_u.autoscale()
            ax_cfl.autoscale()
            return (plot_u, plot_cfl)

        animation_ = animation.FuncAnimation(
            fig=fig,
            func=update_plots,
            frames=range(num_frames),
            interval=1000.0 / animation_fps,  # milliseconds
        )

    animation_.save(filename=filename, writer="pillow", fps=animation_fps)


if __name__ == "__main__":

    t_initial = 0.0
    t_final = 5.0
    x, dx = np.linspace(start=-1.5, stop=+1.5, num=2000, retstep=True)
    u_initial = pyramid(x)
    num_frames = 100
    filename = "burgers.gif"
    suptitle = "Inviscid Burgers w/ Periodic BC: t = {:.2f}"

    print("Propagating...")
    results: OdeSolution = propagate(
        rhs=Burgers(x, dx),
        t_initial=t_initial,
        t_final=t_final,
        u_initial=u_initial,
        dense_output=True,
        # Default explicit marching is stable
        # max_step=0.5 * dx / np.max(u_initial),
        # method="Radau",
        # atol=1e-6,
        # rtol=1e-3,
    )

    print("Processing frames...")
    visualize(
        results=results,
        t_initial=t_initial,
        t_final=t_final,
        num_frames=num_frames,
        filename=filename,
        suptitle=suptitle,
    )
    print(f"Wrote '{filename}'")
