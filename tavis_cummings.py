from collections.abc import Sequence, Callable
from dataclasses import dataclass, field, fields
from functools import reduce
from operator import mul
from itertools import product
from dataclasses import replace

import numpy as np
import qutip
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from qutip import (
    basis,
    destroy,
    expect,
    mesolve,
    qeye,
    sigmax,
    spectrum,
    steadystate,
    tensor,
    sigmaz,
)
from qutip.parallel import parallel_map
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from IPython.display import clear_output


@dataclass
class Emitter:
    g: float
    gamma: float
    dephasing: float  # equivalent to 1/T2, maybe there's a nicer name...
    frequency: float | Callable

    _emitter_idx: int
    _cavity_num_photons: int
    _cavity_num_emitters: int

    @property
    def sigma(self):
        """Returns the lowering operator for this emitter."""
        emitter_ops = [
            destroy(2) if self._emitter_idx == i else qeye(2)
            for i in range(self._cavity_num_emitters)
        ]
        return tensor(qeye(self._cavity_num_photons + 1), *emitter_ops)

    @property
    def sigma_z(self):
        """Returns the Z operator for this emitter."""
        emitter_ops = [
            sigmaz() if self._emitter_idx == i else qeye(2)
            for i in range(self._cavity_num_emitters)
        ]
        return tensor(qeye(self._cavity_num_photons + 1), *emitter_ops)


@dataclass
class Cavity:
    num_emitters: int
    cavity_freq: float
    emitter_freq: list[float] = field(metadata={"length_checked": True})
    g: list[float] = field(metadata={"length_checked": True})
    gamma: list[float] = field(metadata={"length_checked": True}, default=0)
    dephasing: list[float] = field(metadata={"length_checked": True}, default=0)
    kappa: float = 0
    num_photons: int = 1

    _time_dependent: bool = False

    def __post_init__(self):
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if field_.metadata.get("length_checked"):
                if isinstance(value, list):
                    if len(value) != self.num_emitters:
                        raise ValueError(
                            f"Provided {len(value)} values for {field_.name!r},"
                            f" but there are {self.num_emitters} emitters!"
                        )
                else:
                    setattr(self, field_.name, [value] * self.num_emitters)

        if any(isinstance(f, Callable) for f in self.emitter_freq):
            self._time_dependent = True

    @property
    def emitters(self):
        kwargs = dict(_cavity_num_photons=self.num_photons, _cavity_num_emitters=self.num_emitters)
        return [
            Emitter(
                g=self.g[i],
                gamma=self.gamma[i],
                dephasing=self.dephasing[i],
                frequency=self.emitter_freq[i],
                _emitter_idx=i,
                **kwargs,
            )
            for i in range(self.num_emitters)
        ]

    @property
    def a(self):
        """Returns the photon annilihation operator for the cavity"""
        return tensor(destroy(self.num_photons + 1), *([qeye(2)] * self.num_emitters))

    def sigma(self, emitter: int):
        """Returns the emitter lowering operator for the ith emitter (zero-indexed)."""
        return self.emitters[emitter].sigma

    def sigma_x(self, emitter: int):
        l = [qeye(self.num_photons + 1)]
        l += [sigmax() if emitter == i else qeye(2) for i in range(self.num_emitters)]
        return tensor(l)

    def g_zero_op(self, order: int):
        operators = [self.a.dag()] * order + [self.a] * order
        return reduce(mul, operators)

    def collapse_ops(self, pump_power: float | None = None):
        cavity_relaxation = np.sqrt(self.kappa) * self.a
        emitter_relaxation = [np.sqrt(emitter.gamma) * emitter.sigma for emitter in self.emitters]
        emitter_dephasing = [np.sqrt(em.dephasing) * em.sigma_z for em in self.emitters]
        ops = [cavity_relaxation] + emitter_relaxation + emitter_dephasing
        if pump_power is not None:
            return ops + [pump_power * self.a.dag()]
        return ops

    def steady_state(self, pump_frequency: float | None = None, pump_power: float | None = None):
        H = self.hamiltonian(pump_frequency, pump_power)
        return steadystate(H, self.collapse_ops(pump_power))

    def spectrum(self, frequencies: list, pump_power: float | None = None):
        pump_power = pump_power or self.kappa / 50
        H = self.hamiltonian()
        return spectrum(H, frequencies, self.collapse_ops(pump_power), self.a.dag(), self.a)

    def excitation_likelihoods(self):
        return [self.a.dag() * self.a] + [em.sigma.dag() * em.sigma for em in self.emitters]

    def hamiltonian(self, pump_freq: float = 0, pump_rate: float = 0):
        """Returns the full qutip hamiltonian"""
        pump_rate = self.kappa / 50 if pump_freq != 0 and pump_rate == 0 else pump_rate
        a = self.a 
        
        if self._time_dependent:
            H0 = (self.cavity_freq - pump_freq) * a.dag() * a + pump_rate * (a + a.dag())
            time_deps = []
            for emitter in self.emitters:
                s = emitter.sigma
                time_deps.append([s.dag() * s, emitter.frequency])
                H0 += emitter.g * (a.dag() * s + a * s.dag())
                H0 += - pump_freq * s.dag() * s
            return [H0, *time_deps]


        H = (self.cavity_freq - pump_freq) * a.dag() * a + pump_rate * (a + a.dag())
        for emitter in self.emitters:
            s = emitter.sigma
            H += (emitter.frequency - pump_freq) * s.dag() * s 
            H += emitter.g * (a.dag() * s + a * s.dag())
        return H

    def cavity_state(self, num_photons: int):
        if num_photons > self.num_photons:
            raise ValueError(f"Cavity can only accept up to {self.num_photons}.")
        return tensor(
            basis(self.num_photons + 1, num_photons),
            *([basis(2, 0)] * self.num_emitters),
        )

    def emitter_state(self, excited_emitter_index: int):
        """Create a state where one emitter has 100% of the excitation."""

        if excited_emitter_index < 0:
            raise ValueError(f"Invalid emitter index {excited_emitter_index}!")
        if excited_emitter_index > self.num_emitters:
            raise ValueError(f"Cavity only has {self.num_emitters} emitters!")

        component_bases = [
            basis(2, 1) if i == excited_emitter_index else basis(2, 0)
            for i in range(self.num_emitters)
        ]
        return tensor(basis(self.num_photons + 1, 0), *component_bases)

    def ground_state(self):
        emitters = [basis(2, 0) for i in range(self.num_emitters)]
        return tensor(basis(self.num_photons + 1, 0), *emitters)

    def subradiant_state(self):
        states = []
        for i in range(self.num_emitters):
            phase = np.exp(-1j * i * 2 * np.pi / self.num_emitters) / np.sqrt(self.num_emitters)
            states.append(phase * self.emitter_state(i))
        return sum(states)

    def superradiant_state(self):
        states = []
        for i in range(self.num_emitters):
            states.append(self.emitter_state(i))
        return sum(states) / np.sqrt(self.num_emitters)

    def mesolve(self, initial_state, times, measurements=None, pump_freq=0, pump_rate=0, options=None):
        """Evolve `initial_state` through `times` according to the Lindbladian."""
        H = self.hamiltonian(pump_freq, pump_rate)
        c_ops = self.collapse_ops()
        measurements = measurements or self.excitation_likelihoods()
        return mesolve(H, initial_state, times, c_ops, measurements, options=options)

    def product_state(self, cavity: int = 0, **kwargs):
        if cavity > self.num_photons:
            raise ValueError(f"Cavity can only accept up to {self.num_photons}.")
        if max(kwargs.values()) > 1:
            raise ValueError("Emitters can only have states 0 or 1.")

        emitter_states = [basis(2, kwargs.get(f"emitter_{i}", 0)) for i in range(self.num_emitters)]
        return tensor(basis(self.num_photons + 1, cavity), *emitter_states)

    def effective_hamiltonian(self):
        emitter_terms = [
            emitter.gamma / 2 * emitter.sigma.dag() * emitter.sigma for emitter in self.emitters
        ]
        H_eff = (
            self.hamiltonian()
            - 1j * self.kappa / 2 * self.a.dag() * self.a
            - 1j * sum(emitter_terms)
        )
        return H_eff

    def emitter_quality(self, state):
        return sum(
            np.abs(self.emitter_state(i).overlap(state)) ** 2 for i in range(self.num_emitters)
        )

    def cavity_quality(self, state):
        return sum(
            np.abs(self.cavity_state(i).overlap(state)) ** 2 for i in range(self.num_photons + 1)
        )


def plot_populations(result: qutip.solver.Result, title: str = ""):
    """Make a plot of emitter and cavity populations over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Cavity",
            x=result.times,
            y=result.expect[0],
            mode="lines",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Population",
        margin=dict(l=20, r=20, t=35, b=20),
    )

    for i, data in enumerate(result.expect[1:]):
        fig.add_trace(
            go.Scatter(
                name=f"Atom {i} excited state",
                x=result.times,
                y=data,
                mode="lines",
            )
        )
    return fig


def g_correlations(
    drive_freq: float, cavity: Cavity, orders: list[int], pump_power: float
) -> float:
    H = cavity.hamiltonian(pump_freq=drive_freq, pump_rate=pump_power)
    c_ops = cavity.collapse_ops()
    ss = steadystate(H, c_ops)

    correlation_ops = [cavity.g_zero_op(order=o) for o in orders]

    correlations = [expect(ss, op) for op in correlation_ops]
    n = expect(ss, cavity.a.dag() * cavity.a)

    normalized_correlation = [corr / n**o for corr, o in zip(correlations, orders)]
    return np.real(normalized_correlation)


def g_correlation_swept_pump(
    cavity: Cavity,
    orders: int | list[int],
    drive_frequencies: Sequence[float],
    pump_power=None,
    parallel=False,
) -> go.Figure:
    """Make a plot of zero-time correlations at the given orders and pump power."""
    pump_power = pump_power or cavity.kappa / 50
    orders = [orders] if isinstance(orders, int) else orders

    if parallel:
        # OpenBLAS (used by qutip) ordinarily uses multithreading. This interferes with multicore
        # scheduling, preventing the use of `parallel_map`, so let's temporarily disable that.
        with threadpool_limits(limits=1, user_api="blas"):
            correlations_expect = parallel_map(
                g_correlations,
                drive_frequencies,
                task_kwargs=dict(cavity=cavity, orders=orders, pump_power=pump_power),
                progress_bar=True,
            )
    else:
        # At this point I'm not sure if it's safe for all users to enable parallelism by default
        correlations_expect = [
            g_correlations(wd, cavity, orders, pump_power) for wd in tqdm(drive_frequencies)
        ]

    correlations_expect = np.vstack(correlations_expect).T

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.update_yaxes(title="Zero-time correlations", row=1, col=1)
    fig.update_yaxes(title="Output spectrum", row=2, col=1)
    fig.update_xaxes(title="ω", row=2, col=1)

    # Plot the g-function correlations in the top panel
    for o, corr in zip(orders, correlations_expect):
        trace = go.Scatter(x=drive_frequencies, y=corr, name=f"g{o}(0)")
        fig.add_trace(trace, row=1, col=1)

    # Plot the emission spectrum of the cavity in the lower panel
    spectrum_trace = go.Scatter(
        x=drive_frequencies,
        y=cavity.spectrum(drive_frequencies, pump_power),
        name="Transmission",
    )
    fig.add_trace(spectrum_trace, row=2, col=1)

    return fig


def _g_correlation_with_emitter_detuning(delta_e_and_wd, cavity: Cavity, order, pump_power):
    """This is meant for use within `g_correlation_vs_emitter_detuning()`."""
    ΔE, wd = delta_e_and_wd

    emitter_freqs = cavity.emitter_freq
    emitter_freqs[1] = emitter_freqs[0] + ΔE
    cavity = replace(cavity, emitter_freq=emitter_freqs)

    H = cavity.hamiltonian(pump_freq=wd, pump_rate=cavity.kappa / 50)
    c_ops = cavity.collapse_ops()

    ss = steadystate(H, c_ops)
    g2_op = cavity.g_zero_op(order=2)

    G2 = expect(ss, g2_op)
    n = expect(ss, cavity.a.dag() * cavity.a)

    g2 = G2 / n**2

    return np.real(g2)


def _map(
    func,
    values,
    *,
    args=(),
    kwargs=None,
    parallel: bool = False,
    parallel_progress_bar: bool = True,
):
    """Helper to allow switching between sequential and parallel mapping."""
    if parallel:
        # OpenBLAS (used by qutip) ordinarily uses multithreading. This interferes with multicore
        # scheduling, preventing the use of `parallel_map`, so let's temporarily disable that.
        with threadpool_limits(limits=1, user_api="blas"):
            return parallel_map(
                func,
                values,
                task_args=args,
                task_kwargs=kwargs,
                progress_bar=parallel_progress_bar,
            )
    else:
        # At this point I'm not sure if it's safe for all users to enable parallelism by default
        return [func(value, *args, **kwargs) for value in tqdm(values)]


def g_correlation_vs_emitter_detuning(
    cavity: Cavity,
    order: int,
    second_emitter_detunings: Sequence[float],
    drive_frequencies: Sequence[float],
    pump_power=None,
    parallel=False,
    alternate_sweep: dict[str, Sequence] | None = None,
) -> go.Figure:
    """Generate a heatmap of g-correlation at `order` vs emitter-emitter detuning for 2 emitters."""
    pump_power = pump_power or cavity.kappa / 50
    points = list(product(second_emitter_detunings, drive_frequencies))
    alternate_sweep = alternate_sweep or {"num_photons": [cavity.num_photons]}  # dummy sweep
    color_scale = [[0.0, "red"], [0.5, "white"], [1.0, "blue"]]
    fig = go.Figure()

    if len(alternate_sweep) > 1:
        raise ValueError("More than one alternate sweep value not currently supported.")

    property_name = list(alternate_sweep.keys())[0]
    property_values = list(alternate_sweep.values())[0]

    for value in tqdm(property_values):
        cavity = replace(cavity, **{property_name: value})

        correlations = _map(
            _g_correlation_with_emitter_detuning,
            points,
            kwargs=dict(cavity=cavity, order=order, pump_power=pump_power),
            parallel=parallel,
        )

        g2s = np.reshape(correlations, (len(second_emitter_detunings), len(drive_frequencies))).T
        g2s = np.clip(g2s, a_min=None, a_max=2)

        fig.add_trace(
            go.Heatmap(
                x=second_emitter_detunings,
                y=drive_frequencies,
                z=g2s,
                colorscale=color_scale,
                visible=False,
            )
        )
        clear_output()

    fig.data[0].visible = True

    if len(property_values) > 1:
        steps = []
        for i, val in enumerate(property_values):
            step = dict(
                method="update",
                label=f"{property_name}: {val:.2f}",
                args=[{"visible": [False] * len(fig.data)}],
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)

        slider = go.layout.Slider(steps=steps, currentvalue={"prefix": f"{property_name}: "})
        fig.update_layout(sliders=[slider])

    fig.update_layout(title=f"g{order}(0)", width=800, height=800)
    fig.update_xaxes(title="ΔE")
    fig.update_yaxes(title="ω")
    return fig
