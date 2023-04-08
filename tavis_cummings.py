from dataclasses import dataclass, field, fields
from operator import mul
from functools import reduce

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import qutip
from qutip import destroy, tensor, qeye, sigmax, mesolve, steadystate, basis, spectrum, expect
from tqdm import tqdm

@dataclass
class Emitter:
    g: float
    gamma: float
    frequency: float


@dataclass
class Cavity:
    num_emitters: int
    cavity_freq: float
    emitter_freq: list[float] = field(metadata={"length_checked": True})
    g: list[float] = field(metadata={"length_checked": True})
    gamma: list[float] = field(metadata={"length_checked": True})
    kappa: float
    num_photons: int = 1

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
    
    @property
    def emitters(self):
        yield from [
            Emitter(g=self.g[i], gamma=self.gamma[i], frequency=self.emitter_freq[i])
            for i in range(self.num_emitters)
        ]

    def a(self):
        """Returns the photon annilihation operator for the cavity"""
        return tensor(destroy(self.num_photons + 1), *([qeye(2)] * self.num_emitters))

    def sigma(self, emitter: int):
        """Returns the emitter lowering operator for the ith emitter (zero-indexed)."""
        l = [qeye(self.num_photons + 1)]
        l += [destroy(2) if emitter == i else qeye(2) for i in range(self.num_emitters)]
        return tensor(l)
    
    def sigma_x(self, emitter: int):
        l = [qeye(self.num_photons + 1)]
        l += [sigmax() if emitter == i else qeye(2) for i in range(self.num_emitters)]
        return tensor(l)

    def g_zero_op(self, order: int):
        a = self.a()
        operators = [a.dag()] * order + [a] * order
        return reduce(mul, operators)

    def collapse_ops(self, pump_power: float | None = None):
        cavity_relaxation = np.sqrt(self.kappa) * self.a()
        emitter_relaxation = [
            np.sqrt(emitter.gamma) * self.sigma(i) for i, emitter in enumerate(self.emitters)
        ]
        ops = [cavity_relaxation] + emitter_relaxation
        if pump_power is not None:
            return ops + [pump_power * self.a().dag()]
        return ops
    
    def steady_state(self, pump_frequency: float | None = None, pump_power: float | None = None):
        H = self.hamiltonian(pump_frequency, pump_power)
        return steadystate(H, self.collapse_ops(pump_power))

    def spectrum(self, frequencies: list, pump_power: float | None = None):
        pump_power = pump_power or self.kappa / 50
        H = self.hamiltonian()
        a = self.a()
        return spectrum(H, frequencies, self.collapse_ops(pump_power), a.dag(), a)

    def excitation_likelihoods(self):
        number_ops = [self.a().dag() * self.a()]
        for i in range(self.num_emitters):
            sigma = self.sigma(i)
            number_ops.append(sigma.dag() * sigma)
        return number_ops
    
    def hamiltonian(self, pump_freq: float | None = None, pump_rate: float = 1):
        """Returns the full qutip hamiltonian"""
        a = self.a()
        if pump_freq is not None:
            H = (self.cavity_freq - pump_freq) * a.dag() * a
            for i, emitter in enumerate(self.emitters):
                s = self.sigma(i)
                H += 0.5 * (emitter.frequency - pump_freq) * s.dag() * s + emitter.g * (a.dag() * s + a * s.dag())
                H += pump_rate * (a + a.dag())
            return H

        else:
            H = self.cavity_freq * a.dag() * a
            for i, emitter in enumerate(self.emitters):
                s = self.sigma(i)
                H += emitter.frequency * s.dag() * s 
                H += emitter.g * (a.dag() * s + a * s.dag())
            return H

    def cavity_state(self, num_photons: int):
        if num_photons > self.num_photons:
            raise ValueError(f"Cavity can only accept up to {self.num_photons}.")
        return tensor(basis(self.num_photons + 1, num_photons), *([basis(2, 0)] * self.num_emitters))

    def excited_emitter_state(self, excited_emitter_index: int):
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
            states.append(phase * self.excited_emitter_state(i))
        return sum(states)

    def superradiant_state(self):
        states = []
        for i in range(self.num_emitters):
            states.append(self.excited_emitter_state(i))
        return sum(states) / np.sqrt(self.num_emitters)

    def mesolve(self, initial_state, times, measurements=None, pump_freq=None, pump_rate=1):
        """Evolve `initial_state` through `times` according to the Lindbladian."""
        H = self.hamiltonian(pump_freq, pump_rate)
        c_ops = self.collapse_ops()
        measurements = measurements or self.excitation_likelihoods()
        return mesolve(H, initial_state, times, c_ops, measurements)


def plot_populations(result: qutip.solver.Result, title: str = ''):
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

    fig.show()


def zero_time_correlation_under_pump(
    cavity: Cavity, orders: int | list[int], drive_frequencies, pump_power=None
) -> go.Figure:
    """Make a plot of zero-time correlations at the given orders and pump power."""
    c_ops = cavity.collapse_ops()
    a = cavity.a()
    pump_power = pump_power or cavity.kappa / 50
    orders = [orders] if isinstance(orders, int) else orders

    correlations_expect = []
    for wd in tqdm(drive_frequencies):
        H = cavity.hamiltonian(pump_freq=wd, pump_rate=pump_power)
        ss = steadystate(H, c_ops)

        correlation_ops = [cavity.g_zero_op(order=o) for o in orders]

        correlations = [expect(ss, op) for op in correlation_ops]
        n = expect(ss, a.dag() * a)

        normalized_correlation = [corr / n ** o for corr, o in zip(correlations, orders)]
        correlations_expect.append(np.real(normalized_correlation))

    correlations_expect = np.vstack(correlations_expect).T

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.update_yaxes(title="Zero-time correlations", row=1, col=1)
    fig.update_yaxes(title="Output spectrum", row=2, col=1)
    fig.update_xaxes(title=r"$\omega - \omega_C$", row=1, col=2)

    for o, corr in zip(orders, correlations_expect):
        fig.add_trace(
            go.Scatter(x=drive_frequencies - cavity.cavity_freq, y=corr, name=f"g{o}(0)"),
            row=1,
            col=1,
        )
    
    fig.add_trace(
        go.Scatter(x=drive_frequencies - cavity.cavity_freq, y=cavity.spectrum(drive_frequencies)),
        row=2,
        col=1,
    )

    return fig

