"""Paramètres d'un run : fichier TOML + surcharges `section.cle=valeur`."""

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class RunConfig:
    name: str = "default"
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"


@dataclass
class ProblemConfig:
    dim: int = 1
    a1: float = 1.0
    a2: float = 0.01
    v0: float = 0.5
    source: float = 1.0
    band: list[float] = field(default_factory=lambda: [0.2, 0.3])
    objective: str = "band"          # "band" : int_bande u^2 ; "target" : int (u - u*)^2 (1D)
    target_amp: float = 0.35         # cible : rho* = v0 + target_amp sin(2 pi target_periods x)
    target_periods: int = 1


@dataclass
class QuadratureConfig:
    n_elements: int = 100
    n_gauss: int = 10


@dataclass
class ModelConfig:
    width: int = 20
    depth: int = 3


@dataclass
class LossConfig:
    lam: float = 1e-2
    beta: float = 1e2
    mu: float = 1e2
    # "strong" : résidu fort -div(A grad u) - f ; "mixed" : flux sigma (réseau en plus),
    # équilibre div sigma + f et relation de comportement int |sigma - A grad u|^2 / A (sans dérivée de A)
    form: str = "strong"


def _geometric(a: float, b: float, n: int) -> list[float]:
    """n valeurs géométriques de a à b inclus (n = 1 : [a])."""
    return [a] if n == 1 else [a * (b / a) ** (k / (n - 1)) for k in range(n)]


@dataclass
class ContinuationConfig:
    """Calendrier des paliers (ε, q).

    Mode couplé (q_values vide) : n_stages paliers, ε géométrique de eps0 à eps_min ; q vaut q0
    jusqu'au palier q_start, puis monte linéairement jusqu'à q_max au dernier palier.

    Mode en phases (q_values non vide) :
      1. topologie : n_stages paliers à q = q0, ε géométrique de eps0 à eps_topo ;
      2. binarisation : un palier par valeur de q_values, ε = eps_topo ;
      3. netteté : n_sharp_stages paliers à q = q_values[-1], ε géométrique de eps_topo à eps_min.
    """
    eps0: float = 0.1
    eps_min: float = 4e-3
    n_stages: int = 10
    q0: float = 0.0
    q_max: float = 0.0
    q_start: int = 0
    eps_topo: float = 0.0
    q_values: list[float] = field(default_factory=list)
    n_sharp_stages: int = 0

    @property
    def phased(self) -> bool:
        return len(self.q_values) > 0

    @property
    def q_final(self) -> float:
        return self.q_values[-1] if self.phased else self.q_max

    def schedule(self) -> list[tuple[float, float]]:
        """(ε, q) de chaque palier."""
        if not self.phased:
            eps = _geometric(self.eps0, self.eps_min, self.n_stages)
            return [(e, self.q(stage)) for stage, e in enumerate(eps)]
        if not self.eps_min <= self.eps_topo <= self.eps0:
            raise ValueError(f"mode en phases : eps_topo = {self.eps_topo:g} doit être entre eps_min et eps0")
        topo = [(eps, self.q0) for eps in _geometric(self.eps0, self.eps_topo, self.n_stages)]
        binarize = [(self.eps_topo, float(q)) for q in self.q_values]
        sharp = [(eps, self.q_final) for eps in _geometric(self.eps_topo, self.eps_min, self.n_sharp_stages + 1)[1:]]
        return topo + binarize + sharp

    def describe(self) -> list[str]:
        """Calendrier en clair, pour la figure."""
        if not self.phased:
            return [f"q (RAMP) : {self.q0:g} -> {self.q_max:g} (monte après le palier {self.q_start + 1})",
                    f"ε : {self.eps0:g} -> {self.eps_min:g} ({self.n_stages} paliers)"]
        q_text = ", ".join(f"{q:g}" for q in self.q_values)
        return [f"q (RAMP) : {self.q0:g}, puis {q_text} à ε = {self.eps_topo:g}",
                f"ε : {self.eps0:g} -> {self.eps_topo:g} ({self.n_stages} paliers), "
                f"puis -> {self.eps_min:g} ({self.n_sharp_stages} paliers)"]

    def q(self, stage: int) -> float:
        """Mode couplé : q0 jusqu'au palier q_start inclus, puis linéaire jusqu'à q_max au dernier palier."""
        last = self.n_stages - 1
        if stage <= self.q_start or last <= self.q_start:
            return self.q0
        return self.q0 + (self.q_max - self.q0) * (stage - self.q_start) / (last - self.q_start)


@dataclass
class TrainConfig:
    adam_iters: int = 2000
    adam_lr: float = 1e-3
    lbfgs_iters: int = 1000
    lbfgs_history: int = 50
    # nucléation : avant L-BFGS, Adam sur des points tirés au hasard à chaque pas (Monte-Carlo).
    # Le bruit d'intégration donne au gradient des composantes oscillantes qui font sortir un gris
    # lisse du point selle du double puits (sinon la descente, déterministe, y reste).
    nucleation_stages: list[int] = field(default_factory=list)   # paliers, comptés depuis 1
    nucleation_iters: int = 2000
    nucleation_lr: float = 1e-3


@dataclass
class OutputConfig:
    runs_dir: str = "runs"
    plot_every: int = 100
    log_every: int = 100
    n_exact: int = 4000
    n_grid: int = 200


@dataclass
class ReferenceConfig:
    """Référence 2D : optimisation par densité (RAMP, filtre, adjoint) sur la grille volumes finis."""
    n_cells: int = 100
    filter_radius: float = 0.03
    q_penal: float = 300.0
    n_q_steps: int = 6
    iters_per_step: int = 300


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    quadrature: QuadratureConfig = field(default_factory=QuadratureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)


def _parse_value(text: str):
    """Valeur au format TOML (1e-3, true, [0.2, 0.3], "nom") ; texte brut sinon."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _build(cls, data: dict, where: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"clés inconnues dans [{where}] : {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        sub = known[name].type
        if isinstance(value, dict) and is_dataclass(sub):
            kwargs[name] = _build(sub, value, name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: list[str] = ()) -> Config:
    data: dict = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep or "." not in key:
            raise ValueError(f"surcharge attendue sous la forme section.cle=valeur : {item!r}")
        section, name = key.strip().split(".", 1)
        data.setdefault(section, {})[name] = _parse_value(text.strip())
    return _build(Config, data, "racine")


def to_toml(cfg: Config) -> str:
    """Config complète (valeurs effectives) au format TOML, pour l'archive du run."""
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, list | tuple):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return repr(v)

    lines = []
    for section, values in asdict(cfg).items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {fmt(v)}" for k, v in values.items()]
        lines.append("")
    return "\n".join(lines)
