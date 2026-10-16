"""Paramètres d'un run : fichier TOML + surcharges `section.cle=valeur`."""

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class RunConfig:
    name: str = "default"
    seed: int = 0
    device: str = "cpu"


@dataclass
class ProblemConfig:
    a1: float = 1.0
    a2: float = 0.01
    v0: float = 0.5
    source: float = 1.0
    band: list[float] = field(default_factory=lambda: [0.2, 0.3])


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


@dataclass
class ContinuationConfig:
    eps0: float = 0.1
    eps_min: float = 4e-3
    n_stages: int = 10

    def eps(self, stage: int) -> float:
        if self.n_stages == 1:
            return self.eps0
        return self.eps0 * (self.eps_min / self.eps0) ** (stage / (self.n_stages - 1))


@dataclass
class TrainConfig:
    adam_iters: int = 2000
    adam_lr: float = 1e-3
    lbfgs_iters: int = 1000
    lbfgs_history: int = 50


@dataclass
class OutputConfig:
    runs_dir: str = "runs"
    plot_every: int = 100
    log_every: int = 100
    n_exact: int = 4000


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
