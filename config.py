import typing


class RunConfig(typing.NamedTuple):
    save_fig: int = 0
    save_pdf: int = 0


class ModelConfig(typing.NamedTuple):
    architecture: int = None
    lr_gen: float = 0.001
    lr_disc: float = 0.001
    optim_gen: str = "Adam"
    optim_disc: str = "Adam"
    dec_gen: float = 0
    dec_disc: float = 0
    random_seed: int = 1992
    activation: str = "elu"
    z_input_size: int = 1


class TrainingConfig(typing.NamedTuple):
    n_epochs: int = 20
    batch_size: int = 128
    n_samples: int = 100  # the size of sampling times for which samples drawn from p(y|x)


class DatasetConfig(typing.NamedTuple):
    scenario: str = "linear"
    n_instance: int = 1000


class Config(typing.NamedTuple):
    run: RunConfig
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig


class MLPConfig(typing.NamedTuple):
    input_size: int = 15
    output_size: int = 1
    lr: float = 0.001
    decay: float = 0.1
    dropout_rate: float = 0.1
    batch_size: int = 100
    epochs: int = 2000
    activation: str = "relu"
    optimizer: str = "Adam"
