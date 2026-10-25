
"""Modern TensorFlow definition of the Chapter 4 DeepLog baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeeplogModelConfig:
    """Architecture settings matching the supplied DeepLog defaults."""

    vocabulary_size: int
    sequence_length: int = 10
    hidden_layers: int = 2
    hidden_units: int = 64
    embedding_dim: int = 64
    activation: str = "relu"
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 4:
            raise ValueError("vocabulary_size must include learned events after four special tokens")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        if self.hidden_layers <= 0:
            raise ValueError("hidden_layers must be greater than zero")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be greater than zero")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")


def lstm_unit_plan(config: DeeplogModelConfig) -> tuple[int, ...]:
    """Return the legacy DeepLog LSTM widths."""

    return (config.hidden_units,) * config.hidden_layers


def _tensorflow() -> Any:
    try:
        import tensorflow
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to build DeepLog. Run this step in the configured "
            "Colab or Micromamba environment."
        ) from error
    return tensorflow


def build_deeplog(config: DeeplogModelConfig) -> Any:
    """Build and compile the next-event LSTM classifier used as the baseline."""

    tensorflow = _tensorflow()
    keras = tensorflow.keras

    event_ids = keras.layers.Input(
        shape=(config.sequence_length,), dtype="int32", name="event_ids"
    )
    hidden = keras.layers.Embedding(
        input_dim=config.vocabulary_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="event_embedding",
    )(event_ids)

    for index, units in enumerate(lstm_unit_plan(config)):
        is_final = index == config.hidden_layers - 1
        hidden = keras.layers.LSTM(
            units,
            activation="tanh" if is_final else config.activation,
            return_sequences=not is_final,
            name=f"lstm_{index + 1}",
        )(hidden)

    probabilities = keras.layers.Dense(
        config.vocabulary_size, activation="softmax", name="event_probabilities"
    )(hidden)
    model = keras.Model(inputs=event_ids, outputs=probabilities, name="deeplog")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
