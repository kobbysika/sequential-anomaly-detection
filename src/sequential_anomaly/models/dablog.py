
"""Modern TensorFlow definition of the Chapter 4 DabLog architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DablogModelConfig:
    """Architecture settings matching the supplied Chapter 4 defaults."""

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


def lstm_unit_plan(config: DablogModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the legacy encoder and mirrored decoder LSTM widths."""

    encoder = tuple(
        max(1, config.hidden_units // divisor)
        for divisor in range(1, config.hidden_layers + 1)
    )
    decoder = tuple(reversed(encoder))
    return encoder, decoder


def _tensorflow() -> Any:
    try:
        import tensorflow
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to build DabLog. Run this step in the configured "
            "Colab or Micromamba environment."
        ) from error
    return tensorflow


def build_dablog(config: DablogModelConfig) -> Any:
    """Build and compile the reverse-sequence LSTM autoencoder used by DabLog."""

    tensorflow = _tensorflow()
    keras = tensorflow.keras
    encoder_units, decoder_units = lstm_unit_plan(config)

    event_ids = keras.layers.Input(
        shape=(config.sequence_length,), dtype="int32", name="event_ids"
    )
    hidden = keras.layers.Embedding(
        input_dim=config.vocabulary_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="event_embedding",
    )(event_ids)

    for index, units in enumerate(encoder_units):
        is_bottleneck = index == len(encoder_units) - 1
        hidden = keras.layers.LSTM(
            units,
            activation=config.activation,
            return_sequences=not is_bottleneck,
            name=f"encoder_lstm_{index + 1}",
        )(hidden)

    hidden = keras.layers.RepeatVector(
        config.sequence_length, name="repeat_bottleneck"
    )(hidden)

    for index, units in enumerate(decoder_units):
        hidden = keras.layers.LSTM(
            units,
            activation=config.activation,
            return_sequences=True,
            name=f"decoder_lstm_{index + 1}",
        )(hidden)

    probabilities = keras.layers.TimeDistributed(
        keras.layers.Dense(config.vocabulary_size, activation="softmax"),
        name="event_probabilities",
    )(hidden)

    model = keras.Model(inputs=event_ids, outputs=probabilities, name="dablog")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
