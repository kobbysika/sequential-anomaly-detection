"""Evaluate existing checkpoints without retaining whole-test-set predictions.

Run from the partner repository root. Uses its existing sample construction,
rank scoring, group aggregation rule, metrics, and plotting functions.
Prepared records and DeepLog inputs still reside in RAM; only prediction
storage is bounded by batch size. No training is performed.
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from sequential_anomaly.data.vocabulary import EventVocabulary
from sequential_anomaly.evaluation.metrics import evaluate_scores
from sequential_anomaly.evaluation.plots import save_evaluation_plots
from sequential_anomaly.evaluation.scoring import (
    AnomalyScore, deeplog_scoring_examples,
    score_dablog_records, score_deeplog_predictions,
)
from sequential_anomaly.training.train import read_prepared_windows


def merge_scores(groups, scores):
    for item in scores:
        old = groups.get(item.group_id)
        groups[item.group_id] = item if old is None else AnomalyScore(
            item.group_id, max(old.score, item.score),
            old.is_anomalous or item.is_anomalous,
            tuple(sorted(set(old.attack_categories) | set(item.attack_categories))),
        )


def score_batches(model_name, records, predict, sequence_length, batch_size):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    groups = {}
    if model_name == 'deeplog':
        inputs, targets, metadata = deeplog_scoring_examples(
            records, sequence_length=sequence_length)
        count = len(inputs)
    else:
        count = len(records)
    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        if model_name == 'dablog':
            batch = records[begin:end]
            probabilities = predict([r['event_ids'] for r in batch])
            scores = score_dablog_records(batch, probabilities)
        else:
            probabilities = predict(inputs[begin:end])
            scores = score_deeplog_predictions(
                metadata[begin:end], targets[begin:end], probabilities)
        merge_scores(groups, scores)
        del probabilities, scores
        if begin == 0 or (begin // batch_size + 1) % 100 == 0 or end == count:
            print(f'{model_name}: scored {end:,}/{count:,} examples', flush=True)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['dablog', 'deeplog'])
    parser.add_argument('--prepared', default='results/runs/shared/preprocessed')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    print('TensorFlow:', tf.__version__, 'GPUs:', devices, flush=True)
    if not devices:
        raise RuntimeError('No GPU detected; check the job allocation/environment')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    checkpoint_dir = Path('artifacts/checkpoints') / args.model
    checkpoint = checkpoint_dir / 'best.keras'
    settings = json.loads((checkpoint_dir / 'metadata.json').read_text())['config']
    if settings['model'] != args.model:
        raise ValueError('Checkpoint metadata model mismatch')
    sequence_length = settings['sequence_length']
    prepared = Path(args.prepared)
    vocabulary = EventVocabulary.load(prepared / 'vocabulary.json')
    if len(vocabulary) != settings['vocabulary_size']:
        raise ValueError('Prepared vocabulary size differs from training metadata')
    if args.model == 'dablog':
        model = tf.keras.models.load_model(checkpoint, compile=False)
    else:
        # Rebuild to preserve the left-padding/cuDNN fix used during the pilot.
        from sequential_anomaly.models.deeplog import DeeplogModelConfig, build_deeplog
        model = build_deeplog(DeeplogModelConfig(
            vocabulary_size=len(vocabulary), sequence_length=sequence_length))
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                layer.use_cudnn = False
        model.load_weights(checkpoint)
    records = []
    for partition, filename in [('normal', 'test_normal.jsonl'), ('victim', 'test_victim.jsonl')]:
        current = read_prepared_windows(prepared / filename)
        for record in current:
            record['group_id'] = f"{partition}:{record['group_id']}"
        records.extend(current)
        del current
    print(f'Loaded {len(records):,} test windows', flush=True)

    def predict(inputs):
        return model(tf.convert_to_tensor(inputs, dtype=tf.int32), training=False).numpy()

    scores = score_batches(args.model, records, predict, sequence_length, args.batch_size)
    output = Path('results/runs') / args.model / 'evaluation_batched'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'scores.jsonl').open('w') as stream:
        for score in scores:
            stream.write(json.dumps(asdict(score), sort_keys=True) + '\n')
    metrics = evaluate_scores(scores, threshold=args.threshold)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    with (output / 'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics['curve'][0]))
        writer.writeheader()
        writer.writerows(metrics['curve'])
    save_evaluation_plots(metrics, output)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'curve'}, indent=2))
    print('Saved evaluation to:', output, flush=True)


if __name__ == '__main__':
    main()
