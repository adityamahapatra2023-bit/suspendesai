#!/usr/bin/env python3
"""Fetch, preprocess, train, and evaluate the demo cascade classifier."""

import argparse
import csv
import json
import math
import urllib.request
from pathlib import Path


DATASET = "ClarusC64/cascade-f1-suspension-rideheight-aerostall-porpoise-v0.1"
FEATURES = [
    "suspension_stiffness",
    "ride_height",
    "aero_stall_margin",
    "vertical_oscillation",
]
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data"
OUTPUT = Path(__file__).with_name("model.json")


def fetch_rows(split):
    request = urllib.request.Request(
        f"{BASE_URL}/{split}.csv", headers={"User-Agent": "strut-suspension-workbench/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or not set(FEATURES + ["label_cascade"]).issubset(rows[0]):
        raise ValueError(f"Unexpected or empty Hugging Face {split} split")

    prepared = []
    for row_number, row in enumerate(rows, start=2):
        try:
            values = [float(row[name]) for name in FEATURES]
            label_value = float(row["label_cascade"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric value in {split}.csv row {row_number}") from error
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"Features must be finite normalized values in {split}.csv row {row_number}")
        if label_value not in (0, 1):
            raise ValueError(f"Expected binary label in {split}.csv row {row_number}")
        prepared.append((values, int(label_value)))
    if len({label for _, label in prepared}) < 2:
        raise ValueError(f"The {split} split must contain both classes")
    return prepared


def sigmoid(value):
    value = max(-35, min(35, value))
    return 1 / (1 + math.exp(-value))


def standardize(training_rows):
    means = [sum(row[0][index] for row in training_rows) / len(training_rows) for index in range(len(FEATURES))]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[0][index] - mean) ** 2 for row in training_rows) / len(training_rows)
        scales.append(max(math.sqrt(variance), 0.05))
    return means, scales


def transformed(values, means, scales):
    return [(value - means[index]) / scales[index] for index, value in enumerate(values)]


def train(training_rows, epochs, learning_rate, regularization):
    means, scales = standardize(training_rows)
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    row_count = len(training_rows)
    for _ in range(epochs):
        weight_gradient = [0.0] * len(FEATURES)
        bias_gradient = 0.0
        for values, label in training_rows:
            features = transformed(values, means, scales)
            error = sigmoid(bias + sum(weight * value for weight, value in zip(weights, features))) - label
            bias_gradient += error
            for index, value in enumerate(features):
                weight_gradient[index] += error * value
        bias -= learning_rate * bias_gradient / row_count
        for index in range(len(weights)):
            gradient = weight_gradient[index] / row_count + regularization * weights[index]
            weights[index] -= learning_rate * gradient
    return {"mean": means, "scale": scales, "weights": weights, "bias": bias}


def evaluate(rows, model):
    confusion = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for values, label in rows:
        features = transformed(values, model["mean"], model["scale"])
        probability = sigmoid(model["bias"] + sum(weight * value for weight, value in zip(model["weights"], features)))
        predicted = int(probability >= 0.5)
        key = ("true_" if predicted == label else "false_") + ("positive" if predicted else "negative")
        confusion[key] += 1
    correct = confusion["true_positive"] + confusion["true_negative"]
    precision_denominator = confusion["true_positive"] + confusion["false_positive"]
    recall_denominator = confusion["true_positive"] + confusion["false_negative"]
    precision = confusion["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = confusion["true_positive"] / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--regularization", type=float, default=0.02)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.regularization < 0:
        parser.error("epochs and learning rate must be positive; regularization cannot be negative")

    training_rows = fetch_rows("train")
    test_rows = fetch_rows("tester")
    trained = train(training_rows, args.epochs, args.learning_rate, args.regularization)
    metrics = evaluate(test_rows, trained)
    artifact = {
        "feature_names": FEATURES,
        **trained,
        "epochs": args.epochs,
        "training_rows": len(training_rows),
        "validation_rows": len(test_rows),
        "dataset": DATASET,
        "license": "MIT",
        "metrics": metrics,
        "limitations": "Synthetic demonstration data with very few rows; not real telemetry and not suitable for vehicle safety decisions.",
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_file": str(OUTPUT), "epochs": args.epochs, "training_rows": len(training_rows), "validation_rows": len(test_rows), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()