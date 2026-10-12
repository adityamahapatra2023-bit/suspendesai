# STRUT Suspension Engineering Suite

A local, browser-based suspension design workbench with first-pass spring sizing, ride-frequency estimates, setup comparisons, JSON reports, suspension and agent-architecture schematics, and a suspension-focused copilot.

## Run

Requires Python 3.10 or later; the server and trainer use only the Python standard library.

```bash
python3 server.py
```

Open [http://localhost:4173](http://localhost:4173). Vehicle presets are editable reference estimates, not manufacturer specifications. The built-in copilot and engineering calculations work without an API key.

## Train the Demo Classifier

The training script downloads both CSV splits from the Hugging Face dataset, validates and standardizes its four normalized features, trains a logistic classifier with full-batch gradient descent, evaluates it on the held-out split, and writes `model.json`:

```bash
python3 train_model.py --epochs 600
```

Dataset: [ClarusC64 F1 Suspension–RideHeight–AeroStall–Porpoise Cascade](https://huggingface.co/datasets/ClarusC64/cascade-f1-suspension-rideheight-aerostall-porpoise-v0.1) (MIT). Its card describes synthetic setup examples rather than measured telemetry. The current dataset has only 9 training and 9 evaluation rows; reported metrics are illustrative and cannot establish useful predictive performance. In the interface, its score is explicitly labeled experimental and is never treated as a vehicle-specific probability or safety limit.

## Agent Mode

The copilot is a bounded, read-only tool-using agent. It can calculate spring rates, compare ride targets, assess the explicitly entered synthetic cascade inputs, prepare a use-case validation checklist, and inspect the trained model card. With hosted inference enabled, the model chooses tools, observes their results, and can make follow-up tool calls for up to four rounds (six total tool calls). Each executed action is shown in the chat trace. Vehicle values are never changed by the agent.

To enable model-driven planning, configure a Hugging Face Inference Providers token on the server before starting it. The token stays server-side and is never sent to the browser:

```bash
export HF_TOKEN="your-token"
export HF_CHAT_MODEL="Qwen/Qwen2.5-7B-Instruct"
python3 server.py
```

Without a token, a local intent planner selects and runs the same allowlisted tools; it does not use a language model. If hosted inference is unavailable, the request falls back to this local planner. Configure a tool-calling-capable chat model through `HF_CHAT_MODEL` if you change the default.

## Calculation Scope

The spring calculator estimates sprung corner mass from axle weight share minus entered unsprung mass, then computes wheel rate from the undamped ride-frequency target and converts to spring rate using the entered motion ratio. Static corner load is a simple mass-times-gravity estimate. These are first-pass, linear-spring estimates; the workbench does not model damping, tire stiffness, kinematic changes, pitch/roll coupling, bump stops, or vehicle-specific aero. Validate with measured corner weights, motion ratios, travel checks, simulation, and qualified vehicle testing before selecting or using hardware.