# STRUT Suspension Engineering Suite

A local, browser-based suspension design workbench with first-pass spring sizing, ride-frequency estimates, setup comparisons, JSON reports, suspension and agent-architecture schematics, and a suspension-focused copilot.

## Run

Requires Python 3.10 or later; the server and trainer use only the Python standard library.

```bash
python3 server.py
```

Open [http://localhost:4173](http://localhost:4173). Vehicle presets are editable reference estimates, not manufacturer specifications. The built-in copilot and engineering calculations work without an API key.

The server listens on port `4173` by default. Set `PORT` to use a different port:

```bash
PORT=5180 python3 server.py
```

Then open `http://localhost:5180`.

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

## HTTP Endpoints

The Python server also serves the browser app and project files (including `/model.json` and files under `/assets/`) from the project directory.

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/` | Serves `index.html`. |
| `GET` | `/api/health` | Returns JSON with `status`, `huggingface_enabled`, and `model_file_available`. |
| `POST` | `/api/chat` | Runs the local suspension copilot or the optional Hugging Face-backed copilot. |

`POST /api/chat` expects JSON with a non-empty `messages` array and a `context` object. The last message must be a user message with string `content`. `context.vehicle` must contain numeric `mass`, `frontShare`, `frontUnsprung`, `rearUnsprung`, `frontMotion`, `rearMotion`, `frontFreq`, and `rearFreq` values; `context.cascade_inputs` may contain the four demo inputs: `suspension_stiffness`, `ride_height`, `aero_stall_margin`, and `vertical_oscillation`. The browser constructs this context from the current workbench inputs.

For example:

```json
{
  "messages": [
    {"role": "user", "content": "Calculate spring rates for this setup"}
  ],
  "context": {
    "vehicle": {
      "mass": 1250,
      "frontShare": 52,
      "frontUnsprung": 42,
      "rearUnsprung": 38,
      "frontMotion": 0.92,
      "rearMotion": 0.88,
      "frontFreq": 1.65,
      "rearFreq": 1.82
    }
  }
}
```

A successful response contains `reply`, `trace`, and `mode` (`local-agent` or `huggingface-agent`); `warning` may be present when hosted inference falls back to the local planner. Invalid requests return a JSON error, and request bodies over 40 KB are rejected.

## Calculation Scope

For each axle, the calculator splits axle mass evenly between the two corners, then subtracts the entered unsprung mass per wheel:

```text
corner mass (kg) = vehicle mass (kg) × axle weight share / 2
sprung corner mass (kg) = corner mass − unsprung mass per wheel
wheel rate (N/m) = (2 × π × ride frequency (Hz))² × sprung corner mass (kg)
spring rate (N/mm) = wheel rate / motion ratio² / 1000
static corner load (kN) = corner mass × 9.80665 / 1000
```

The entered motion ratio is spring travel divided by wheel travel, as shown in the interface. These are first-pass, linear-spring estimates; the workbench does not model damping, tire stiffness, kinematic changes, pitch/roll coupling, bump stops, or vehicle-specific aero. Validate with measured corner weights, motion ratios, travel checks, simulation, and qualified vehicle testing before selecting or using hardware.
