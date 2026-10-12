#!/usr/bin/env python3
"""Serve the STRUT workbench and optional Hugging Face chat API."""

import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.json"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
FEATURES = ["suspension_stiffness", "ride_height", "aero_stall_margin", "vertical_oscillation"]
MAX_AGENT_ROUNDS = 4
MAX_TOOL_CALLS = 6

TOOLS = [
    {"type": "function", "function": {"name": "calculate_suspension", "description": "Calculate front and rear sprung corner mass, spring rates, wheel rates, and static loads from the current vehicle inputs.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_ride_targets", "description": "Compare current spring sizing against changed front and rear ride-frequency targets. Deltas are percentages and are limited to plus or minus 30 percent.", "parameters": {"type": "object", "properties": {"front_delta_percent": {"type": "number", "description": "Front frequency change in percent; negative is softer."}, "rear_delta_percent": {"type": "number", "description": "Rear frequency change in percent; negative is softer."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "assess_cascade", "description": "Score the four explicitly supplied normalized experimental F1 cascade-model inputs. The score is synthetic and not vehicle-calibrated.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "design_validation_checklist", "description": "Return engineering checks relevant before choosing and validating suspension hardware.", "parameters": {"type": "object", "properties": {"use_case": {"type": "string", "enum": ["street", "track", "off-road", "prototype"], "description": "Intended use for prioritizing checklist items."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "dataset_model_info", "description": "Inspect the generated cascade classifier's dataset, training size, epoch count, metrics, and limitations.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def calculate(vehicle):
    required = ["mass", "frontShare", "frontUnsprung", "rearUnsprung", "frontMotion", "rearMotion", "frontFreq", "rearFreq"]
    try:
        values = {key: float(vehicle[key]) for key in required}
    except (KeyError, TypeError, ValueError):
        raise ValueError("Provide all vehicle inputs as numbers") from None
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Vehicle inputs must be finite numbers")
    if not 500 <= values["mass"] <= 5000 or not 30 <= values["frontShare"] <= 70:
        raise ValueError("Vehicle mass or front weight share is outside the supported range")
    for key in ("frontMotion", "rearMotion"):
        if not 0.4 <= values[key] <= 1.5:
            raise ValueError("Motion ratio must be between 0.4 and 1.5")
    for key in ("frontFreq", "rearFreq"):
        if not 0.5 <= values[key] <= 5:
            raise ValueError("Ride target must be between 0.5 and 5 Hz")
    for key in ("frontUnsprung", "rearUnsprung"):
        if not 10 <= values[key] <= 200:
            raise ValueError("Unsprung mass must be between 10 and 200 kg per wheel")

    result = {"vehicle": values, "calculations": {}}
    for axle, share, unsprung, motion, frequency in (
        ("front", values["frontShare"] / 100, values["frontUnsprung"], values["frontMotion"], values["frontFreq"]),
        ("rear", 1 - values["frontShare"] / 100, values["rearUnsprung"], values["rearMotion"], values["rearFreq"]),
    ):
        corner_mass = values["mass"] * share / 2
        sprung_mass = corner_mass - unsprung
        if sprung_mass <= 0:
            raise ValueError("Unsprung mass must be smaller than each axle's corner mass")
        wheel_rate = (2 * math.pi * frequency) ** 2 * sprung_mass
        result["calculations"].update({
            f"{axle}_sprung_mass_kg": sprung_mass,
            f"{axle}_spring_rate_n_per_mm": wheel_rate / motion**2 / 1000,
            f"{axle}_wheel_rate_n_per_mm": wheel_rate / 1000,
            f"{axle}_static_load_kn": corner_mass * 9.80665 / 1000,
        })
    return result


def cascade_score(cascade_inputs):
    try:
        artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        features = {name: float(cascade_inputs[name]) for name in FEATURES}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in features.values()):
            return None
        values = [features[name] for name in artifact["feature_names"]]
        standardized = [(value - artifact["mean"][index]) / artifact["scale"][index] for index, value in enumerate(values)]
        logit = artifact["bias"] + sum(weight * value for weight, value in zip(artifact["weights"], standardized))
        return 1 / (1 + math.exp(-max(-30, min(30, logit))))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def execute_tool(name, arguments, context):
    vehicle = context.get("vehicle", {})
    if name == "calculate_suspension":
        output = calculate(vehicle)
        output["assumptions"] = "Linear springs, constant entered motion ratios, axle load share split evenly left/right, and unsprung mass subtracted per corner."
        return {"summary": "Calculated front/rear spring rates, wheel rates, corner sprung mass, and static loads.", "result": output}
    if name == "compare_ride_targets":
        base = calculate(vehicle)
        front_delta = float(arguments.get("front_delta_percent", 0))
        rear_delta = float(arguments.get("rear_delta_percent", 0))
        if not all(math.isfinite(value) and -30 <= value <= 30 for value in (front_delta, rear_delta)):
            raise ValueError("Ride-frequency comparison deltas must be between -30 and 30 percent")
        alternative_vehicle = dict(vehicle)
        alternative_vehicle["frontFreq"] *= 1 + front_delta / 100
        alternative_vehicle["rearFreq"] *= 1 + rear_delta / 100
        alternative = calculate(alternative_vehicle)
        return {"summary": f"Compared ride targets ({front_delta:+g}% front, {rear_delta:+g}% rear) against baseline.", "result": {"baseline": base["calculations"], "alternative": alternative["calculations"], "ride_frequency_delta_percent": {"front": front_delta, "rear": rear_delta}}}
    if name == "assess_cascade":
        features = context.get("cascade_inputs", {})
        score = cascade_score(features)
        if score is None:
            raise ValueError("Cascade score unavailable; provide all four normalized inputs between 0 and 1 and train the model first")
        return {"summary": f"Scored four explicit normalized cascade inputs; demo score {score:.0%}.", "result": {"score": score, "inputs": features, "warning": "Synthetic F1 demonstration data; not calibrated to this vehicle, telemetry, or a safety limit."}}
    if name == "design_validation_checklist":
        use_case = arguments.get("use_case", "prototype")
        if use_case not in ("street", "track", "off-road", "prototype"):
            raise ValueError("Unsupported use case")
        checklist = ["Measure corner weights at representative operating load.", "Measure motion ratio across the full suspension travel, not at one point only.", "Check spring, damper, tire, bump-stop, and chassis clearance through bump and rebound.", "Verify coil bind margin, damper stroke, bump-stop engagement, and droop travel.", "Include tire stiffness, unsprung mass, and damping in a quarter-car or multibody model.", "Validate alignment, handling balance, and durability with controlled testing and qualified review."]
        if use_case == "track":
            checklist.insert(0, "Define track, tire, aero load, fuel, and driver-load conditions for the setup target.")
        elif use_case == "off-road":
            checklist.insert(0, "Define wheel travel, articulation, payload, impact, and terrain requirements.")
        elif use_case == "street":
            checklist.insert(0, "Check ride comfort, braking/steering behavior, legal requirements, and full passenger/cargo load cases.")
        return {"summary": f"Prepared a {use_case} suspension validation checklist.", "result": {"use_case": use_case, "checks": checklist}}
    if name == "dataset_model_info":
        try:
            artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"summary": "Model artifact is unavailable; run the training script.", "result": {"dataset": "ClarusC64 F1 synthetic suspension cascade", "available": False}}
        details = {key: artifact.get(key) for key in ("dataset", "epochs", "training_rows", "validation_rows", "metrics", "limitations", "license")}
        return {"summary": f"Read model card: {details['training_rows']} synthetic training rows, {details['epochs']} epochs.", "result": details}
    raise ValueError("Tool is not in the allowlist")


def run_local_agent(message, context):
    text = message.lower()
    if re.fullmatch(r"(?:hi|hello|hey|hiya|howdy)(?: there)?[!., ]*", text.strip()):
        return {"reply": "Hi! I can help calculate spring rates, compare ride targets, review setup stability, or build a validation checklist. What would you like to work on?", "trace": []}
    if text.strip() in ("thanks", "thank you", "thanks!", "thank you!"):
        return {"reply": "You’re welcome. What suspension question can I help with next?", "trace": []}
    plan = []
    if any(term in text for term in ("spring", "rate", "calculat", "stiffness", "wheel", "design", "setup", "compare", "softer", "stiffer", "optim", "improv")):
        plan.append(("calculate_suspension", {}))
    if any(term in text for term in ("compare", "softer", "stiffer", "optim", "improv", "alternative")):
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        magnitude = min(float(match.group(1)), 30) if match else 5
        delta = magnitude if "stiffer" in text else -magnitude
        plan.append(("compare_ride_targets", {"front_delta_percent": delta, "rear_delta_percent": delta}))
    if any(term in text for term in ("cascade", "risk", "stability", "aero", "porpois")):
        plan.append(("assess_cascade", {}))
    if any(term in text for term in ("checklist", "validat", "measure", "before", "test", "what should", "design")):
        use_case = next((case for case in ("off-road", "track", "street", "prototype") if case in text), "prototype")
        plan.append(("design_validation_checklist", {"use_case": use_case}))
    if any(term in text for term in ("model", "dataset", "epoch", "trained")):
        plan.append(("dataset_model_info", {}))
    if not plan:
        plan.append(("calculate_suspension", {}))
    trace = []
    results = {}
    for name, arguments in plan[:MAX_TOOL_CALLS]:
        try:
            tool_result = execute_tool(name, arguments, context)
            trace.append({"tool": name, "summary": tool_result["summary"]})
            results[name] = tool_result["result"]
        except (ValueError, TypeError, KeyError) as error:
            trace.append({"tool": name, "summary": f"Tool request rejected: {error}"})
    return {"reply": summarize_local_agent(results), "trace": trace}


def summarize_local_agent(results):
    parts = []
    calculation = results.get("calculate_suspension")
    if calculation:
        values = calculation["calculations"]
        parts.append(f"First-pass spring rates: front {values['front_spring_rate_n_per_mm']:.1f} N/mm and rear {values['rear_spring_rate_n_per_mm']:.1f} N/mm. Wheel rates: {values['front_wheel_rate_n_per_mm']:.1f} / {values['rear_wheel_rate_n_per_mm']:.1f} N/mm. " + calculation["assumptions"])
    comparison = results.get("compare_ride_targets")
    if comparison:
        before, after = comparison["baseline"], comparison["alternative"]
        parts.append(f"Comparison: front spring rate {before['front_spring_rate_n_per_mm']:.1f} → {after['front_spring_rate_n_per_mm']:.1f} N/mm; rear {before['rear_spring_rate_n_per_mm']:.1f} → {after['rear_spring_rate_n_per_mm']:.1f} N/mm for the selected ride-target changes.")
    assessment = results.get("assess_cascade")
    if assessment:
        parts.append(f"Experimental cascade model score: {assessment['score']:.0%}. {assessment['warning']}")
    checklist = results.get("design_validation_checklist")
    if checklist:
        parts.append("Validation checklist:\n- " + "\n- ".join(checklist["checks"]))
    model = results.get("dataset_model_info")
    if model:
        if model.get("available") is False:
            parts.append("Model artifact unavailable; run python3 train_model.py to generate it.")
        else:
            parts.append(f"Model: {model['dataset']} · {model['training_rows']} training rows · {model['validation_rows']} evaluation rows · {model['epochs']} epochs. Limitations: {model['limitations']}")
    parts.append("These are first-pass engineering estimates, not a validated vehicle design. Confirm with measured data, simulation, packaging checks, and qualified testing.")
    return "\n\n".join(parts)


def run_hosted_agent(messages, context):
    system_message = {
        "role": "system",
        "content": (
            "You are STRUT, an agentic vehicle-suspension engineering assistant. Plan using the user goal, call the provided engineering tools "
            "when their computed outputs are needed, inspect tool results, and continue with follow-up tool calls when useful before answering. "
            "Never invent engineering calculations or OEM specifications; rely on tool results for numbers. State assumptions and missing measurements. "
            "The cascade classifier is trained on a tiny synthetic dataset, is not calibrated to this vehicle, and is not a safety limit. Do not "
            "claim safety certification. Recommend qualified review and physical validation for safety-critical decisions. Tools are read-only, "
            "allowlisted, and at most six may be called in this turn. Current validated vehicle inputs and explicit normalized demo inputs: "
            + json.dumps({"vehicle": context.get("vehicle"), "cascade_inputs": context.get("cascade_inputs")}, separators=(",", ":"))
        ),
    }
    conversation = [system_message]
    for item in messages[-8:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            conversation.append({"role": item["role"], "content": item["content"][:2000]})
    trace = []
    calls_used = 0

    def model_turn(allow_tools):
        body = {"model": HF_MODEL, "messages": conversation, "temperature": 0.2, "max_tokens": 700}
        if allow_tools:
            body.update({"tools": TOOLS, "tool_choice": "auto", "parallel_tool_calls": False})
        request = Request("https://router.huggingface.co/v1/chat/completions", data=json.dumps(body).encode("utf-8"), headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]

    for _ in range(MAX_AGENT_ROUNDS):
        if calls_used >= MAX_TOOL_CALLS:
            break
        assistant_message = model_turn(True)
        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls:
            return {"reply": str(assistant_message.get("content") or "I could not produce a final answer."), "trace": trace}
        tool_calls = tool_calls[:MAX_TOOL_CALLS - calls_used]
        assistant_message = {**assistant_message, "tool_calls": tool_calls}
        conversation.append(assistant_message)
        for call in tool_calls:
            if calls_used >= MAX_TOOL_CALLS:
                break
            if not isinstance(call, dict):
                trace.append({"tool": "invalid_tool", "summary": "Tool request rejected: malformed tool call"})
                continue
            function = call.get("function", {})
            name = function.get("name", "")
            try:
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                output = execute_tool(name, arguments, context)
                trace.append({"tool": name, "summary": output["summary"]})
                tool_content = output["result"]
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                tool_content = {"error": str(error)}
                trace.append({"tool": name or "invalid_tool", "summary": f"Tool request rejected: {error}"})
            conversation.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(tool_content, separators=(",", ":"))})
            calls_used += 1
    final_message = model_turn(False)
    return {"reply": str(final_message.get("content") or "I completed the available tool steps but could not produce a final answer."), "trace": trace}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json(200, {"status": "ok", "huggingface_enabled": bool(HF_TOKEN), "model_file_available": MODEL_PATH.is_file()})
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 40_000:
                self.send_json(413, {"error": "Request body must be between 1 byte and 40 KB"})
                return
            request_data = json.loads(self.rfile.read(length))
            messages = request_data.get("messages", [])
            if not isinstance(messages, list) or not messages:
                raise ValueError("At least one chat message is required")
            last_message = messages[-1]
            if not isinstance(last_message, dict) or last_message.get("role") != "user" or not isinstance(last_message.get("content"), str):
                raise ValueError("The last chat message must be a user message containing text")
            context = request_data.get("context", {})
            if not isinstance(context, dict):
                raise ValueError("Chat context must be an object")
            calculate(context.get("vehicle", {}))
            if not HF_TOKEN:
                outcome = run_local_agent(last_message["content"][:2000], context)
                self.send_json(200, {**outcome, "mode": "local-agent"})
                return
            outcome = run_hosted_agent(messages, context)
            self.send_json(200, {**outcome, "mode": "huggingface-agent"})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": f"Invalid request: {error}"})
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            try:
                outcome = run_local_agent(last_message["content"][:2000], context)
                self.send_json(200, {**outcome, "mode": "local-agent", "warning": "Hosted inference was unavailable; completed this request with the local tool planner."})
            except (ValueError, TypeError, KeyError) as local_error:
                self.send_json(400, {"error": str(local_error)})


def main():
    port = int(os.environ.get("PORT", "4173"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"STRUT workbench running at http://localhost:{port}")
    if not HF_TOKEN:
        print("Hugging Face chat is off; local engineering tools are active. Set HF_TOKEN to enable hosted chat.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down STRUT workbench.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()