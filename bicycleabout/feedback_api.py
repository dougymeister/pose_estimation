from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import json
import os

router = APIRouter()

FIT_RULES_PATH = os.path.join(os.path.dirname(__file__), "fit_rules.json")

def load_fit_rules() -> dict:
    with open(FIT_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    from_unit = normalize_display_unit(from_unit)
    to_unit = normalize_display_unit(to_unit)

    if from_unit == to_unit:
        return value
    elif from_unit == "cm" and to_unit == "in":
        return value / 2.54
    elif from_unit == "in" and to_unit == "cm":
        return value * 2.54
    else:
        return value  # fallback if unit types are unknown


def normalize_display_unit(unit: str) -> str:
    if not isinstance(unit, str):
        return unit
    return unit.replace("\u00C2\u00B0", "\u00B0").replace("\\u00B0", "\u00B0")


@router.post("/feedback")
async def feedback_handler(request: Request):
    try:
        body = await request.json()
        metrics: Dict[str, Any] = body.get("metrics", {})
        unit: str = body.get("unit", "cm")
        bike_type: str = body.get("bike_type", "road")
        style: str = body.get("style", "endurance")

        print("[SERVER DEBUG] /feedback Received input:", body)

        feedback = generate_feedback(metrics, unit=unit, bike_type=bike_type, style=style)

        print("[SERVER DEBUG] /feedback Returned feedback:", feedback)
        return JSONResponse(content={"feedback": feedback})

    except Exception as e:
        print("[SERVER ERROR] /feedback Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": "Feedback generation failed."})


def generate_feedback(metrics: Dict[str, Any], unit: str = "cm", bike_type: str = "road", style: str = "endurance") -> List[Dict[str, Any]]:
    feedback = []
    seen_keys = set()
    fit_rules = load_fit_rules()

    # Flatten all distances from all_fit_points (if present)
    all_fit_distances = metrics.get("all_fit_points", {}).get("distances", [])

    for category, data in metrics.items():
        # Use fallback to global all_fit_points if distances are missing
        if category != "all_fit_points":
            all_distances = data.get("distances", [])
            if not all_distances:
                all_distances = all_fit_distances
        else:
            all_distances = data.get("distances", [])

        # Process all entries (angles + distances)
        for entry in data.get("angles", []) + data.get("distances", []):
            points = entry.get("points")
            if not points or not isinstance(points, list):
                continue

            value = entry.get("angle_deg") if "angle_deg" in entry else entry.get("distance") or entry.get("distance_in") or entry.get("distance_cm")
            if value is None:
                print(f"[DEBUG] No distance value found for entry: {entry}")
                continue

            symbol = "\u00B0" if "angle_deg" in entry else unit
            label = entry.get("label", "")
            layer = entry.get("layer", "").lower()

            # ✅ Use explicit key if present
            rule_key = entry.get("key")

            # If no key is provided, construct it
            if not rule_key:
                # ANGLES: 3 points
                if "angle_deg" in entry and len(points) == 3:
                    a, b, c = points
                    if a in [11, 13, 15] or b in [11, 13, 15] or c in [11, 13, 15] or a == 5:
                        side = "left"
                    elif a in [12, 14, 16] or b in [12, 14, 16] or c in [12, 14, 16] or a == 6:
                        side = "right"
                    else:
                        side = ""
                    rule_key = f"{layer}_{side}_{a}_{b}_{c}" if side else f"{layer}_{a}_{b}_{c}"

                # DISTANCES: 2 points
                elif len(points) == 2:
                    a, b = points
                    if a in [5, 9, 11, 13, 15] or b in [5, 9, 11, 13, 15]:
                        side = "left"
                    elif a in [6, 10, 12, 14, 16] or b in [6, 10, 12, 14, 16]:
                        side = "right"
                    else:
                        side = ""
                    rule_key = f"{layer}_{side}_{a}_{b}" if side else f"{layer}_{a}_{b}"

            if not rule_key:
                continue

            if rule_key in seen_keys:
                continue

            seen_keys.add(rule_key)

            print(f"[DEBUG] Checking rule_key: {rule_key}")
            rule = fit_rules.get(rule_key)
            if not rule:
                print(f"[DEBUG] No matching rule for key: {rule_key}")
                continue

            # Retrieve thresholds
            rule_entry = (
                rule.get(bike_type, {}).get(style, {})
                or rule.get(bike_type, {}).get("default", {})
                or {}
            )
            target_min = rule_entry.get("min")
            target_max = rule_entry.get("max")
            explanation = rule_entry.get("explanation", rule.get("explanation", ""))
            rule_unit = normalize_display_unit(rule_entry.get("units", rule.get("units", unit)))

            # Convert value to match rule unit
            converted_value = convert_unit(value, from_unit=unit, to_unit=rule_unit)

            if entry.get("valid_for_feedback") is False:
                status = "Not Evaluated"
                explanation = entry.get("validity_reason") or entry.get("reason") or explanation
            elif target_min is not None and target_max is not None:
                if converted_value < target_min:
                    status = "Too Low"
                elif converted_value > target_max:
                    status = "Too High"
                else:
                    status = "OK"
            else:
                status = "Unknown"

            print("...generate_feedback()...rule_key="+rule_key)

            feedback.append({
                "key": rule_key,  # ← raw metric key goes here
                "metric": rule.get("label", rule_key),
                "value": f"{converted_value:.1f} {rule_unit}",
                "target": f"{target_min}-{target_max} {rule_unit}" if target_min is not None else "N/A",
                "status": status,
                "explanation": explanation,
            })

    return feedback



