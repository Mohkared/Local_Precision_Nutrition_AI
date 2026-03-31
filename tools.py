"""
tools.py
────────────────────────────────────────────────────────────────────────────
Seven tools for the Precision Nutrition ReAct agent:
  1. calculate_tdee_bmi          — BMR, TDEE, BMI with validation
  2. calculate_macro_targets     — Personalised macro breakdown from TDEE
  3. get_food_macros             — USDA-calibrated macros for 35+ foods
  4. retrieve_rag_context        — Real ChromaDB semantic retrieval + citations
  5. check_supplement_safety     — Cross-check dose vs NIH Upper Limits
  6. analyze_meal_nutrition      — Estimate macros for a described meal
  7. calculate_hydration_needs   — Daily water target from body weight / activity

FIX: Replaced erroneous `from sympy import re` with the correct `import re`
     from the standard library.  The local `import re` inside
     analyze_meal_nutrition() has been removed accordingly.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re                        # ← FIX: was `from sympy import re` (wrong library)
import json
from typing import Any

from rag_engine import retrieve_as_string

# ══════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════

def calculate_tdee_bmi(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_multiplier: float,
) -> str:
    """Calculate BMI, BMR (Mifflin–St Jeor), and TDEE with input validation."""
    try:
        # Input validation
        if not (20 <= weight_kg <= 300):
            return "ERROR: weight_kg must be between 20 and 300 kg."
        if not (100 <= height_cm <= 250):
            return "ERROR: height_cm must be between 100 and 250 cm."
        if not (10 <= age <= 110):
            return "ERROR: age must be between 10 and 110 years."
        if gender.lower() not in ("male", "female"):
            return "ERROR: gender must be 'male' or 'female'."
        if not (1.0 <= activity_multiplier <= 2.5):
            return (
                "ERROR: activity_multiplier must be between 1.0 and 2.5. "
                "Use: 1.2 (sedentary), 1.375 (light), 1.55 (moderate), "
                "1.725 (very active), 1.9 (extra active)."
            )

        # Mifflin–St Jeor BMR
        if gender.lower() == "male":
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

        tdee = bmr * activity_multiplier
        bmi  = weight_kg / ((height_cm / 100) ** 2)

        # BMI category
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25.0:
            bmi_cat = "Normal weight"
        elif bmi < 30.0:
            bmi_cat = "Overweight"
        elif bmi < 35.0:
            bmi_cat = "Obese Class I"
        elif bmi < 40.0:
            bmi_cat = "Obese Class II"
        else:
            bmi_cat = "Obese Class III"

        # Safe deficit / surplus ranges
        lose_slow = tdee - 300
        lose_fast = tdee - 500
        gain      = tdee + 300

        return (
            f"SUCCESS:\n"
            f"  BMI      : {bmi:.1f}  ({bmi_cat})\n"
            f"  BMR      : {bmr:.0f} kcal/day\n"
            f"  TDEE     : {tdee:.0f} kcal/day\n"
            f"  For slow weight loss (~0.3 kg/wk): {lose_slow:.0f} kcal/day\n"
            f"  For moderate loss (~0.5 kg/wk)  : {lose_fast:.0f} kcal/day\n"
            f"  For muscle gain (+0.3 kg/wk)    : {gain:.0f} kcal/day"
        )
    except Exception as e:
        return f"ERROR: Could not calculate — {e}"


def calculate_macro_targets(
    tdee_calories: float,
    goal: str,
    weight_kg: float = 70.0,
) -> str:
    """
    Calculate personalised daily macro targets (g and %) from TDEE.
    goal: 'weight_loss' | 'maintenance' | 'muscle_gain' | 'athletic_performance'
    """
    try:
        goal = goal.lower().replace(" ", "_")
        valid_goals = ("weight_loss", "maintenance", "muscle_gain", "athletic_performance")
        if goal not in valid_goals:
            return f"ERROR: goal must be one of {valid_goals}."

        # Macro ratios by goal (protein_pct, carb_pct, fat_pct)
        ratios = {
            "weight_loss":           (0.35, 0.35, 0.30),
            "maintenance":           (0.25, 0.50, 0.25),
            "muscle_gain":           (0.30, 0.45, 0.25),
            "athletic_performance":  (0.20, 0.55, 0.25),
        }
        p_pct, c_pct, f_pct = ratios[goal]

        p_g = (tdee_calories * p_pct) / 4   # protein: 4 kcal/g
        c_g = (tdee_calories * c_pct) / 4   # carbs: 4 kcal/g
        f_g = (tdee_calories * f_pct) / 9   # fat: 9 kcal/g
        p_per_kg = p_g / weight_kg

        return (
            f"SUCCESS — Macro targets for {goal.replace('_', ' ')} "
            f"({tdee_calories:.0f} kcal/day):\n"
            f"  Protein : {p_g:.0f} g/day  ({p_pct*100:.0f}%)  "
            f"[{p_per_kg:.1f} g/kg body weight]\n"
            f"  Carbs   : {c_g:.0f} g/day  ({c_pct*100:.0f}%)\n"
            f"  Fat     : {f_g:.0f} g/day  ({f_pct*100:.0f}%)\n"
            f"  Fibre   : 28–38 g/day  (USDA guideline)\n"
            f"  Note    : These ratios are starting points; adjust based on "
            f"individual response and preferences."
        )
    except Exception as e:
        return f"ERROR: Could not calculate macros — {e}"


# ── Expanded USDA-calibrated food database ────────────────────────────────
_FOOD_DATABASE: dict[str, dict] = {
    # ─ Proteins ────────────────────────────────────────────────────
    "chicken breast":       {"cal": 165, "protein": 31.0, "carbs": 0.0,  "fat": 3.6,  "fiber": 0.0},
    "turkey breast":        {"cal": 189, "protein": 29.0, "carbs": 0.0,  "fat": 7.0,  "fiber": 0.0},
    "salmon":               {"cal": 206, "protein": 22.0, "carbs": 0.0,  "fat": 12.0, "fiber": 0.0},
    "tuna (canned water)":  {"cal": 116, "protein": 26.0, "carbs": 0.0,  "fat": 1.0,  "fiber": 0.0},
    "shrimp":               {"cal": 99,  "protein": 24.0, "carbs": 0.2,  "fat": 0.3,  "fiber": 0.0},
    "eggs":                 {"cal": 155, "protein": 13.0, "carbs": 1.1,  "fat": 11.0, "fiber": 0.0},
    "egg whites":           {"cal": 52,  "protein": 11.0, "carbs": 0.7,  "fat": 0.2,  "fiber": 0.0},
    "ground beef (lean)":   {"cal": 215, "protein": 26.0, "carbs": 0.0,  "fat": 13.0, "fiber": 0.0},
    "tofu (firm)":          {"cal": 76,  "protein": 8.0,  "carbs": 2.0,  "fat": 4.0,  "fiber": 0.3},
    "tempeh":               {"cal": 193, "protein": 19.0, "carbs": 9.0,  "fat": 11.0, "fiber": 0.0},
    "lentils (cooked)":     {"cal": 116, "protein": 9.0,  "carbs": 20.0, "fat": 0.4,  "fiber": 7.9},
    "chickpeas (cooked)":   {"cal": 164, "protein": 8.9,  "carbs": 27.0, "fat": 2.6,  "fiber": 7.6},
    "black beans (cooked)": {"cal": 132, "protein": 8.9,  "carbs": 24.0, "fat": 0.5,  "fiber": 8.7},
    "cottage cheese":       {"cal": 84,  "protein": 11.0, "carbs": 3.0,  "fat": 2.3,  "fiber": 0.0},
    "greek yogurt (nonfat)":{"cal": 59,  "protein": 10.0, "carbs": 3.6,  "fat": 0.4,  "fiber": 0.0},
    "cooked quinoa":        {"cal": 120, "protein": 4.4,  "carbs": 21.3, "fat": 1.9,  "fiber": 2.8},
    # ─ Grains ──────────────────────────────────────────────────────
    "quinoa (cooked)":      {"cal": 120, "protein": 4.4,  "carbs": 21.0, "fat": 1.9,  "fiber": 2.8},
    "brown rice (cooked)":  {"cal": 112, "protein": 2.3,  "carbs": 23.5, "fat": 0.9,  "fiber": 1.8},
    "white rice (cooked)":  {"cal": 130, "protein": 2.7,  "carbs": 28.0, "fat": 0.3,  "fiber": 0.4},
    "oats (dry)":           {"cal": 389, "protein": 17.0, "carbs": 66.0, "fat": 7.0,  "fiber": 10.6},
    "whole wheat bread":    {"cal": 247, "protein": 13.0, "carbs": 41.0, "fat": 4.2,  "fiber": 6.8},
    "sweet potato (baked)": {"cal": 90,  "protein": 2.0,  "carbs": 21.0, "fat": 0.2,  "fiber": 3.3},
    "pasta (cooked)":       {"cal": 131, "protein": 5.0,  "carbs": 25.0, "fat": 1.1,  "fiber": 1.8},
    # ─ Vegetables ──────────────────────────────────────────────────
    "broccoli":             {"cal": 34,  "protein": 2.8,  "carbs": 7.0,  "fat": 0.4,  "fiber": 2.6},
    "spinach":              {"cal": 23,  "protein": 2.9,  "carbs": 3.6,  "fat": 0.4,  "fiber": 2.2},
    "kale":                 {"cal": 49,  "protein": 4.3,  "carbs": 9.0,  "fat": 0.9,  "fiber": 3.6},
    "avocado":              {"cal": 160, "protein": 2.0,  "carbs": 9.0,  "fat": 15.0, "fiber": 7.0},
    # ─ Fruits ──────────────────────────────────────────────────────
    "banana":               {"cal": 89,  "protein": 1.1,  "carbs": 23.0, "fat": 0.3,  "fiber": 2.6},
    "apple":                {"cal": 52,  "protein": 0.3,  "carbs": 14.0, "fat": 0.2,  "fiber": 2.4},
    "blueberries":          {"cal": 57,  "protein": 0.7,  "carbs": 14.0, "fat": 0.3,  "fiber": 2.4},
    "orange":               {"cal": 47,  "protein": 0.9,  "carbs": 12.0, "fat": 0.1,  "fiber": 2.4},
    # ─ Fats / Nuts ─────────────────────────────────────────────────
    "almonds":              {"cal": 579, "protein": 21.0, "carbs": 22.0, "fat": 50.0, "fiber": 12.5},
    "walnuts":              {"cal": 654, "protein": 15.0, "carbs": 14.0, "fat": 65.0, "fiber": 6.7},
    "peanut butter":        {"cal": 588, "protein": 25.0, "carbs": 20.0, "fat": 50.0, "fiber": 6.0},
    "olive oil":            {"cal": 884, "protein": 0.0,  "carbs": 0.0,  "fat": 100.0,"fiber": 0.0},
    # ─ Dairy ───────────────────────────────────────────────────────
    "whole milk":           {"cal": 61,  "protein": 3.2,  "carbs": 4.8,  "fat": 3.3,  "fiber": 0.0},
    "cheddar cheese":       {"cal": 404, "protein": 25.0, "carbs": 1.3,  "fat": 33.0, "fiber": 0.0},
}


def get_food_macros(food_item: str, serving_grams: float = 100.0) -> str:
    """Return macronutrient data for a food item per specified serving size."""
    try:
        if serving_grams <= 0:
            return "ERROR: serving_grams must be a positive number."

        # Fuzzy match: try exact, then partial substring
        key = food_item.lower().strip()
        data = _FOOD_DATABASE.get(key)
        if data is None:
            # Partial match
            matches = [k for k in _FOOD_DATABASE if key in k or k in key]
            if matches:
                key  = matches[0]
                data = _FOOD_DATABASE[key]
            else:
                available = ", ".join(sorted(_FOOD_DATABASE.keys()))
                return (
                    f"DATA NOT FOUND for '{food_item}'. "
                    f"Available items: {available}"
                )

        # Scale to serving size
        scale = serving_grams / 100.0
        return (
            f"DATA FOUND — {key.title()} ({serving_grams:.0f} g serving):\n"
            f"  Calories : {data['cal']  * scale:.0f} kcal\n"
            f"  Protein  : {data['protein'] * scale:.1f} g\n"
            f"  Carbs    : {data['carbs']   * scale:.1f} g\n"
            f"  Fat      : {data['fat']     * scale:.1f} g\n"
            f"  Fibre    : {data['fiber']   * scale:.1f} g\n"
            f"  Source   : USDA FoodData Central (values per 100 g, scaled)"
        )
    except Exception as e:
        return f"ERROR: {e}"


def retrieve_rag_context(query: str) -> str:
    """
    Search the evidence-based nutrition knowledge base (ChromaDB + embeddings).
    Returns grounded context with citation labels ready for injection into prompts.
    """
    try:
        context_str, chunks = retrieve_as_string(query, top_k=4)

        if not chunks:
            return (
                "RAG: No highly relevant documents found for this specific query. "
                "Apply general USDA/ADA/AHA evidence-based guidelines."
            )

        citations_summary = " | ".join(
            f"{c['citation']} {c['source']} (score: {c['score']:.2f})"
            for c in chunks
        )
        return (
            f"RAG CONTEXT (cite these sources in your response):\n"
            f"{context_str}\n\n"
            f"CITATIONS: {citations_summary}"
        )
    except Exception as e:
        return f"RAG ERROR: {e}. Apply general evidence-based nutrition guidelines."


# ── NIH Upper Limit reference ─────────────────────────────────────────────
# (ul)  → Tolerable Upper Intake Level
# (rda) → Recommended Dietary Allowance
_SUPPLEMENT_LIMITS: dict[str, dict] = {
    "vitamin c":       {"ul": 2000, "unit": "mg",  "rda": 90,   "rda_female": 75},
    "vitamin d":       {"ul": 4000, "unit": "IU",  "rda": 600,  "rda_female": 600},
    "vitamin e":       {"ul": 1000, "unit": "mg",  "rda": 15,   "rda_female": 15},
    "vitamin a":       {"ul": 3000, "unit": "mcg", "rda": 900,  "rda_female": 700},
    "niacin":          {"ul": 35,   "unit": "mg",  "rda": 16,   "rda_female": 14},
    "iron":            {"ul": 45,   "unit": "mg",  "rda": 8,    "rda_female": 18},
    "zinc":            {"ul": 40,   "unit": "mg",  "rda": 11,   "rda_female": 8},
    "calcium":         {"ul": 2500, "unit": "mg",  "rda": 1000, "rda_female": 1000},
    "magnesium":       {"ul": 350,  "unit": "mg",  "rda": 420,  "rda_female": 320},
    "selenium":        {"ul": 400,  "unit": "mcg", "rda": 55,   "rda_female": 55},
    "iodine":          {"ul": 1100, "unit": "mcg", "rda": 150,  "rda_female": 150},
    "omega-3":         {"ul": 3000, "unit": "mg",  "rda": 1600, "rda_female": 1100},
    "folic acid":      {"ul": 1000, "unit": "mcg", "rda": 400,  "rda_female": 400},
    "vitamin b6":      {"ul": 100,  "unit": "mg",  "rda": 1.3,  "rda_female": 1.3},
    "vitamin b12":     {"ul": None, "unit": "mcg", "rda": 2.4,  "rda_female": 2.4},
}


def check_supplement_safety(supplement_name: str, daily_dose: float, unit: str) -> str:
    """
    Check whether a supplement dose is within NIH-established Upper Limits.
    Returns safety assessment with context.
    """
    try:
        key = supplement_name.lower().strip()
        info = _SUPPLEMENT_LIMITS.get(key)

        if info is None:
            # Partial match
            matches = [k for k in _SUPPLEMENT_LIMITS if key in k or k in key]
            if matches:
                key  = matches[0]
                info = _SUPPLEMENT_LIMITS[key]
            else:
                known = ", ".join(sorted(_SUPPLEMENT_LIMITS.keys()))
                return (
                    f"SUPPLEMENT NOT IN DATABASE: '{supplement_name}'. "
                    f"Known supplements: {known}. "
                    f"Please consult a healthcare provider for safety information."
                )

        expected_unit = info["unit"]
        ul  = info["ul"]
        rda = info["rda"]

        if unit.lower() != expected_unit.lower():
            return f"WARNING: Unit mismatch. Database uses {expected_unit}, but you provided {unit}."

        if ul is None:
            safety = "✅ NO established Upper Limit (water-soluble, generally safe at high doses)"
            risk   = "low"
        elif daily_dose > ul:
            safety = f"⛔ EXCEEDS Upper Limit ({ul} {expected_unit}/day) — potentially harmful"
            risk   = "high"
        elif daily_dose > ul * 0.75:
            safety = f"⚠️  APPROACHING Upper Limit ({ul} {expected_unit}/day) — use caution"
            risk   = "medium"
        elif daily_dose >= rda:
            safety = f"✅ WITHIN SAFE RANGE — at or above RDA ({rda} {expected_unit}/day)"
            risk   = "low"
        else:
            safety = f"ℹ️  BELOW RDA ({rda} {expected_unit}/day) — may not meet daily needs"
            risk   = "info"

        return (
            f"SUPPLEMENT SAFETY CHECK — {supplement_name.title()} @ {daily_dose} {unit}/day:\n"
            f"  Status : {safety}\n"
            f"  RDA    : {rda} {expected_unit}/day  |  UL: {ul or 'N/A'} {expected_unit}/day\n"
            f"  Risk   : {risk}\n"
            f"  Source : NIH Office of Dietary Supplements"
        )
    except Exception as e:
        return f"ERROR: {e}"


def analyze_meal_nutrition(meal_description: str) -> str:
    """
    Estimate the nutritional content of a described meal using the food database.
    Parses food items and quantities from the description.
    NOTE: `re` is now imported at the module level (no local import needed).
    """
    try:
        meal_lower = meal_description.lower()
        found_items = []

        for food_key in sorted(_FOOD_DATABASE.keys(), key=len, reverse=True):
            if food_key in meal_lower:
                # Look for a gram quantity near the food name
                pattern = (
                    rf"(\d+)\s*(?:g|gram).{{0,4}}{re.escape(food_key)}"
                    rf"|{re.escape(food_key)}.{{0,4}}(\d+)\s*(?:g|gram)"
                )
                m = re.search(pattern, meal_lower)
                grams = float(m.group(1) or m.group(2)) if m else 100.0
                found_items.append((food_key, grams))

        if not found_items:
            return (
                "ANALYSIS: Could not identify specific food items. "
                "Try listing ingredients with amounts, e.g. "
                "'150g chicken breast, 100g brown rice, 200g broccoli'."
            )

        total = {"cal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
        breakdown_lines = []
        for food_key, grams in found_items:
            data  = _FOOD_DATABASE[food_key]
            scale = grams / 100.0
            for k in total:
                total[k] += data[k] * scale
            breakdown_lines.append(
                f"  • {food_key.title()} ({grams:.0f}g): "
                f"{data['cal']*scale:.0f} kcal | "
                f"P {data['protein']*scale:.0f}g | "
                f"C {data['carbs']*scale:.0f}g | "
                f"F {data['fat']*scale:.0f}g"
            )

        breakdown = "\n".join(breakdown_lines)
        return (
            f"MEAL ANALYSIS — '{meal_description[:60]}':\n"
            f"{breakdown}\n\n"
            f"  TOTALS: {total['cal']:.0f} kcal | "
            f"Protein {total['protein']:.0f}g | "
            f"Carbs {total['carbs']:.0f}g | "
            f"Fat {total['fat']:.0f}g | "
            f"Fibre {total['fiber']:.0f}g"
        )
    except Exception as e:
        return f"ERROR: Could not analyse meal — {e}"


def calculate_hydration_needs(
    weight_kg: float,
    activity_level: str = "moderate",
    climate: str = "temperate",
) -> str:
    """Calculate daily water intake needs based on body weight, activity, and climate."""
    try:
        if not (20 <= weight_kg <= 300):
            return "ERROR: weight_kg must be between 20 and 300 kg."

        activity_level = activity_level.lower()
        climate        = climate.lower()

        # Base: 35 ml/kg body weight (WHO recommendation)
        base_ml = weight_kg * 35

        activity_adj = {
            "sedentary":  0,
            "light":      350,
            "moderate":   500,
            "active":     700,
            "very active":1000,
            "athlete":    1500,
        }
        act_bonus = next(
            (v for k, v in activity_adj.items() if k in activity_level),
            500
        )

        climate_adj = {"cold": -200, "temperate": 0, "hot": 400, "humid": 300, "desert": 600}
        clim_bonus  = next(
            (v for k, v in climate_adj.items() if k in climate),
            0
        )

        total_ml   = base_ml + act_bonus + clim_bonus
        total_l    = total_ml / 1000
        glasses_8oz = total_ml / 237

        return (
            f"HYDRATION NEEDS ({weight_kg:.0f} kg, {activity_level}, {climate} climate):\n"
            f"  Daily target : {total_l:.1f} L  ({total_ml:.0f} ml)  "
            f"≈ {glasses_8oz:.0f} cups (8 oz each)\n"
            f"  Base (35 ml/kg)         : {base_ml:.0f} ml\n"
            f"  Activity adjustment     : +{act_bonus} ml\n"
            f"  Climate adjustment      : +{clim_bonus} ml\n"
            f"  Note: ~20 % comes from food; aim for {total_l*0.8:.1f} L from beverages.\n"
            f"  Source: WHO / EFSA hydration guidelines"
        )
    except Exception as e:
        return f"ERROR: {e}"


# ══════════════════════════════════════════════════════════════════════════
# TOOL FUNCTION MAP
# ══════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS: dict[str, Any] = {
    "calculate_tdee_bmi":        calculate_tdee_bmi,
    "calculate_macro_targets":   calculate_macro_targets,
    "get_food_macros":           get_food_macros,
    "retrieve_rag_context":      retrieve_rag_context,
    "check_supplement_safety":   check_supplement_safety,
    "analyze_meal_nutrition":    analyze_meal_nutrition,
    "calculate_hydration_needs": calculate_hydration_needs,
}

# ══════════════════════════════════════════════════════════════════════════
# OLLAMA TOOL SCHEMAS
# ══════════════════════════════════════════════════════════════════════════

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_tdee_bmi",
            "description": (
                "Calculate BMI, Basal Metabolic Rate (BMR), and Total Daily Energy Expenditure "
                "(TDEE) using the Mifflin–St Jeor equation. Use whenever the user mentions "
                "their weight, height, age, gender, and activity level."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg":          {"type": "number",  "description": "Body weight in kilograms"},
                    "height_cm":          {"type": "number",  "description": "Height in centimetres"},
                    "age":                {"type": "integer", "description": "Age in years"},
                    "gender":             {"type": "string",  "enum": ["male", "female"]},
                    "activity_multiplier":{
                        "type": "number",
                        "description": (
                            "Activity multiplier: 1.2 sedentary, 1.375 lightly active, "
                            "1.55 moderately active, 1.725 very active, 1.9 extra active"
                        )
                    },
                },
                "required": ["weight_kg", "height_cm", "age", "gender", "activity_multiplier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_macro_targets",
            "description": (
                "Calculate personalised daily protein, carbohydrate, and fat targets in grams "
                "from a TDEE value and a nutritional goal. Call AFTER calculate_tdee_bmi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tdee_calories": {"type": "number", "description": "TDEE in kcal/day"},
                    "goal": {
                        "type": "string",
                        "enum": ["weight_loss", "maintenance", "muscle_gain", "athletic_performance"],
                        "description": "Nutritional goal of the user",
                    },
                    "weight_kg": {"type": "number", "description": "Body weight in kg (for per-kg protein)"},
                },
                "required": ["tdee_calories", "goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_food_macros",
            "description": (
                "Retrieve exact macronutrient data (calories, protein, carbs, fat, fibre) "
                "for a specific food item from the USDA-calibrated database. "
                "Use to build meal plans or answer questions about specific foods."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "food_item": {
                        "type": "string",
                        "description": "Food name, e.g. 'chicken breast', 'quinoa (cooked)', 'almonds'",
                    },
                    "serving_grams": {
                        "type": "number",
                        "description": "Serving size in grams (default 100 g)",
                    },
                },
                "required": ["food_item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_supplement_safety",
            "description": (
                "Check if a supplement dose is within NIH-established Upper Limits. "
                "Always call this when a user asks about supplement amounts or megadosing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "supplement_name": {
                        "type": "string",
                        "description": "Supplement name, e.g. 'vitamin c', 'zinc', 'omega-3'",
                    },
                    "daily_dose":  {"type": "number", "description": "Daily dose amount"},
                    "unit": {"type": "string", "description": "Unit: mg, mcg, IU, g"},
                },
                "required": ["supplement_name", "daily_dose", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_meal_nutrition",
            "description": (
                "Estimate total calories and macros for a described meal. "
                "Use when a user describes what they ate or a specific meal combination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_description": {
                        "type": "string",
                        "description": "Description of meal with foods and amounts, "
                                       "e.g. '150g chicken breast with 100g brown rice and broccoli'",
                    },
                },
                "required": ["meal_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_hydration_needs",
            "description": (
                "Calculate daily water intake needs based on body weight, activity level, "
                "and climate. Use when user asks about hydration or water intake."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "Body weight in kg"},
                    "activity_level": {
                        "type": "string",
                        "description": "One of: sedentary, light, moderate, active, very active, athlete",
                    },
                    "climate": {
                        "type": "string",
                        "description": "One of: cold, temperate, hot, humid, desert",
                    },
                },
                "required": ["weight_kg"],
            },
        },
    },
]

RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_rag_context",
        "description": (
            "Search the evidence-based nutrition knowledge base for clinical guidelines, "
            "dietary standards, nutrient upper limits, disease-specific diet recommendations, "
            "and supplement safety data. Use for any question involving medical conditions, "
            "specific nutrient recommendations, or clinical guidelines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query for the knowledge base",
                },
            },
            "required": ["query"],
        },
    },
}
