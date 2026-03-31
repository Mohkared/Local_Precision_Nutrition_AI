"""
safety.py
────────────────────────────────────────────────────────────────────────────
Safety guardrail system for the Precision Nutrition AI.
• Pre-screens user messages BEFORE the agent runs
• Detects eating disorders, paediatric nutrition, kidney disease, etc.
• Returns a SafetyResult with flags, warnings, and mandatory messages
• The agent checks this and can short-circuit with a safe response
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


# ── Safety Categories ──────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    is_safe: bool = True
    risk_level: str = "low"          # low | medium | high | critical
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mandatory_message: Optional[str] = None   # if set, return this verbatim
    resources: list[str] = field(default_factory=list)
    allow_nutrition_advice: bool = True       # if False, skip meal planning etc.


# ── Keyword Patterns ───────────────────────────────────────────────────────

_EATING_DISORDER_PATTERNS = [
    r"\b(anorex|bulim|purging|laxative|restrict.*eat|starv|fast.*lose|not.*eat)\b",
    r"\b(800|700|600|500)\s*(cal|kcal|calories)\b",
    r"\b(thinspo|thinspiration|pro.?ana|pro.?mia)\b",
    r"\bi\s+(don.?t|never|won.?t|refuse to)\s+eat\b",
    r"\blose.*weight.*fast.*without eating\b",
    r"\bhow.*lose.*\d+\s*(kg|lb|pound).*in\s*\d+\s*(day|week)\b",
]

_PEDIATRIC_PATTERNS = [
    r"\b(my\s+)?(child|kid|son|daughter|toddler|infant|baby|8|9|10|11|12|13)\s+(year.?old|yo)\b",
    r"\b(children|kids|pediatric|paediatric)\b",
    r"\bchild.*(weight|diet|calorie|macro)\b",
    r"\bmy\s+(8|9|10|11|12|13).?year\b",
]

_KIDNEY_PATTERNS = [
    r"\b(ckd|chronic kidney|renal failure|kidney disease|dialysis|nephropathy)\b",
    r"\b(egfr|gfr)\s*[<=>]\s*\d+\b",
]

_MEDICAL_CONDITION_PATTERNS = [
    r"\b(chemotherapy|cancer|oncology|radiation therapy)\b",
    r"\b(transplant|organ\s+failure|ventilator|icu|intensive care)\b",
    r"\b(liver\s+(failure|cirrhosis|disease))\b",
    r"\b(heart\s+failure|cardiac\s+failure)\b",
]

_UNSAFE_CALORIE_PATTERNS = [
    r"\b([1-9]\d{2})\s*(cal|kcal|calories)\s*(per day|a day|daily)?\b",  # catches 3-digit kcal
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    tl = text.lower()
    return any(re.search(p, tl) for p in patterns)


def _extract_calorie_value(text: str) -> Optional[int]:
    """Extract an explicitly stated daily calorie goal if present."""
    m = re.search(r"\b(\d{3,4})\s*(cal|kcal|calories)\b", text.lower())
    if m:
        return int(m.group(1))
    return None


# ── Main Screening Function ────────────────────────────────────────────────

def screen_message(message: str) -> SafetyResult:
    """
    Screen a user message for safety concerns.
    Returns a SafetyResult.  Check .mandatory_message first — if set,
    return it directly and skip the agent.
    """
    result = SafetyResult()

    # ── Eating disorder check ──────────────────────────────────────────────
    if _matches_any(message, _EATING_DISORDER_PATTERNS):
        result.is_safe = False
        result.risk_level = "critical"
        result.flags.append("eating_disorder_risk")
        result.allow_nutrition_advice = False
        result.resources = [
            "NEDA Helpline: 1-800-931-2237 (Mon–Thu 9am–9pm ET, Fri 9am–5pm ET)",
            "Crisis Text Line: Text 'NEDA' to 741741",
            "Alliance for Eating Disorders: allianceforeatingdisorders.com",
        ]
        result.mandatory_message = (
            "I care about your wellbeing and want to respond thoughtfully to what you've shared. "
            "What you're describing sounds like it may involve a challenging relationship with food, "
            "and I'm not the right resource for that — I could inadvertently cause harm by offering "
            "dietary advice in this situation.\n\n"
            "Please reach out to specialists who can actually help:\n"
            "• **NEDA Helpline:** 1-800-931-2237\n"
            "• **Crisis Text Line:** Text 'NEDA' to 741741\n"
            "• **Alliance for Eating Disorders:** allianceforeatingdisorders.com\n\n"
            "You deserve proper support. A Registered Dietitian specialising in eating disorder "
            "treatment can work with you safely and compassionately. 💙"
        )
        return result

    # ── Very low calorie check ─────────────────────────────────────────────
    cal_val = _extract_calorie_value(message)
    if cal_val and cal_val < 1200:
        result.is_safe = False
        result.risk_level = "high"
        result.flags.append("unsafe_calorie_target")
        result.allow_nutrition_advice = False
        result.warnings.append(
            f"Stated calorie goal ({cal_val} kcal/day) is below safe minimum "
            "(1,200 kcal women / 1,500 kcal men). This requires clinical supervision."
        )
        result.mandatory_message = (
            f"I noticed you mentioned a calorie goal of around **{cal_val} kcal/day**. "
            "I'm unable to build a plan around that target — "
            "intakes below 1,200 kcal/day (women) or 1,500 kcal/day (men) "
            "carry real health risks (nutrient deficiencies, metabolic slowdown, muscle loss) "
            "and require direct medical and dietitian supervision.\n\n"
            "If you're aiming for weight loss, I can absolutely help you design a safe, "
            "sustainable plan at a healthy deficit (typically 300–500 kcal below TDEE). "
            "Want to start with that instead?"
        )
        return result

    # ── Paediatric check ──────────────────────────────────────────────────
    if _matches_any(message, _PEDIATRIC_PATTERNS):
        result.risk_level = "high"
        result.flags.append("pediatric_population")
        result.warnings.append(
            "Paediatric nutrition query detected. "
            "Calorie targets and weight-loss diets are not appropriate for children. "
            "Response will focus on healthy eating patterns and recommend specialist referral."
        )
        result.allow_nutrition_advice = True   # Allow general advice, but no cal targets

    # ── Kidney disease check ───────────────────────────────────────────────
    if _matches_any(message, _KIDNEY_PATTERNS):
        result.risk_level = "high"
        result.flags.append("kidney_disease")
        result.warnings.append(
            "Kidney disease detected. Standard protein recommendations are contraindicated. "
            "Response will advise referral to Renal Dietitian."
        )

    # ── Severe medical condition ───────────────────────────────────────────
    if _matches_any(message, _MEDICAL_CONDITION_PATTERNS):
        result.risk_level = "medium"
        result.flags.append("complex_medical_condition")
        result.warnings.append(
            "Complex medical condition detected. Response will include strong disclaimer "
            "and recommendation for registered dietitian consultation."
        )

    # ── Pregnancy check ───────────────────────────────────────────────────
    if re.search(r"\b(pregnant|pregnancy|trimester|prenatal|gestational)\b", message.lower()):
        result.risk_level = max(result.risk_level, "medium", key=lambda x: ["low","medium","high","critical"].index(x) if x in ["low","medium","high","critical"] else 0)
        result.flags.append("pregnancy")
        result.warnings.append(
            "Pregnancy detected. Will include pregnancy-specific guidelines and medical disclaimer."
        )

    return result


def format_safety_disclaimer(result: SafetyResult) -> str:
    """
    Generate a concise disclaimer string to append to any response
    when safety flags are present (but not critical enough to block).
    """
    if not result.flags or result.risk_level in ("low",):
        return ""

    lines = ["\n\n---\n⚠️ **Clinical Disclaimer**"]
    if "kidney_disease" in result.flags:
        lines.append(
            "Standard protein and potassium recommendations do **not** apply to kidney disease. "
            "Please consult a **Renal Dietitian** before making dietary changes."
        )
    if "pediatric_population" in result.flags:
        lines.append(
            "Caloric restriction is **not appropriate** for children. "
            "Please consult a **paediatric dietitian** for personalised guidance."
        )
    if "pregnancy" in result.flags:
        lines.append(
            "Nutritional needs during pregnancy are unique. "
            "Please discuss all dietary changes with your **OB-GYN or registered dietitian**."
        )
    if "complex_medical_condition" in result.flags:
        lines.append(
            "Given your medical situation, please work with a **Registered Dietitian Nutritionist** "
            "and your care team for personalised nutrition therapy."
        )
    lines.append(
        "*This AI provides general nutrition information only and is not a substitute "
        "for professional medical or dietetic advice.*"
    )
    return "\n".join(lines)
