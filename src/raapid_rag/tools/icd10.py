"""Mock ICD-10-CM lookup with an *illustrative* HCC mapping.

Code descriptions follow ICD-10-CM.  The HCC categories, hierarchies and severity ranks are
simplified for the exercise and mirror ``data/corpus/guideline_002_hcc_hierarchy_reference.md``
so that the corpus and the tool agree.  They are not a real CMS-HCC crosswalk.
"""

from __future__ import annotations

from typing import Any

DISCLAIMER = "Illustrative mapping for a technical exercise; not a real HCC crosswalk. Confirm against the official code set."

# code -> (description, hcc_category or None, hierarchy or None, severity_rank or None)
ICD10_TABLE: dict[str, tuple[str, str | None, str | None, int | None]] = {
    # Diabetes
    "E11.22": ("Type 2 diabetes mellitus with diabetic chronic kidney disease", "Diabetes with Chronic Complications", "Diabetes", 2),
    "E11.42": ("Type 2 diabetes mellitus with diabetic polyneuropathy", "Diabetes with Chronic Complications", "Diabetes", 2),
    "E11.65": ("Type 2 diabetes mellitus with hyperglycemia", "Diabetes with Chronic Complications", "Diabetes", 2),
    "E11.9": ("Type 2 diabetes mellitus without complications", "Diabetes without Complications", "Diabetes", 1),
    # Kidney
    "N18.30": ("Chronic kidney disease, stage 3 unspecified", "Chronic Kidney Disease, Moderate (Stage 3)", "Chronic Kidney Disease", 1),
    "N18.31": ("Chronic kidney disease, stage 3a", "Chronic Kidney Disease, Moderate (Stage 3)", "Chronic Kidney Disease", 1),
    "N18.32": ("Chronic kidney disease, stage 3b", "Chronic Kidney Disease, Moderate (Stage 3)", "Chronic Kidney Disease", 2),
    "N18.4": ("Chronic kidney disease, stage 4 (severe)", "Chronic Kidney Disease, Severe (Stage 4)", "Chronic Kidney Disease", 3),
    "N18.5": ("Chronic kidney disease, stage 5", "Chronic Kidney Disease, Stage 5", "Chronic Kidney Disease", 4),
    "N18.6": ("End stage renal disease", "Dialysis Status / End Stage Renal Disease", "Chronic Kidney Disease", 5),
    "N17.9": ("Acute kidney failure, unspecified", None, None, None),
    # Heart failure
    "I50.20": ("Unspecified systolic (congestive) heart failure", "Heart Failure", "Heart Failure", 1),
    "I50.22": ("Chronic systolic (congestive) heart failure", "Heart Failure", "Heart Failure", 1),
    "I50.23": ("Acute on chronic systolic (congestive) heart failure", "Heart Failure", "Heart Failure", 1),
    "I50.32": ("Chronic diastolic (congestive) heart failure", "Heart Failure", "Heart Failure", 1),
    "I50.33": ("Acute on chronic diastolic (congestive) heart failure", "Heart Failure", "Heart Failure", 1),
    "I50.9": ("Heart failure, unspecified", "Heart Failure", "Heart Failure", 1),
    # Pulmonary
    "J44.9": ("Chronic obstructive pulmonary disease, unspecified", "Chronic Obstructive Pulmonary Disease", "COPD", 1),
    "J44.1": ("Chronic obstructive pulmonary disease with (acute) exacerbation", "Chronic Obstructive Pulmonary Disease", "COPD", 1),
    # Arrhythmia
    "I48.19": ("Other persistent atrial fibrillation", "Specified Heart Arrhythmias", "Arrhythmia", 1),
    "I48.20": ("Chronic atrial fibrillation, unspecified", "Specified Heart Arrhythmias", "Arrhythmia", 1),
    "I48.91": ("Unspecified atrial fibrillation", "Specified Heart Arrhythmias", "Arrhythmia", 1),
    # No payment HCC in the illustrative model
    "I10": ("Essential (primary) hypertension", None, None, None),
    "E66.9": ("Obesity, unspecified", None, None, None),
    "E66.01": ("Morbid (severe) obesity due to excess calories", "Morbid Obesity", "Obesity", 1),
    "Z68.30": ("Body mass index [BMI] 30.0-30.9, adult", None, None, None),
    "Z68.41": ("Body mass index [BMI] 40.0-44.9, adult", "Morbid Obesity", "Obesity", 1),
    "Z85.3": ("Personal history of malignant neoplasm of breast", None, None, None),
    "C50.912": ("Malignant neoplasm of unspecified site of left female breast", "Breast, Prostate, and Other Cancers", "Cancer", 1),
    "E03.9": ("Hypothyroidism, unspecified", None, None, None),
    "Z00.00": ("Encounter for general adult medical examination without abnormal findings", None, None, None),
    "M25.50": ("Pain in unspecified joint", None, None, None),
    "M06.9": ("Rheumatoid arthritis, unspecified", "Rheumatoid Arthritis and Inflammatory Connective Tissue Disease", "Inflammatory Arthritis", 1),
    "M05.79": ("Rheumatoid arthritis with rheumatoid factor of multiple sites without organ or systems involvement", "Rheumatoid Arthritis and Inflammatory Connective Tissue Disease", "Inflammatory Arthritis", 1),
    "F32.A": ("Depression, unspecified", None, None, None),
    "E78.5": ("Hyperlipidemia, unspecified", None, None, None),
    "Z79.899": ("Other long term (current) drug therapy", None, None, None),
}

MAX_TERM_MATCHES = 8


def _entry(code: str) -> dict[str, Any]:
    description, category, hierarchy, rank = ICD10_TABLE[code]
    return {
        "code": code,
        "description": description,
        "hcc_category": category,
        "hcc_hierarchy": hierarchy,
        "severity_rank": rank,
    }


def icd10_lookup(query: str) -> dict[str, Any]:
    """Look up an ICD-10-CM code exactly, or search descriptions by term."""
    q = (query or "").strip()
    if not q:
        return {"error": "query must be a non-empty ICD-10-CM code or diagnosis term"}

    code = q.upper()
    if code in ICD10_TABLE:
        entry = _entry(code)
        return {"query": query, "found": True, "match_type": "code", **entry, "matches": [entry], "note": DISCLAIMER}

    needle = q.casefold()
    matches = [_entry(c) for c, (desc, *_rest) in ICD10_TABLE.items() if needle in desc.casefold()][:MAX_TERM_MATCHES]
    return {
        "query": query,
        "found": bool(matches),
        "match_type": "term" if matches else None,
        "code": None,
        "description": None,
        "hcc_category": None,
        "hcc_hierarchy": None,
        "severity_rank": None,
        "matches": matches,
        "note": DISCLAIMER,
    }
