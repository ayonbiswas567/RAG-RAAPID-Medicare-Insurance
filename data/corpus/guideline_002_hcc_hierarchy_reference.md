---
doc_id: guideline_002
doc_type: coding_guideline
title: HCC Hierarchy Reference for Diabetes, Kidney Disease, and Heart Failure (Mock Internal Guideline CG-02)
version: 2025.1
issuer: Example Coding Department (synthetic)
source: synthetic
---
> SYNTHETIC DATA. The category labels and rankings in this guideline are simplified and illustrative. They must not be used for real risk adjustment submissions.

## 1. Purpose
This guideline explains how hierarchical condition categories (HCCs) interact for the most common chronic conditions in the exercise corpus.

## 2. Hierarchies
2.1 An HCC hierarchy groups related conditions by severity. When a member has diagnoses that map to more than one HCC within the same hierarchy, only the most severe (highest-ranked) HCC contributes to the risk score. Lower-ranked HCCs in the same hierarchy are dropped.
2.2 HCCs in different hierarchies are additive. For example, a diabetes HCC and a heart failure HCC both count toward the risk score.

## 3. Diabetes Hierarchy (illustrative)
3.1 Diabetes with chronic complications (for example E11.22 type 2 diabetes with diabetic chronic kidney disease, and E11.42 type 2 diabetes with diabetic polyneuropathy) maps to the diabetes-with-complications category.
3.2 Diabetes without complications (E11.9) maps to the diabetes-without-complications category, which is ranked below the complications category.
3.3 If both a complicated and an uncomplicated diabetes code are documented in the same year, only the complications category is reported.
3.4 In the illustrative model, diabetes with and without complications carry the same relative weight; the hierarchy still determines which category is reported.

## 4. Chronic Kidney Disease Hierarchy (illustrative)
4.1 CKD stage 5 and end-stage renal disease rank highest, followed by stage 4, then stage 3b, then stage 3a and stage 3 unspecified. Stages 1 and 2 do not map to a payment HCC in the illustrative model.
4.2 CKD stage 3a (N18.31) maps to the moderate CKD category.
4.3 Diabetic CKD requires both the diabetes combination code (E11.22) and the CKD stage code (N18.3x) to be reported; the stage code supplies the severity.

## 5. Heart Failure (illustrative)
5.1 All specified heart failure codes (I50.2x systolic, I50.3x diastolic, I50.4x combined) map to the heart failure category. Acute, chronic, and acute-on-chronic forms map to the same category in the illustrative model.
5.2 Heart failure is additive with atrial fibrillation (I48.x) and with COPD (J44.x), which belong to different hierarchies.

## 6. Interaction Notes
6.1 Acute kidney injury (N17.x) does not map to a payment HCC in the illustrative model and does not affect the CKD hierarchy.
6.2 Essential hypertension (I10) does not map to a payment HCC in the illustrative model.
