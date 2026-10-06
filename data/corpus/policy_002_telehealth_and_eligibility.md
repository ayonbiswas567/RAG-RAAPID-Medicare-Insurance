---
doc_id: policy_002
doc_type: payer_policy
title: Telehealth Acceptability and Member Eligibility Verification (Mock Payer Policy RA-110)
effective_date: 2025-01-01
issuer: Example Health Plan (synthetic)
source: synthetic
---
> SYNTHETIC DATA. This policy was written for a technical take-home exercise. It does not represent any real payer's policy and must not be used for real submissions.

## 1. Scope
This policy is a companion to Policy RA-100. It defines when a telehealth encounter is acceptable for risk adjustment and how member eligibility must be verified before submission.

## 2. Telehealth Encounter Requirements
2.1 A telehealth encounter is acceptable for risk adjustment only when it uses real-time interactive audio and video technology between the member and an acceptable provider.
2.2 Audio-only (telephone) encounters are not acceptable for risk adjustment submission, regardless of whether video was attempted. Diagnoses documented during audio-only visits must be recaptured at a qualifying encounter before submission.
2.3 The note must document the modality used (audio-video or audio-only), the location of the member and of the provider, and the member's consent to telehealth.
2.4 Asynchronous communication (patient portal messages, e-mail, store-and-forward) is not acceptable.

## 3. Member Eligibility Verification
3.1 Before any diagnosis is submitted, the member must be verified as actively enrolled in an Example Health Plan Medicare Advantage product on the date of service.
3.2 Eligibility is verified through the plan eligibility service using the member ID and the date of service. The response must show coverage status ACTIVE for that date.
3.3 If the member's coverage was terminated retroactively, any diagnoses documented after the termination date are withdrawn from submission.
3.4 Eligibility lookups must be logged with the member ID, date of service, response status, and timestamp for audit purposes.

## 4. Submission Timing
4.1 Encounters must be submitted within 90 calendar days of the date of service, and no later than the final sweep deadline published annually.
4.2 Late submissions are accepted only with a documented exception approved by the risk adjustment compliance lead.

## 5. Inpatient Encounters
5.1 For inpatient stays, the date of service for eligibility purposes is the discharge date. Diagnoses documented in the discharge summary by the attending physician are acceptable when signed.
