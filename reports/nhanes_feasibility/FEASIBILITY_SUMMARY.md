# NHANES Feasibility Summary

## What problem we tested

We tested whether age, body measurements, blood pressure, medical history, lifestyle and access-to-care variables can help decide who should receive confirmatory glycemic testing first when testing capacity is limited.

## Why glycemic biomarkers were excluded from X

HbA1c, fasting glucose and OGTT glucose would make the prioritisation task circular. They were used only to define the outcome, never as model inputs. The model is a **pre-test ranking tool**, not a diabetes diagnosis.

## Was Tier-1 non-glycemic data enough?

**Decision: NO-GO — fall back to BRFSS.** The strongest evidence is the held-out testing simulation and the comparison against a transparent rule score. XGBoost's high-risk PR-AUC was **0.809**.

## How much testing burden could be reduced?

Testing the top 40% captured **59.1%** of high-risk participants and reduced immediate testing volume by **60.0%**. Reaching 80% high-risk capture required testing **60.9%** of participants, a **39.1%** reduction versus universal testing.

## Does the project have logical and business value?

The concept has value only if ranking meaningfully reduces immediate testing while retaining an acceptable share of high-risk cases. The acceptance criteria make that trade-off explicit and compare the boosted model with a simple operational rule.

## Final go/no-go decision

**NHANES Tier-1 pre-test prioritisation did not create enough incremental value. Recommended fallback: retain BRFSS as the official project and polish feature engineering, threshold tuning and reporting.**
