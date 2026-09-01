# Legacy Classifier

This directory contains the legacy local ML classifier used for ambiguous CV masks in the V17 pipeline.

- It is **not used** by the current production scoring logic.
- It is kept committed for potential future training and evaluation when enough data exists.
- The trained model artifacts (`.pkl`, `.joblib`) are intentionally **ignored** from source control.
- Any future use of this code in production requires an explicit architectural design and integration spec.
