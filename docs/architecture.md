# Architecture: Hybrid OMR For Fixed Forms

## Scope

The first system processes one known questionnaire template. It detects hand-drawn marks on scanned pages and emits one answer row per page.

The design must later support additional questionnaire layouts without rewriting the core mark-detection pipeline.

## High-Level Flow

```text
Input PDF
  -> PageExtractor
  -> FormDetector
  -> FormProfile
  -> PageAligner
  -> MarkMapBuilder
  -> RoiExtractor
  -> MarkScorer
  -> AmbiguityRouter
       |-> DeterministicDecision
       |-> RoiClassifier
       `-> ReviewRequired
  -> NormalizedAnswerBuilder
  -> ProfileAwareExporter
  -> XLSX output
```

## Layer 1: Deterministic Preprocessing

Responsibilities:

- extract page images;
- normalize image size and color representation;
- correct rotation, translation, scale, and perspective where needed;
- verify alignment quality;
- preserve a debug overlay showing anchors and page boundaries.

This layer should fail clearly when alignment quality is below the configured minimum. It must not silently continue with invalid ROI coordinates.

## Layer 2: ROI And Mark Detection

Responsibilities:

- load the page-specific ROI definitions from the selected form profile;
- build one or more mark maps, such as color-based and grayscale/template-difference maps;
- calculate interpretable mark features and a deterministic mark score;
- support mark strategies for circles, checkboxes, and rating scales;
- return evidence crops and scores for every answer candidate.

The first profile should use fixed coordinates after alignment. The ROI should focus on the answer marker and its surrounding mark corridor, not the full printed answer text.

## Layer 3: Ambiguity Classifier

The classifier is a secondary component, not the primary page parser.

Input:

- normalized ROI image and/or engineered mark features;
- the form profile and mark strategy;
- deterministic score and evidence features.

Output:

```text
selected | unselected | review
confidence: 0..1
modelVersion
```

The first learned model, if needed, should be a small feature-based classifier such as Random Forest, XGBoost, SVM, or logistic regression. A CNN is a later option when the labeled dataset contains enough variation in handwriting, pens, scans, and mark styles.

## Ambiguity Policy

Use explicit thresholds:

```text
high score/confidence -> automatic selected
low score/confidence  -> automatic unselected
middle range          -> classifier or review
low classifier confidence -> review
```

The system must be allowed to abstain. A low-confidence result must never be silently converted into a confident `0` or `1`.

## Form Profiles

A profile is versioned configuration, not a separate application.

Conceptual profile fields:

```text
formId
version
pageCount or page matching rules
alignment template and anchors
page dimensions
question definitions
option identifiers and ROI coordinates
mark strategy per question/group
question-level validation rules
normalized output field mapping
```

Examples of profile changes:

- same question types, different coordinates: new profile only;
- new checkbox or rating mark behavior: add or configure a mark strategy;
- free-form or dynamically laid-out document: add a separate parser/plugin boundary.

## Form Detection

The first version may receive an explicit form profile selection. Later, automatic detection can use visual anchors, header/logo signatures, page dimensions, or perceptual similarity.

If automatic detection is low-confidence, the system should request profile selection or stop with a structured error. It must not process a page using an uncertain profile.

## Normalized Answer Contract

Computer vision must not write Excel columns directly. It should produce a normalized result such as:

```json
{
  "formId": "matera-pre",
  "formVersion": "v1",
  "pageNumber": 1,
  "answers": [
    {
      "questionId": "Q1",
      "optionId": "b",
      "selected": true,
      "confidence": 0.96,
      "decision": "automatic",
      "evidencePath": "debug/page-001/Q1-b.png"
    }
  ]
}
```

The exporter maps this contract to the profile's Excel schema. This keeps layout recognition, mark detection, and data structuring independently replaceable.

## Boundaries

- `PageExtractor` owns PDF-to-image conversion.
- `FormDetector` selects a profile; it does not detect marks.
- `PageAligner` owns geometric normalization.
- `MarkMapBuilder` and `MarkScorer` own image evidence.
- `RoiClassifier` owns only ambiguous ROI classification.
- `NormalizedAnswerBuilder` owns semantic answer objects.
- `ProfileAwareExporter` owns Excel-specific column ordering and serialization.

Avoid a single function that reads a PDF and directly writes an Excel row. That would couple every future form change to the vision implementation.

## Failure Modes

Return structured failure information for:

- unreadable or unsupported PDF;
- missing or low-quality page image;
- unknown form profile;
- failed alignment;
- missing ROI;
- contradictory question-level results;
- classifier confidence below the review threshold;
- exporter/schema mismatch.

Every failure should identify page number, form profile, question/option where applicable, and an evidence path when an image exists.
