# Evaluation Plan

## Evaluation Goal

Measure whether the hybrid pipeline can reliably convert hand-marked questionnaire pages into correct structured answers while safely routing uncertain cases to review.

## Golden Dataset

Create a human-verified annotation set containing:

- original PDF or page image;
- form profile and page number;
- selected/unselected label for every answer option;
- question type: multiple choice, rating scale, or checkbox;
- mark style and scan condition;
- optional difficulty label and reviewer notes.

Keep source fixtures unchanged. Store annotations and derived debug artifacts separately.

## Dataset Splitting

Split by whole page or whole document, never by randomly mixing ROIs from the same page across train and test.

The hidden test set should contain unseen combinations of:

- handwriting and pen styles;
- mark shapes;
- scan brightness and skew;
- clear and ambiguous marks;
- question types.

## Metrics By Layer

### Alignment and ROI layer

- alignment success rate;
- ROI placement validity;
- proportion of ROIs that intersect the intended answer marker;
- rate of ROI overlap with neighboring options.

### Mark detection layer

For selected/unselected labels:

- false-positive rate;
- false-negative rate;
- precision;
- recall;
- F1-score.

Report these per question type and difficulty group, not only as one aggregate.

### Classifier and review layer

Track:

- automatic decision coverage;
- review rate;
- error rate among automatic decisions;
- confidence calibration;
- percentage of low-confidence cases correctly routed to review.

Operationally, the important relationship is:

```text
coverage = automatic decisions / all decisions
risk     = automatic errors / automatic decisions
```

The system should maximize coverage subject to an agreed maximum risk.

### End-to-end layer

Measure:

- option-level accuracy;
- question-level exact match;
- page-level exact match;
- row count equals page count;
- output values are only `0` or `1`;
- output columns and ordering match the profile schema.

Page-level exact match is important because one wrong option can invalidate the row for that page.

## Baselines

Compare the same hidden test set across:

1. deterministic rules only;
2. deterministic rules plus an ambiguity classifier;
3. deterministic rules plus classifier plus review routing.

The hybrid system should improve the difficult subset without materially regressing the easy subset.

## Initial Acceptance Targets

These are starting gates for a pilot and should be revised after real annotations exist:

- option-level F1 at least 98%;
- false-positive rate no higher than 0.5-1%;
- page-level exact match at least 95%;
- review rate no higher than 10-20%;
- no forced automatic decision below the configured confidence threshold.

If the business prefers fewer manual reviews, coverage can increase only after measuring the resulting risk.

## Error Analysis

Every failed case should be categorized as one of:

- alignment failure;
- ROI too wide or too narrow;
- printed text mistaken for a mark;
- low contrast or scan noise;
- large circle crossing multiple options;
- checkbox or rating-scale handling;
- unseen mark style;
- classifier error;
- output/export mapping error.

Use the category distribution to decide which layer to change. Do not retrain a classifier to fix an alignment problem.

## Evidence And Auditability

For every automatic or reviewed answer, retain enough information to inspect:

- aligned page;
- ROI crop;
- mark map;
- deterministic score;
- classifier confidence and version, if used;
- final decision and review status.

This evidence is required for debugging and for comparing pipeline versions.
