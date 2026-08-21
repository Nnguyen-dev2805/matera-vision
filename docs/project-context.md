# Project Context

## Objective

Convert scanned questionnaire PDFs into structured answer rows by detecting handwritten circles, ticks, and other hand marks.

The first version assumes one stable questionnaire layout. Future layouts should be added as versioned form profiles.

## Source Fixtures

| Path | Role |
|---|---|
| `data/pdfs/matera-example.pdf` | 10-page scanned PDF fixture |
| `data/pages/page_1.png` | Rendered visual reference for page 1 |

The PDF is image-only and does not provide a usable text layer. Visual image processing is therefore the primary input path.

## Agreed Solution

```text
PDF
  -> page image extraction
  -> template alignment
  -> handwritten-mark map
  -> fixed ROI extraction
  -> deterministic mark score
  -> classifier for ambiguous ROIs only
  -> confidence / review decision
  -> normalized answers
  -> Excel export
```

The classifier is optional in the first MVP. The initial baseline is deterministic image processing plus thresholds. A supervised classifier is added only if the baseline produces recurring ambiguous cases.

## Context Loading Order

For a focused implementation task, load only:

1. `AGENTS.md`
2. The relevant section of `docs/architecture.md`
3. The relevant section of `docs/evaluation-plan.md`
4. The target source and its tests
5. Only the fixture pages needed for the task

Do not load all rendered pages, debug images, or unrelated documentation into every task context.

## Current Unknowns

- Runtime and language are not selected.
- Excel column naming and ordering are not finalized.
- The amount and diversity of labeled questionnaire pages is not known.
- The operational interface is not selected: CLI, local web app, or service.
- The manual-review experience is not yet designed.

These are implementation inputs, not assumptions to silently fill in.

## Implementation Tracking

- `tasks/plan.md`: ordered implementation plan, dependencies, risks, and open questions.
- `tasks/todo.md`: active checklist and phase checkpoints.

## Next Recommended Work

1. Approve the ready-to-start gate in `tasks/plan.md`.
2. Choose the runtime and scaffold the project.
3. Define the normalized answer contract and initial Excel schema.
4. Render or extract all fixture pages into a reproducible local dataset.
