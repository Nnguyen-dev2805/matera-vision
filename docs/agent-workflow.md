# Coding Agent Workflow

This document is a compact operating checklist for future coding tasks in this repository. `AGENTS.md` is the durable rule source; this file provides reusable task prompts and handoff structure.

## Task Start Template

```text
TASK:
[One sentence describing the requested behavior.]

SCOPE:
[Files or modules that may change.]

RELEVANT CONTEXT:
- AGENTS.md
- [architecture section]
- [evaluation section]
- [source/test/fixture paths]

ASSUMPTIONS:
1. [Only assumptions that are necessary and not already specified.]

ACCEPTANCE:
- [Observable behavior]
- [Test or evidence requirement]

RISKS:
- [Known uncertainty or possible regression]
```

## Recommended Execution Loop

1. Inspect workspace state, relevant files, fixtures, and tests.
2. Confirm the task scope and surface any ambiguity.
3. Make one small implementation change.
4. Run focused verification.
5. Inspect output and evidence images where applicable.
6. Repeat for the next slice.
7. Run broader verification and review the final diff.
8. Update architecture, evaluation, or project context if a durable decision changed.

## Handoff Template

```text
IMPLEMENTED:
- [File and behavior]

VERIFIED:
- [Command/check]
- [Visual or data check]

NOT RUN:
- [Check and reason, if any]

RISKS / LIMITATIONS:
- [Known issue or unsupported case]

NEXT CONTEXT:
- [The smallest useful next task]
```

## Context Hygiene

- Load only the relevant document sections and source files for the current task.
- Prefer a short project summary over pasting full logs or every fixture.
- Start a fresh task context when switching between major areas such as profile authoring, mark detection, classifier training, and export.
- After a long task, refresh the summary in `docs/project-context.md` before starting another feature.
- Keep experimental decisions out of durable rules until they are validated.
