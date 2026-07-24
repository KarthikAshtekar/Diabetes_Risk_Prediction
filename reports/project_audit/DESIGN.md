# Diabetes Project Validation Report Design

## Audience and purpose

Technical reviewers need a compact, source-backed view of what ran, what changed,
which results are current, and which limitations prevent overclaiming.

## Layout

- Lead with the validation outcome and repository check counts.
- Put findings and the primary BRFSS comparison chart before execution detail.
- Separate BRFSS model evidence from the NHANES feasibility decision.
- Follow with scope, methods, limitations, next steps, and further questions.

## Visual system

- Use the portable report reader's default Codex theme and typography.
- Use one zero-based bar chart because the compared measures share a 0-1 scale.
- Keep metric strips limited to directly related values from a single source.

## Data and provenance

All quantitative values are copied from regenerated repository artifacts. Every
metric card, chart, and quantitative narrative block references one canonical
source. The bounded snapshot contains only the rows needed to render this report.

## Interaction and export

The deliverable is a self-contained portable HTML report. The chart must retain
tooltips and source affordances, while the static layout remains readable without
interaction.
