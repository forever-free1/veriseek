# VeriSeek Reward Design

VeriSeek uses a deterministic reward. It does not call embedding models or LLM judges.

## Formula

QASPER currently keeps the original weighted formula:

```text
R = 0.45 * R_answer
  + 0.35 * R_evidence
  + 0.15 * R_format
  + 0.05 * R_conciseness
```

SciFact uses a gated reward after the first SFT+RL run showed evidence of reward hacking:

```text
if format is invalid:
    R = 0

if gold is NOT_ENOUGH_INFO:
    R = 0.80 * R_answer + 0.20 * R_empty_or_concise_evidence

if gold is SUPPORTS or REFUTES and prediction is NOT_ENOUGH_INFO:
    R = 0.05

otherwise:
    R = 0.35 * R_answer + 0.55 * R_evidence + 0.10 * R_conciseness
    if R_evidence < 0.20:
        R = min(R, 0.25)
```

The gated SciFact reward is intended to prevent the model from earning high reward through correct labels and valid XML while providing weak or empty evidence.

## Components

`R_answer`

- SciFact: 1.0 when the predicted label matches the gold label, otherwise 0.0.
- QASPER: token-level F1 between the predicted answer and gold answer.

`R_evidence`

- Extract evidence items from the `<evidence>...</evidence>` block.
- Compare each predicted evidence item with gold evidence items using token-level F1.
- Average the best match for each predicted evidence item.

`R_format`

- 1.0 when both `<answer>` and `<evidence>` blocks exist.
- 0.5 when only one block exists.
- 0.0 otherwise.
- For SciFact gated reward, format is a hard gate rather than a weighted positive reward.

`R_conciseness`

- 1.0 when the evidence block contains 1-5 items.
- 0.0 otherwise.

## Supported Data Sources

- `scifact_evidence`
- `qasper_evidence`

## Ground Truth

The reward expects `reward_model.ground_truth` to be either a dictionary or a JSON string:

```json
{
  "answer": "SUPPORTS",
  "evidence": ["Evidence sentence."]
}
```
