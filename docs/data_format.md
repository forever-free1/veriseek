# VeriSeek Data Format

The MVP keeps the existing RL parquet schema.

Each row should contain:

```python
{
    "prompt": [
        {
            "role": "user",
            "content": "Question or claim..."
        }
    ],
    "reward_model": {
        "ground_truth": "{\"answer\": \"...\", \"evidence\": [\"...\"]}"
    },
    "data_source": "scifact_evidence",
    "ability": "scientific evidence QA",
    "extra_info": {
        "index": "example-id",
        "split": "train"
    }
}
```

`ability` and `extra_info` are not required by the reward function, but they match the existing upstream data style and are useful for debugging.

## Output Format

Prompts should instruct the model to answer with:

```text
<answer>
...
</answer>

<evidence>
[1] evidence sentence
[2] evidence sentence
</evidence>
```

## Data Sources

Use `scifact_evidence` for SciFact-style claim verification.

Use `qasper_evidence` for QASPER-style scientific QA.

Use `qasper_evidence` for LitQA2 evaluation rows as well. LitQA2 is intended for evaluation only in the first version, and this avoids adding a separate training reward route for it.
