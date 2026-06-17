import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_prompt(x):
    if hasattr(x, 'tolist'):
        x = x.tolist()
    if isinstance(x, list):
        return x
    return [{'role': 'user', 'content': str(x)}]


def add_strict_eval_prompt(messages):
    prefix = (
        "Evaluation mode: do not explain your reasoning. "
        "Return only the final XML blocks. "
        "The first characters of your response must be <answer>.\n\n"
    )
    suffix = "\n\nReturn only the final XML now. Do not add any text before <answer>."
    updated = []
    for msg in messages:
        msg = dict(msg)
        if msg.get('role') == 'user':
            msg['content'] = prefix + str(msg.get('content', '')) + suffix
        updated.append(msg)
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=192)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument(
        '--prefill_no_think',
        action='store_true',
        help='Append a closing </think> prefix after the chat template for Qwen3 Thinking evals.',
    )
    ap.add_argument(
        '--strict_eval_prompt',
        action='store_true',
        help='Wrap each prompt with a stronger no-explanation XML-only evaluation instruction.',
    )
    ap.add_argument(
        '--assistant_prefix',
        default='',
        help='Optional assistant-side prefix inserted before generation and prepended to decoded predictions.',
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    df = pd.read_parquet(args.data_path)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict('records')
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with Path(args.out_jsonl).open('w', encoding='utf-8') as f, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [normalize_prompt(row['prompt']) for row in batch]
            if args.strict_eval_prompt:
                prompts = [add_strict_eval_prompt(prompt) for prompt in prompts]
            rendered = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
            if args.prefill_no_think:
                rendered = [text + "\n</think>\n\n" for text in rendered]
            if args.assistant_prefix:
                rendered = [text + args.assistant_prefix for text in rendered]
            inputs = tok(rendered, return_tensors='pt', padding=True, truncation=True, max_length=1800).to(model.device)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            prompt_len = inputs['input_ids'].shape[1]
            responses = tok.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
            for row, pred in zip(batch, responses):
                if args.assistant_prefix:
                    pred = args.assistant_prefix + pred
                rec = {
                    'prediction': pred,
                    'ground_truth': row['reward_model']['ground_truth'],
                    'data_source': row.get('data_source', 'scifact_evidence'),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f'generated {min(start + args.batch_size, len(rows))}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
