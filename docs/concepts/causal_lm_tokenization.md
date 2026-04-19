# Causal LM Tokenization for Supervised Fine-Tuning

## The Task

Fine-tuning a causal LM (Mistral-7B) for summarization requires concatenating a **prompt** and a **target** into a single sequence:

```bash
<bos> Summarize the following article:\n\n{article}\n\nSummary:{highlights} </eos>
```

The model trains on the full sequence, but we only compute loss on the **target tokens** (highlights + eos). Prompt tokens are masked with `-100` in the labels tensor.

This creates three problems that must be handled carefully.

---

## Problem 1: Truncation Eating the Target

The full sequence (prompt + target) can exceed the model's `max_length` (1280 tokens). If we tokenize the full sequence and truncate from the right, we lose the tail of the summary and the `eos_token`. The model never learns where to stop generating.

**Naive approach (broken):**

```python
full_text = prompt + target + eos
ids = tokenizer(full_text, truncation=True, max_length=1280).input_ids
# If prompt is 1200 tokens → only 80 tokens left for target
# Summary gets cut, eos is gone
```

**Solution: Independent budget truncation.**

Split the token budget between prompt and target:

- `max_prompt_len = 1024` (from `data_config.preprocessing.max_input_length`)
- `max_target_len = 256` (from `data_config.preprocessing.max_target_length`)

The long article absorbs all truncation. The summary (typically ~70-80 tokens for CNN/DailyMail) stays intact, and `eos_token` is always present.

---

## Problem 2: BPE Boundary Artifact

BPE tokenization is context-dependent. The merge decisions at the junction between prompt and target depend on surrounding characters.

If we tokenize prompt and target **separately** and concatenate the token IDs:

```python
prompt_ids = tokenizer("...Summary:", add_special_tokens=False).input_ids
target_ids = tokenizer("Harry Potter was...", add_special_tokens=False).input_ids
input_ids = prompt_ids + target_ids
```

The BPE merges at the boundary may differ from single-pass tokenization:

| Approach              | Tokens at boundary              |
|-----------------------|---------------------------------|
| Single tokenization   | `[":", "Harry", " Potter"]`     |
| Separate tokenization | `[":", "H", "arry", " Potter"]` |

The merge `H + arry → Harry` only happens if the tokenizer sees the full context across the boundary. Separate tokenization prevents cross-boundary merges.

**Impact:** Typically 1-2 tokens differ at the boundary. Negligible for training, but the label masking offset becomes slightly inaccurate.

---

## Problem 3: Spurious Special Tokens

Most tokenizers auto-insert `bos_token` and/or `eos_token` on every `tokenizer()` call. Tokenizing prompt and target separately produces:

```bash
<bos> prompt_tokens </eos>  <bos> target_tokens </eos>
```

But we need exactly:

```bash
<bos> prompt_tokens target_tokens </eos>
```

**Solution:** Always use `add_special_tokens=False` when tokenizing parts, then manually control special token placement.

---

## Our Solution: Single-Pass Tokenization + offset_mapping

We solve all three problems with a single approach:

1. **Tokenize the full concatenated text in one call** (no BPE artifact)
2. **Use `offset_mapping` to find the exact prompt-target boundary** in token space
3. **Truncate each side independently** from the single-pass token list

### Implementation

```python
prompt = prompt_template.format(article=sample[input_field])
target = sample[target_field] + tokenizer.eos_token
full_text = prompt + target

# Single tokenizer call — correct BPE merges everywhere
enc = tokenizer(full_text,
                add_special_tokens=False,
                return_offsets_mapping=True)
ids = enc['input_ids']
offsets = enc['offset_mapping']   # [(char_start, char_end), ...]

# Find exact token where target begins via character position
prompt_char_len = len(prompt)
prompt_end_idx = next(i for i, (s, e) in enumerate(offsets) if s >= prompt_char_len)

# Independent budget truncation from single-pass tokens
prompt_ids = ids[ : min(prompt_end_idx, max_prompt_len)]
target_ids = ids[prompt_end_idx : prompt_end_idx + max_target_len]

# Manually add bos, mask prompt in labels
input_ids = [tokenizer.bos_token_id] + prompt_ids + target_ids
labels = [-100] * (1 + len(prompt_ids)) + target_ids
attention_mask = [1] * len(input_ids)
```

### How offset_mapping Works

Each token maps to its character span in the original string:

```bash
Token index:  0        1       2       3         4
Token:        "Summar" "ize"   ":"     "Harry"   " Potter"
Offset:       (0, 6)   (6, 9)  (9, 10) (10, 15)  (15, 22)
```

If `len(prompt) = 10` (prompt ends at the colon), then `next(i for i, (s, e) ... if s >= 10)` returns index 3 — the first token that belongs entirely to the target.

### Edge Case: Boundary Token Spanning the Split

If a token straddles the prompt-target boundary (e.g., offset `(9, 12)` where prompt ends at char 10), it goes to the **prompt side** (gets masked to `-100`). At most 1 target token is lost to masking — negligible.

---

## Why Not Just Tokenize Separately?

It works — every major fine-tuning framework (Alpaca, Axolotl, LLaMA-Factory) uses separate tokenization. The BPE mismatch is 1-2 tokens and doesn't measurably affect training. We chose the `offset_mapping` approach for correctness, but separate tokenization with `add_special_tokens=False` is equally valid in practice.

---

## Label Masking Rationale

```bash
Tokens:  <bos>  Summarize  the  ...  Summary:  Harry  Potter  was  ...  </eos>
Labels:  -100   -100       -100 ...  -100      Harry  Potter  was  ...  </eos>
```

- **Prompt masked (-100)**: The model doesn't learn to reproduce the instruction. Training signal comes only from predicting summary tokens.
- **eos_token in labels**: The model learns to emit `</eos>` when the summary is complete — critical for inference to know when to stop.
- **bos_token masked**: It's a fixed start-of-sequence marker, not a prediction target.

---

## Final Sequence Layout

```bash
Position:     0      1 ... prompt_len   prompt_len+1 ... seq_len-1
Token:        <bos>  prompt_tokens...   target_tokens...  </eos>
input_ids:    1      ...                ...               2
labels:       -100   -100 ...           target_id...      2
attn_mask:    1      1 ...              1 ...             1
```

Total max length: `1 (bos) + 1024 (prompt) + 256 (target) = 1281`
