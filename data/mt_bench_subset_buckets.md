# MT-Bench subset: token length buckets
Tokenizer: `gpt2`. Buckets: **short** ≤ 64, **medium** (64, 256], **long** > 256.
| id | category | raw_tokens | bucket |
|---|---|---:|---|
| mt-001 | writing | 23 | short |
| mt-002 | writing | 18 | short |
| mt-003 | roleplay | 23 | short |
| mt-004 | reasoning | 20 | short |
| mt-005 | reasoning | 21 | short |
| mt-006 | math | 18 | short |
| mt-007 | coding | 16 | short |
| mt-008 | coding | 15 | short |
| mt-009 | stem | 13 | short |
| mt-010 | humanities | 15 | short |
| mt-011 | writing | 19 | short |
| mt-012 | reasoning | 18 | short |
| mt-013 | long_context_anchor | 44 | short |
| mt-014 | long_context_anchor | 33 | short |
| mt-015 | long_context_anchor | 38 | short |
| mt-long-01 | long | 200 | medium |
| mt-long-02 | long | 214 | medium |
| mt-long-03 | long | 376 | long |
| mt-long-04 | long | 441 | long |

## Summary

- **short**: 15
- **medium**: 2
- **long**: 2
- **total prompts**: 19

## Phase H runs (`configs/phase_h_full_modal.yaml`)

Prompts are **repeated** until `min_prompt_tokens` (e.g. 512) is reached, *then* tokenized. The CSV
`context_bucket` column reflects lengths **after** that padding (with `short_token_max` / `medium_token_max`
from the YAML). For stratification by **native** prompt length, filter on `prompt_id` using the table above.
