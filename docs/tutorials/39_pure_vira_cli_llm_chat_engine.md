# Tutorial 39: Pure-Vira CLI LLM Chat Engine — GGUF Vocabulary Parsing, Stateful KV-Cache Attention & Interactive Terminal REPL (Zero FFI)

Building a conversational Large Language Model (LLM) CLI engine typically requires stitching together Python scripts, C++ bindings (`llama.cpp`), external tokenizer libraries (`sentencepiece` / `tiktoken`), and foreign memory wrappers. In **Vira**, every primitive required to load a GGUF model, parse its embedded vocabulary, tokenize user prompts, maintain a stateful Key-Value (KV) attention cache, and stream responses in an interactive terminal REPL is **built directly into the language and standard library—with 100% pure Vira and zero FFI (`extern "C"` or `unsafe`)**.

In this tutorial, you will build a complete, standalone CLI LLM Chat Engine ([`41_cli_llm_chat_engine.vir`](../examples/41_cli_llm_chat_engine.vir)) that loads the conversational [`models/vira_chat.gguf`](../../models/vira_chat.gguf) model and executes multi-turn conversations in ~330 microseconds per turn with **`active_leaks=0`**.

---

## 1. Architectural Overview: 100% Pure Vira, Zero FFI

Every stage of the CLI chat pipeline executes in safe, deterministic Vira:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     models/vira_chat.gguf (GGUF v3)                      │
│  - Binary KV Metadata (general.architecture, general.name, vocab_size)   │
│  - Embedded Vocabulary Array (tokenizer.ggml.tokens, 32 UTF-8 strings)   │
│  - 12 LLaMA Transformer Tensors (token_embd, Q/K/V/O, SwiGLU FFN, head)  │
└───────────────────┬──────────────────────────────────┬───────────────────┘
                    │ os.read_bytes(path)              │ gguf_load_tensor()
┌───────────────────▼───────────────────┐  ┌───────────▼───────────────────┐
│  1. Pure-Vira GGUF Metadata & Vocab   │  │  2. 64-Byte Aligned Weights   │
│     load_gguf_metadata_and_vocab()    │  │     AVX2 / NEON tensor[f32]   │
└───────────────────┬───────────────────┘  └───────────┬───────────────────┘
                    │ []string vocab                   │
┌───────────────────▼──────────────────────────────────▼───────────────────┐
│                3. Pure-Vira Prompt Tokenizer & Role Formatter            │
│      "<user>" (id=2) + keyword/subword IDs + "<assistant>" (id=3)        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ []int prompt_tokens
┌───────────────────────────────────▼──────────────────────────────────────┐
│           4. Stateful KV-Cache Transformer Engine (arena { ... })        │
│  - Prefill & Autoregressive Decode with RoPE (tensor_rope)               │
│  - In-place KV-Cache Update: k_cache.copy_row(pos, k)                    │
│  - Active Context View:      k_cache.slice_rows(0, pos + 1)              │
│  - Multi-Token Attention:    softmax((q @ k_active.T) / sqrt(d)) @ v     │
│  - SwiGLU FFN:               (silu(h2 @ w_gate) * (h2 @ w_up)) @ w_down  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ logits (1 x vocab_size)
┌───────────────────────────────────▼──────────────────────────────────────┐
│        5. Logit Constraints, Temperature Sampling & Streaming CLI        │
│  - Repetition penalty & control masking via logits.get() / logits.set()  │
│  - sample_token(logits, temperature) -> print(vocab[next_tok])           │
│  - Interactive REPL (io.prompt) or CLI Flags (flag.FlagSet)              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why Zero FFI Matters
- **Memory Safety by Construction**: No raw C pointers, no manual `malloc`/`free`, and no `unsafe` blocks.
- **Zero-Leak Per-Token Inference**: Wrapping each token's forward pass inside `arena { ... }` allocates all intermediate activations (`q_proj`, `k_proj`, `k_active`, `qk`, `scores`, `av`, `attn_out`, `gate_proj`, `up`, `ffn_out`, `logits`) on a 1 MB bump-pointer slab that resets in $O(1)$ at the end of every token step.
- **Single-Binary Portability**: Compiles via `vira build` into a standalone native executable with zero external shared library dependencies.

---

## 2. Pure-Vira GGUF v2/v3 Binary Header & Vocabulary Parser

While `gguf_load_tensor(path, name)` loads and dequantizes `F32`, `F16`, `Q8_0`, and `Q4_0` weight matrices into 64-byte aligned `tensor[f32]` buffers, a conversational CLI also needs the model's metadata and embedded token vocabulary (`tokenizer.ggml.tokens`).

Using `os.read_bytes(path)`, we can parse the GGUF binary header and metadata key-value table directly in pure Vira:

```v
pub struct GgufMeta {
pub mut:
	version      int
	tensor_count int
	kv_count     int
	arch         string
	model_name   string
	embed_dim    int
	vocab_size   int
	vocab        []string
}

fn read_u32_le(data []u8, pos int) int {
	b0 := int(data[pos])
	b1 := int(data[pos + 1])
	b2 := int(data[pos + 2])
	b3 := int(data[pos + 3])
	return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}
```

### Extracting `tokenizer.ggml.tokens` (GGUF Value Type `9` = `ARRAY`)
In the GGUF specification:
- Value type `4` (`UINT32`) stores 32-bit integers such as `llama.embedding_length` and `llama.vocab_size`.
- Value type `8` (`STRING`) stores a 64-bit little-endian byte length followed by UTF-8 bytes (not NUL-terminated).
- Value type `9` (`ARRAY`) stores a 32-bit element type (`8` for strings), a 64-bit element count, and the contiguous sequence of elements:

```v
		} else if key == 'tokenizer.ggml.tokens' && vtype == 9 {
			elem_type := read_u32_le(raw, pos)
			pos += 4
			elem_count := read_u64_le(raw, pos)
			pos += 8
			mut idx := 0
			for idx < elem_count {
				if elem_type == 8 {
					tlen := read_u64_le(raw, pos)
					pos += 8
					tok_bytes := raw.slice(pos, pos + tlen)
					vocab.push(tok_bytes.to_string())
					pos += tlen
				} else {
					pos = skip_gguf_val(raw, pos, elem_type)
				}
				idx += 1
			}
		}
```

Every `tok_bytes` slice is automatically freed at loop iteration exit by Vira's static `autofree` pass, while `vocab` retains ownership of the decoded UTF-8 token strings.

---

## 3. Pure-Vira Prompt Tokenizer & Role Control Tokens

Conversational instruction-tuned models frame each user turn with special control tokens so the attention mechanism distinguishes user prompts from assistant replies:

| Token ID | Token String | Purpose |
| :--- | :--- | :--- |
| `0` | `<pad>` | Padding token (masked during generation) |
| `1` | `<eos>` | End-of-sequence stop token |
| `2` | `<user>` | Marks the start of the user prompt |
| `3` | `<assistant>` | Triggers the assistant's autoregressive response |
| `4..11` | `hello`, `what`, `vira`, `memory`, `actor`, ... | User prompt domain tokens |
| `12..31` | Phrase/subword tokens | Assistant conversational response vocabulary |

Our pure-Vira `tokenize_prompt` function normalizes punctuation, splits the prompt using `strings.fields()`, maps words against the GGUF vocabulary table, and wraps the sequence in `[<user>, ...tokens..., <assistant>]`:

```v
pub fn tokenize_prompt(prompt string, vocab []string) []int {
	mut tokens := []int{}
	// Prepend <user> role control token (id = 2)
	tokens.push(2)

	trimmed_prompt := prompt.trim()
	lower := trimmed_prompt.to_lower()
	c1 := lower.replace('?', '')
	c2 := c1.replace('!', '')
	c3 := c2.replace('.', '')
	cleaned := c3.replace(',', '')
	words := strings.fields(cleaned)

	for w in words {
		mut matched_id := -1
		mut v_idx := 4
		for v_idx < vocab.len {
			if vocab[v_idx] == w {
				matched_id = v_idx
				break
			}
			v_idx += 1
		}
		if matched_id >= 0 {
			tokens.push(matched_id)
		}
	}

	if tokens.len == 1 {
		tokens.push(6) // Default to general 'vira' topic token
	}

	// Append <assistant> role trigger token (id = 3)
	tokens.push(3)
	return tokens
}
```

---

## 4. Stateful KV-Cache Management with `tensor[f32]` Receiver Methods

In a single-token benchmark (like Tutorial 24), the model only attends to the current token (`q @ k.transpose()`). In a real **multi-token CLI chat engine**, each generated token must attend to **every preceding prompt token and previously generated token** without recomputing their Key and Value projections.

Vira provides four native receiver methods on `tensor[f32]` designed specifically for stateful KV caches and logit manipulation:

| Receiver Method | Signature | Description |
| :--- | :--- | :--- |
| **`.get(row, col)`** | `fn (t tensor[f32]) get(row int, col int) f32` | Bounds-checked scalar read at `(row, col)`. |
| **`.set(row, col, val)`** | `fn (t tensor[f32]) set(row int, col int, val f32)` | Bounds-checked scalar write at `(row, col)` (for logit masking & repetition penalties). |
| **`.copy_row(dst_row, src)`** | `fn (t tensor[f32]) copy_row(dst_row int, src tensor[f32])` | Copies a `1 x D` vector `src` directly into row `dst_row` of a pre-allocated `[max_seq_len, D]` KV cache tensor. |
| **`.slice_rows(start, end)`** | `fn (t tensor[f32]) slice_rows(start int, end int) tensor[f32]` | Extracts rows `[start, end)` as a `(end - start) x D` tensor (arena-allocated when called inside `arena {}`). |

### Allocating the Persistent KV Cache
Before entering the chat loop, we allocate `k_cache` and `v_cache` once on the heap with capacity for `64` sequence positions of dimension `meta.embed_dim` (`64`):

```v
k_cache := tensor_fill(64, meta.embed_dim, 0.0)
v_cache := tensor_fill(64, meta.embed_dim, 0.0)
```

---

## 5. Multi-Token Causal Attention, RoPE & SwiGLU Inside `arena { ... }`

During each turn, inference proceeds in two phases:
1. **Phase A (Prompt Prefill)**: For each token in `prompt_tokens[0 .. len-1]`, we compute its RoPE-rotated Key `k` and Value `v` at sequence position `pos` and store them into `k_cache` and `v_cache` using `.copy_row(pos, k)`.
2. **Phase B (Autoregressive Generation)**: Starting from `<assistant>` (`id = 3`), each step appends the current token's `k` and `v` at `pos`, slices the active context window `0 .. pos + 1`, and computes scaled dot-product causal attention across the entire active sequence:

$$\text{Attention}(Q_t, K_{0..t}, V_{0..t}) = \text{softmax}\!\left(\frac{Q_t K_{0..t}^T}{\sqrt{d_k}}\right) V_{0..t}$$

```v
		arena {
			// 1. Token Embedding Lookup (1 x 64)
			h_embed := tensor_embedding(token_emb, cur_token)

			// 2. RMSNorm + Q, K, V Projections + RoPE Positional Encoding
			h1 := tensor_rmsnorm(h_embed, attn_norm, 0.00001)
			q_proj := h1 @ wq
			k_proj := h1 @ wk
			q := tensor_rope(q_proj, pos, 10000.0)
			k := tensor_rope(k_proj, pos, 10000.0)
			v := h1 @ wv

			// 3. Append current step K, V to KV Cache & Slice Active Context [0 .. pos + 1]
			k_cache.copy_row(pos, k)
			v_cache.copy_row(pos, v)
			k_active := k_cache.slice_rows(0, pos + 1)
			v_active := v_cache.slice_rows(0, pos + 1)

			// 4. Causal Multi-Token Attention over full prompt + generated history
			k_t := k_active.transpose()
			qk := q @ k_t
			scores := tensor_scale(qk, 0.125) // 1.0 / sqrt(64) = 0.125
			attn_probs := tensor_softmax(scores)
			av := attn_probs @ v_active
			attn_out := av @ wo
			h_attn := h_embed + attn_out

			// 5. SwiGLU Feed-Forward Network Block
			h2 := tensor_rmsnorm(h_attn, ffn_norm, 0.00001)
			gate_proj := h2 @ w_gate
			gate := tensor_silu(gate_proj)
			up := h2 @ w_up
			ffn_mid := tensor_elemwise_mul(gate, up)
			ffn_out := ffn_mid @ w_down
			h_ffn := h_attn + ffn_out

			// 6. Final RMSNorm & LM Head Logits (1 x vocab_size)
			h_final := tensor_rmsnorm(h_ffn, out_norm, 0.00001)
			logits := h_final @ lm_head

			// 7. Repetition Penalty & Temperature Token Sampling
			apply_logit_constraints(logits, generated_ids, vocab.len, 1.3)
			next_tok = sample_token(logits, temperature)
		}
```

Because `k_active`, `v_active`, `k_t`, `qk`, `scores`, `av`, `attn_out`, `h_attn`, `gate_proj`, `up`, `ffn_out`, `h_ffn`, and `logits` are created inside `arena { ... }`, they are allocated from a single thread-local bump slab and reclaimed in $O(1)$ at the closing brace `}`.

---

## 6. Repetition Penalty & Control-Token Masking

To prevent the model from emitting control tokens (`<pad>`, `<user>`, `<assistant>`) or repeating the same phrase token in a loop, `apply_logit_constraints` uses `.get(0, id)` and `.set(0, id, val)` on the `1 x vocab_size` logit vector before calling `sample_token(logits, temperature)`:

```v
fn apply_logit_constraints(logits tensor[f32], generated []int, vocab_size int, penalty f32) {
	// Mask control tokens (<pad>=0, <user>=2, <assistant>=3) from generation output
	logits.set(0, 0, -1000.0)
	logits.set(0, 2, -1000.0)
	logits.set(0, 3, -1000.0)

	// Apply repetition penalty to tokens already emitted in this response turn
	for prev_id in generated {
		if prev_id >= 0 && prev_id < vocab_size {
			cur_val := logits.get(0, prev_id)
			if cur_val > 0.0 {
				logits.set(0, prev_id, cur_val / penalty)
			} else {
				logits.set(0, prev_id, cur_val * penalty)
			}
		}
	}
}
```

As each token ID `next_tok` is sampled, it is detokenized and streamed immediately to the terminal via `print(vocab[next_tok])` until the model emits `<eos>` (`id = 1`) or reaches `max_tokens`.

---

## 7. CLI Flags (`import flag`) & Interactive REPL (`io.prompt`)

Using Vira's standard `flag` and `io` modules, the binary supports three execution modes out of the box:

1. **Automated Multi-Turn Verification Session (Default)**: Runs a 4-turn conversation and verifies `active_leaks=0`.
2. **Single-Shot CLI Prompt (`--prompt` / `-p`)**:
   ```bash
   ./vira build -o build/vira_chat docs/examples/41_cli_llm_chat_engine.vir
   ./build/vira_chat --prompt "How does memory work?"
   ```
3. **Interactive Terminal Chat REPL (`--interactive` / `-i`)**:
   ```bash
   ./build/vira_chat --interactive
   ```
   Supports `/help`, `/stats`, `/clear`, and `/exit` (`/quit`) slash commands using `io.prompt('You      > ')`.

---

## 8. Running the Example & Expected Output

Run the complete example directly with the Vira compiler:

```bash
./vira run docs/examples/41_cli_llm_chat_engine.vir
```

### Expected Output

```text
=== 1. PURE-VIRA GGUF v3 MODEL & VOCABULARY INGESTION ===
Model Name      : Vira-MicroChat-32 (GGUF v3)
Architecture    : llama (hidden_dim=64, vocab_size=32)
Tensors / Vocab : 12 tensors | 32 embedded tokens loaded
=== 2. MULTI-TURN KV-CACHED CONVERSATIONAL CLI SESSION ===
You      > Hello there!
ViraChat > Hello! I am ViraChat, running 100% natively on CPU SIMD with zero FFI.
           [kv_tokens=8 | generated=5 | latency=335 us]
You      > What is the Vira compiler?
ViraChat > Vira is an AOT systems and AI language compiling to C99 with AVX2/NEON tensors.
           [kv_tokens=9 | generated=5 | latency=313 us]
You      > How does memory and arena management work?
ViraChat > Vira uses static escape analysis (Autofree) and 64-byte bump arenas with zero GC pauses.
           [kv_tokens=9 | generated=5 | latency=331 us]
You      > Explain Vira actors and threads
ViraChat > Vira actors schedule across a lock-free Chase-Lev work-stealing pool with OneForOne supervision.
           [kv_tokens=10 | generated=5 | latency=326 us]
=== 3. SESSION SUMMARY ===
Completed 4 conversational turns (36 total tokens) in 100% Pure Vira.
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle, Multi-Threaded)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=476
```

---

## 9. Downloading & Running a Pretrained 5-Layer Hugging Face GGUF Model (`42_huggingface_gguf_llm_inference.vir`)

In addition to `models/vira_chat.gguf`, Vira includes **[`docs/examples/42_huggingface_gguf_llm_inference.vir`](../examples/42_huggingface_gguf_llm_inference.vir)**, which automatically downloads and executes the pretrained **[`ggml-org/test-model-stories260K`](https://huggingface.co/ggml-org/test-model-stories260K)** LLaMA model (`models/stories260K.gguf`, 1.18 MB, GGUF v3, 48 tensors) in **100% Pure Vira (Zero FFI)**:

- **Automatic Hugging Face Downloader**: Checks `os.exists('models/stories260K.gguf')` and automatically downloads the model from Hugging Face via `exec.Command.new('curl', ['-L', '-s', '-o', model_path, hf_url])` if missing (or when `--download` is passed).
- **Pure-Vira SentencePiece BPE Tokenizer & IEEE-754 Score Parser**: Parses all 512 SentencePiece vocabulary strings (`tokenizer.ggml.tokens`) and 512 IEEE-754 binary32 merge scores (`tokenizer.ggml.scores`) directly from the GGUF header, prepends the SentencePiece `'▁'` word boundary (`id = 410`), and iteratively merges the highest-scoring adjacent subword pairs.
- **5-Layer, 8-Query-Head / 4-KV-Head Grouped-Query Attention (GQA)**: Loads all 48 tensors across `blk.0` through `blk.4` (`embed_dim = 64`, `ffn_dim = 172`, `n_heads = 8`, `n_kv_heads = 4`, `head_dim = 8`), applies multi-head interleaved RoPE, updates per-layer KV caches (`k_cache_0 .. k_cache_4`, `v_cache_0 .. v_cache_4`), and executes SwiGLU FFN inside `arena { ... }` at **>3,000 tokens/sec** with `active_leaks=0`.

### Running the Pretrained Hugging Face Model

```bash
# Default 3-story generation suite (auto-downloads models/stories260K.gguf if missing)
./vira run docs/examples/42_huggingface_gguf_llm_inference.vir

# Custom single-shot story prompt with 60 generated BPE tokens
./vira run docs/examples/42_huggingface_gguf_llm_inference.vir --prompt "A little bird" --max-tokens 60

# Interactive terminal REPL mode
./vira run docs/examples/42_huggingface_gguf_llm_inference.vir --interactive
```

### Output from `42_huggingface_gguf_llm_inference.vir`

```text
=== 1. HUGGING FACE GGUF MODEL CHECK & BINARY HEADER PARSING ===
Model File      : models/stories260K.gguf (1185376 bytes, GGUF v3)
HF Source Repo  : ggml-org/test-model-stories260K (llama)
Architecture    : 5 layers | dim=64 | ffn_dim=172 | GQA=8Q/4KV heads
BPE Tokenizer   : 512 SentencePiece tokens | 512 IEEE-754 merge scores | 48 tensors
=== 2. 5-LAYER PRETRAINED HUGGING FACE LLAMA INFERENCE (PURE VIRA) ===
Prompt     > Once upon a time
BPE Tokens > [1, 403, 407, 261, 378] (5 prompt tokens)
Story      > Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play
           [layers=5 | heads=8Q/4KV | kv_steps=49 | speed=3044 tok/s | latency=16097 us]
Prompt     > One day, a little
BPE Tokens > [1, 385, 328, 432, 261, 376] (6 prompt tokens)
Story      > One day, a little girl named Lily went to the park with her mom. She saw a big box with a big box. She wanted to play with it, but it was too hig
           [layers=5 | heads=8Q/4KV | kv_steps=50 | speed=3051 tok/s | latency=16384 us]
Prompt     > In a big forest
BPE Tokens > [1, 359, 416, 261, 370, 272, 414, 276, 356] (9 prompt tokens)
Story      > In a big forest, there was a little girl named Lily. She loved to play with her toys and her friends. One day, she saw a big box with a big box. She want
           [layers=5 | heads=8Q/4KV | kv_steps=53 | speed=2959 tok/s | latency=17910 us]
=== 3. INFERENCE SUMMARY ===
Completed 3 pretrained Hugging Face LLaMA generations (152 total BPE steps) in 100% Pure Vira.
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle, Multi-Threaded)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=12253
```

---

## 10. Qwen 2.5 (`qwen2`) 24-Layer GGUF ChatML Inference (`43_qwen2_5_gguf_chat_inference.vir`)

Vira also ships with **[`docs/examples/43_qwen2_5_gguf_chat_inference.vir`](../examples/43_qwen2_5_gguf_chat_inference.vir)**, which automatically downloads and runs the real 24-layer, 490M-parameter **[`Qwen/Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_0.gguf`)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)** model (`291` tensors, `151,936` ChatML/GPT-2 BPE tokens) in **100% Pure Vira (Zero FFI)**:

- **Multi-Threaded AVX2+FMA / NEON Linear Projection (`x.linear(w)`)**: Computes $x W^T$ directly over GGUF's row-major `[out_dim, in_dim]` weight matrices without transposing, saving 1.5 GB of RAM and parallelizing output projections across worker threads.
- **Biased QKV Linear Projections (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`)**: Unlike LLaMA, Qwen 2.5 adds learned 1D bias vectors to the Query, Key, and Value projections (`q_lin := xn.linear(l_wq[l]) + l_bq[l]`).
- **Per-Head NeoX Split-Half RoPE (`qwen2.rope.freq_base = 1,000,000.0`)**: Reshapes Q (`14 x 64`) and K (`2 x 64`) into heads before rotating half-head pairs `(i, i + 32)` with a 1M-context frequency base (`1000000.0`) and RMSNorm epsilon `1e-6`.
- **14Q/2KV Grouped-Query Attention (`kv_mul = 7`) & Mixed `Q4_0`/`Q8_0`/`F32` Quantization**: Groups the 14 query heads by their 2 shared KV heads (`7` Q heads per KV head) and dequantizes all `291` `Q4_0` (`dtype = 2`), `Q8_0` (`dtype = 8`), and `F32` (`dtype = 0`) tensors via `gguf_load_tensor`.
- **$O(1)$ 65,536-Bucket Hash-Indexed GPT-2 BPE & ChatML Formatter**: Indexes all `151,936` GPT-2 byte-level BPE tokens (`Ġ` space and `Ċ` newline markers) in ~128 ms using `os.read_bytes_head` and formats multi-turn `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` conversations.

### Running the Qwen 2.5 Example

```bash
# Default 2-turn Qwen 2.5-0.5B-Instruct showcase (auto-downloads GGUF if missing)
./vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir

# Custom single-shot prompt
./vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir --prompt "Write a minimal HTML page about Vira."

# Interactive multi-turn Qwen 2.5 ChatML REPL (-i or --interactive)
./vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir -i
```

### Output from `43_qwen2_5_gguf_chat_inference.vir`

```text
========================================================================
   Pure Vira Qwen 2.5 (0.5B-Instruct) GGUF v3 Chat Engine (Zero FFI)   
========================================================================
Model File      : models/qwen2.5-0.5b-instruct-q4_0.gguf
Architecture    : qwen2 (qwen2.5-0.5b-instruct, GGUF v3, 291 tensors)
Hyperparameters : layers=24, dim=896, ffn_dim=4864, head_dim=64
GQA & RoPE      : 14 Q heads / 2 KV heads (7:1 GQA), NeoX RoPE=1000000
ChatML Vocab    : 151936 tokens indexed in 128 ms (<|im_start|>=151644, <|im_end|>=151645)
Loading 291 GGUF tensors (24 layers, Q4_0/Q8_0/F32)... done in 1677 ms!
------------------------------------------------------------------------
User     > What is 25 * 4? Answer briefly.
Qwen2.5  > 25 * 4 = 100.
           [arch=qwen2 | layers=24 | GQA=14Q/2KV | prefill=3402ms | gen=11 tok (~8 tok/s) | kv_pos=54]
User     > Write a minimal HTML page about Vira.
Qwen2.5  > Here's a simple HTML page about Vira:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Vira</title>
</head>
<body>
    <h1>Vira</h1>
    <p>Vira is a popular online platform that offers a wide range of services to its users
           [arch=qwen2 | layers=24 | GQA=14Q/2KV | prefill=1498ms | gen=64 tok (~8 tok/s) | kv_pos=137]
------------------------------------------------------------------------
Qwen 2.5 GGUF v3 Inference Completed Successfully (Zero FFI).
```

---

## Summary of Pure-Vira Capabilities Used

| Layer | Pure-Vira Feature / API | Role in CLI LLM Engine |
| :--- | :--- | :--- |
| **Model Download** | `os.exists()`, `os.mkdir_all()`, `exec.Command.new('curl', ...)` | Automatically fetches pretrained `.gguf` weights from Hugging Face |
| **Binary Ingestion** | `os.read_bytes(path)`, `.slice()`, `.to_string()` | Parses GGUF v3 header, KV metadata, `tokenizer.ggml.tokens` & `tokenizer.ggml.scores` |
| **Weight Loading** | `gguf_load_tensor(path, name)`, `.transpose()` | Loads `F32`, `Q4_0`, and `Q8_0` LLaMA & Qwen 2.5 tensors into 64-byte aligned buffers |
| **Tokenization** | SentencePiece BPE (`bpe_encode`), ChatML (`encode_chatml_prompt`) & Word Tokenizer | Encodes prompts into SentencePiece subwords, Qwen 2.5 ChatML (`<\|im_start\|>`/`<\|im_end\|>`), or role tokens |
| **Stateful KV Cache** | `k_cache.copy_row(pos, k)`, `k_cache.slice_rows(0, pos + 1)` | Appends RoPE-rotated K/V vectors and slices active context window across layers |
| **Transformer Math** | `@`, `+` (QKV bias), `tensor_embedding`, `tensor_rmsnorm`, `tensor_rope`, `tensor_softmax`, `tensor_silu`, `tensor_elemwise_mul` | Executes multi-layer Grouped-Query Attention (GQA), NeoX/Interleaved RoPE, and SwiGLU FFN inside `arena {}` |
| **Logit Sampling** | `logits.get()`, `logits.set()`, `sample_token()` | Applies repetition penalty, masks control tokens, and samples next token ID |
| **CLI & REPL** | `flag.FlagSet`, `io.prompt()`, `print()`, `println()` | Parses CLI flags and drives interactive streaming chat REPL |



