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
GQA & RoPE      : 14 Q heads / 2 KV heads, RoPE=1000000
Context Window  : max_context=512 tokens (KV cache: 6144 KB across 24 layers)
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

## 11. How to Load & Run *Any* Hugging Face GGUF Model in Pure Vira

Because GGUF v2/v3 is a self-describing container format, you can point a pure-Vira engine at **any** decoder-only Transformer model on Hugging Face (LLaMA 2, LLaMA 3.1/3.2, Qwen 2.5, Mistral 7B, SmolLM2, TinyLlama, DeepSeek-R1-Distill) by handling five standard architectural variations.

### 11.1 Model Family Comparison Table

| Model Family | GGUF `general.architecture` | QKV Bias (`attn_q/k/v.bias`) | Output Head (`output.weight`) | RoPE Base (`rope.freq_base`) | Tokenizer Pre-Tokenizer | Chat Template |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA 2 / TinyLlama** | `llama` | No | Untied (`output.weight`) | `10,000.0` | SentencePiece (`▁` = `0xE2 0x96 0x81`) | `[INST] ... [/INST]` or ChatML |
| **LLaMA 3 / 3.1 / 3.2** | `llama` | No | Tied (1B/3B) or Untied (8B+) | `500,000.0` | GPT-2 Byte-Level BPE (`Ġ` = `0xC4 0xA0`) | `<\|begin_of_text\|><\|start_header_id\|>` |
| **Qwen 2 / Qwen 2.5** | `qwen2` | **Yes** (`blk.{l}.attn_q/k/v.bias`) | Tied or Untied | `1,000,000.0` | GPT-2 Byte-Level BPE (`Ġ` = `0xC4 0xA0`) | ChatML (`<\|im_start\|>...<\|im_end\|>`) |
| **Mistral / Ministral** | `llama` | No | Untied (`output.weight`) | `10,000.0` – `1,000,000.0` | SentencePiece or Tekken BPE | `[INST] ... [/INST]` |
| **SmolLM2 (135M–1.7B)** | `llama` | No | Tied (`token_embd.weight`) | `10,000.0` – `100,000.0` | GPT-2 Byte-Level BPE (`Ġ` = `0xC4 0xA0`) | ChatML (`<\|im_start\|>...<\|im_end\|>`) |

---

### 11.2 Architecture-Agnostic GGUF Metadata Discovery

In GGUF files, hyperparameter keys are prefixed with the value of `general.architecture` (for example, `llama.block_count` vs. `qwen2.block_count` vs. `phi3.block_count`). Instead of hardcoding a single architecture prefix, match the key suffixes using `.ends_with()` so your parser works across **all** GGUF architectures:

```vira
if key == 'general.architecture' && vtype == 8 {
    meta.arch = read_gguf_string(raw, cursor)
    cursor = skip_gguf_value(raw, cursor, vtype)
} else if key == 'general.name' && vtype == 8 {
    meta.name = read_gguf_string(raw, cursor)
    cursor = skip_gguf_value(raw, cursor, vtype)
} else if key.ends_with('.block_count') && vtype == 4 {
    meta.n_layers = read_u32_le(raw, cursor)
    cursor = cursor + 4
} else if key.ends_with('.embedding_length') && vtype == 4 {
    meta.dim = read_u32_le(raw, cursor)
    cursor = cursor + 4
} else if key.ends_with('.feed_forward_length') && vtype == 4 {
    meta.ffn_dim = read_u32_le(raw, cursor)
    cursor = cursor + 4
} else if key.ends_with('.attention.head_count') && vtype == 4 {
    meta.n_heads = read_u32_le(raw, cursor)
    cursor = cursor + 4
} else if key.ends_with('.attention.head_count_kv') && vtype == 4 {
    meta.n_kv_heads = read_u32_le(raw, cursor)
    cursor = cursor + 4
} else if key.ends_with('.rope.freq_base') && vtype == 6 {
    meta.rope_base = read_f32_le(raw, cursor)
    cursor = cursor + 4
} else if key.ends_with('.attention.layer_norm_rms_epsilon') && vtype == 6 {
    meta.rms_eps = read_f32_le(raw, cursor)
    cursor = cursor + 4
}
```

---

### 11.3 Handling Optional QKV Biases (`load_gguf_bias_or_zeros`)

Models like **Qwen 2.5** include additive 1D bias tensors (`blk.{l}.attn_q.bias`, `blk.{l}.attn_k.bias`, `blk.{l}.attn_v.bias`), whereas **LLaMA**, **Mistral**, and **SmolLM2** omit them. Using Vira's `or {}` error-handling block on `gguf_load_tensor`, you can load the bias tensor when present or fall back to a zero vector so the exact same forward pass runs both biased and bias-free architectures:

```vira
// Loads `blk.{l}.attn_q/k/v.bias` if present in the GGUF file,
// or returns a zero bias vector `(1, dim)` for bias-free models (LLaMA, Mistral, SmolLM).
fn load_gguf_bias_or_zeros(path string, name string, dim int) tensor[f32] {
    t := gguf_load_tensor(path, name) or {
        return tensor_fill(1, dim, 0.0)
    }
    return t
}
```

---

### 11.4 Handling Tied vs. Untied Output Embeddings (`output.weight`)

Compact models (such as **SmolLM2-135M**, **LLaMA-3.2-1B**, and some **Qwen 2.5** quantizations) tie their language-modeling classifier head directly to `token_embd.weight` and omit `output.weight` from the GGUF tensor directory to save ~15–25% disk space. Always load `output.weight` with a fallback to `tok_emb.clone()`:

```vira
tok_emb := load_gguf_or_die(model_path, 'token_embd.weight')
out_norm := load_gguf_or_die(model_path, 'output_norm.weight')

// Automatically handle both untied (`output.weight`) and tied (`token_embd.weight`) models:
out_weight := gguf_load_tensor(model_path, 'output.weight') or {
    tok_emb.clone()
}
```

---

### 11.5 Choosing the Right RoPE Layout (`tensor_rope`)

Vira's built-in `tensor_rope(x, pos, rope_base)` adapts its rotation pairing based on the shape of `x`:
1. **Per-Head NeoX Split-Half RoPE (Qwen 2.5, Phi-3, Gemma):**
   When you reshape `q_lin` and `k_lin` to `(n_heads, head_dim)` (`rows > 1`) *before* calling `tensor_rope`, Vira rotates `(d, d + head_dim / 2)` inside each attention head independently:
   ```vira
   q_heads := tensor_rope(q_lin.reshape_2d(meta.n_heads, head_dim), i64(pos), meta.rope_base)
   k_heads := tensor_rope(k_lin.reshape_2d(meta.n_kv_heads, head_dim), i64(pos), meta.rope_base)
   ```
2. **Interleaved Pair RoPE (LLaMA 1/2/3, TinyLlama, Stories260K):**
   When you pass a flat row vector `(1, dim)` (`rows == 1`) to `tensor_rope`, Vira rotates adjacent even/odd pairs `(2*i, 2*i + 1)`:
   ```vira
   q_rot := tensor_rope(q_lin, i64(pos), meta.rope_base)
   k_rot := tensor_rope(k_lin, i64(pos), meta.rope_base)
   ```

---

### 11.6 Swapping Prompt Templates (ChatML vs. LLaMA-3 vs. Mistral)

To switch between instruction-tuned model families, format each user turn with the model's native control tokens:

- **ChatML (Qwen 2.5, SmolLM2, OpenHermes, Yi):**
  ```text
  <|im_start|>system
  {sys_prompt}<|im_end|>
  <|im_start|>user
  {user_prompt}<|im_end|>
  <|im_start|>assistant
  ```
- **LLaMA-3 / LLaMA-3.1 / LLaMA-3.2 Instruct:**
  ```text
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  {sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

  {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

  ```
- **Mistral / LLaMA-2 Instruct:**
  ```text
  <s>[INST] {sys_prompt}

  {user_prompt} [/INST]
  ```

---

## 12. Controlling & Managing the Context Window (`max_context` & KV Cache)

In an autoregressive Transformer, the **Context Window** (`max_context`) determines how many total tokens (system prompt + multi-turn conversation history + generated reply) the model can attend to simultaneously.

### 12.1 KV-Cache Memory Formula & Sizing Table

For every layer $l \in [0, N_{\text{layers}})$, Vira pre-allocates two 2D `f32` matrices—`k_caches[l]` and `v_caches[l]`—each of shape `(max_context, kv_dim)`, where $\text{kv\_dim} = N_{\text{kv\_heads}} \times d_{\text{head}}$.

The total persistent RAM consumed by the KV cache is:

$$\text{KV Cache Bytes} = 2 \times N_{\text{layers}} \times \text{max\_context} \times \left(N_{\text{kv\_heads}} \times d_{\text{head}}\right) \times 4\text{ bytes}$$

Because Grouped-Query Attention (GQA) reduces $N_{\text{kv\_heads}}$ relative to $N_{\text{heads}}$ (for example, Qwen2.5-0.5B uses $14$ Q heads but only $2$ KV heads, a $7:1$ reduction), the KV cache remains remarkably compact even at large context lengths:

| Context Window (`--ctx`) | Stories260K (`8L, kv_dim=32`) | Qwen2.5-0.5B (`24L, kv_dim=128`) | LLaMA-3.2-1B (`16L, kv_dim=512`) | Qwen2.5-7B (`28L, kv_dim=512`) |
| :--- | :--- | :--- | :--- | :--- |
| **`256` tokens** | `512 KB` | `3.0 MB` | `16.0 MB` | `28.0 MB` |
| **`512` tokens** *(default)* | `1.0 MB` | `6.0 MB` | `32.0 MB` | `56.0 MB` |
| **`2048` tokens** | `4.0 MB` | `24.0 MB` | `128.0 MB` | `224.0 MB` |
| **`8192` tokens** | `16.0 MB` | `96.0 MB` | `512.0 MB` | `896.0 MB` |
| **`32768` tokens** | `64.0 MB` | `384.0 MB` | `2.0 GB` | `3.5 GB` |

---

### 12.2 Configuring `max_context` via CLI Flags (`-c` / `--ctx`)

In [43_qwen2_5_gguf_chat_inference.vir](../examples/43_qwen2_5_gguf_chat_inference.vir), `max_context` is configurable at startup via `-c` / `--ctx` / `--context` and directly controls the row capacity of `k_caches` and `v_caches`:

```powershell
# Run with a 2048-token context window and generate up to 256 tokens per turn
.\vira.exe run docs/examples/43_qwen2_5_gguf_chat_inference.vir -autofree=false -- --ctx 2048 --steps 256 -i

# Run with a compact 256-token context window for low-memory edge devices
.\vira.exe run docs/examples/43_qwen2_5_gguf_chat_inference.vir -autofree=false -- --ctx 256 -i
```

Inside `main()`, Vira computes the exact KV-cache memory footprint and allocates the per-layer cache tensors to `(max_context, kv_dim)`:

```vira
kv_cache_kb := (2 * meta.n_layers * max_context * kv_dim * 4) / 1024
println('Context Window  : max_context=${max_context} tokens (KV cache: ${kv_cache_kb} KB across ${meta.n_layers} layers)')

mut l := 0
for l < meta.n_layers {
    k_caches.push(tensor_fill(max_context, kv_dim, 0.0))
    v_caches.push(tensor_fill(max_context, kv_dim, 0.0))
    l += 1
}
```

---

### 12.3 Three Context Window Management Strategies in Pure Vira

#### Strategy 1: Active Prefix Slicing (`.slice_rows(0, pos + 1)`)
Even if you allocate `max_context = 8192`, you should **never** multiply queries against unused zero rows `[pos + 1 .. max_context)`, which would waste CPU cycles and corrupt `tensor_softmax` with $\exp(0)$ terms. Instead, always bound attention to `seq_len := pos + 1` using `.slice_rows(0, seq_len)` (or iterating `t < seq_len`):

```vira
seq_len := pos + 1
k_caches[l].copy_row(pos, k_rot)
v_caches[l].copy_row(pos, v)

// Slice ONLY the active context window [0 .. pos + 1)
k_ctx := k_caches[l].slice_rows(0, seq_len)
v_ctx := v_caches[l].slice_rows(0, seq_len)
```

This keeps per-token attention complexity strictly $O(\text{kv\_pos})$ instead of $O(\text{max\_context})$.

#### Strategy 2: Sliding-Window Context Shifting with Pinned System-Prompt "Attention Sinks"
In long multi-turn REPL sessions, `kv_pos` will eventually approach `max_context`. If you simply reset `kv_pos = 0`, the model loses all conversation memory; if you shift all rows uniformly (`[shift_by .. kv_pos) -> [0 .. kv_pos - shift_by)`), the model loses its initial `<|im_start|>system...` instructions and "attention sink" tokens.

The production solution implemented in [`shift_kv_cache_window`](../examples/43_qwen2_5_gguf_chat_inference.vir#L357-L387) pins the first `keep_prefix` rows (`[0 .. keep_prefix)`, containing the system prompt) and shifts only the rolling conversation window `[keep_prefix + shift_by .. kv_pos)` left into `[keep_prefix .. kv_pos - shift_by)`:

```vira
// Preserves rows `[0 .. keep_prefix)` (the pinned system prompt / attention sinks)
// and shifts recent conversation rows `[keep_prefix + shift_by .. kv_pos)` left.
fn shift_kv_cache_window(
    mut k_caches []tensor[f32],
    mut v_caches []tensor[f32],
    n_layers int,
    keep_prefix int,
    kv_pos int,
    shift_by int
) int {
    if shift_by <= 0 || keep_prefix + shift_by >= kv_pos {
        return keep_prefix
    }
    new_kv_pos := kv_pos - shift_by
    arena {
        mut l := 0
        for l < n_layers {
            mut r := keep_prefix
            for r < new_kv_pos {
                src_r := r + shift_by
                k_row := k_caches[l].slice_rows(src_r, src_r + 1)
                v_row := v_caches[l].slice_rows(src_r, src_r + 1)
                k_caches[l].copy_row(r, k_row)
                v_caches[l].copy_row(r, v_row)
                r += 1
            }
            l += 1
        }
    }
    return new_kv_pos
}
```

Before each turn in [`generate_reply`](../examples/43_qwen2_5_gguf_chat_inference.vir#L542-L555), Vira checks if `kv_pos + turn_tokens.len + max_new_tokens` would exceed `max_context` and automatically slides the KV-cache window:

```vira
needed := turn_tokens.len + max_new_tokens + 4
if kv_pos > 0 && kv_pos + needed >= max_context {
    mut keep_prefix := 24
    if keep_prefix > kv_pos / 4 {
        keep_prefix = kv_pos / 4
    }
    mut shift_by := (kv_pos + needed) - max_context + 32
    if shift_by < (kv_pos - keep_prefix) / 2 {
        shift_by = (kv_pos - keep_prefix) / 2
    }
    kv_pos = shift_kv_cache_window(mut k_caches, mut v_caches, meta.n_layers, keep_prefix, kv_pos, shift_by)
}
```

#### Strategy 3: Interactive Context Monitoring (`/ctx` & `/reset`)
During interactive REPL sessions (`-i`), you can inspect your live KV-cache occupancy at any time with `/ctx` or wipe conversation history in $O(1)$ time with `/reset`:

```text
Interactive Qwen2.5-0.5B-Instruct CLI Session (Commands: /ctx, /reset, /exit)
User     > /ctx
[Context Window: 137 / 512 tokens used (6144 KB KV cache)]
User     > /reset
[KV Cache cleared: kv_pos reset to 0]
```

---

## Summary of Pure-Vira Capabilities Used

| Layer | Pure-Vira Feature / API | Role in CLI LLM Engine |
| :--- | :--- | :--- |
| **Model Download** | `os.exists()`, `os.mkdir_all()`, `exec.Command.new('curl', ...)` | Automatically fetches pretrained `.gguf` weights from Hugging Face |
| **Binary Ingestion** | `os.read_bytes_head()`, `os.read_bytes()`, `.slice()`, `.to_string()` | Parses GGUF v2/v3 header, dynamic `{arch}.*` metadata, and 151,936-token vocabularies |
| **Weight Loading** | `gguf_load_tensor(path, name) or { ... }`, `.linear(w)` | Loads `F32`, `Q4_0`, and `Q8_0` tensors + optional QKV biases & tied embeddings |
| **Tokenization** | SentencePiece (`▁`), GPT-2 Byte-Level BPE (`Ġ`, `Ċ`), FNV-1a Hash Index | Encodes prompts into SentencePiece subwords, ChatML (`<\|im_start\|>`), or LLaMA-3 templates |
| **Context & KV Cache** | `tensor_fill(max_context, kv_dim, 0.0)`, `.copy_row()`, `.slice_rows()`, `shift_kv_cache_window()` | Configurable `--ctx` window, $O(\text{kv\_pos})$ active slicing, and sliding-window attention sinks |
| **Transformer Math** | `.linear()`, `@`, `+`, `tensor_embedding`, `tensor_rmsnorm`, `tensor_rope`, `tensor_softmax`, `tensor_silu`, `tensor_elemwise_mul` | Executes multi-layer Grouped-Query Attention (GQA), NeoX/Interleaved RoPE, and SwiGLU FFN inside `arena {}` |
| **Logit Sampling** | `logits.get()`, `logits.set()`, `sample_token()` | Applies repetition penalty, masks control tokens, and samples next token ID |
| **CLI & REPL** | `os.args()`, `io.prompt()`, `print()`, `println()` | Parses `--model`, `--ctx`, `--steps`, `--temp`, `--system` and drives real-time streaming REPL |




