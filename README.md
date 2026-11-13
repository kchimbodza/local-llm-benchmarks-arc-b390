# Local LLMs on an Intel Arc B390 laptop: benchmarks and setup

I tested which open-source AI models are practical to run **fully offline** on a Framework Laptop 13 Pro: an Intel Arc B390 integrated GPU sharing **16GB of memory** with the CPU and desktop. Over September–October 2026 I set up llama.cpp on Vulkan, ran a set of chat, reasoning and coding prompts, and checked document OCR and image upscaling. The result is a model recommendation for each kind of task.

**Full write-up with every command and result:** [docs/guide.md](docs/guide.md)

## Key findings

- **Memory, not compute, is the limit.** The GPU sees about 11.4GB of the shared 16GB. Models over about 12GB fail with a Vulkan out-of-memory error, so model size decides what's possible.
- **Small Mixture-of-Experts models are the sweet spot.** They activate only 1–4B parameters per token, so they reach 20–35 tokens/sec on an integrated GPU. Dense models under 9B were ruled out: none would beat about 15–20 tokens/sec here.
- **Best coding model: Mellum2 Thinking.** It passed all four coding tests, with the code compiled and run, at 32–35 tokens/sec. That's about 2.5x faster than gpt-oss-20b, which made a TypeScript syntax slip and a Kotlin concurrency bug.
- **Tuning gpt-oss-20b raised prompt speed from 33 to 193 tokens/sec** (all layers on the GPU, 8 threads, a single slot, low reasoning effort). Offloading experts to the CPU, a common trick on discrete GPUs, made generation slower here because the CPU and GPU share the same RAM.
- **OCR:** DeepSeek-OCR 2 (2.5GB) read printed pages and tables accurately at about 35 seconds per page. On handwriting it read 3 of 7 lines correctly, while Gemma 4 12B's vision read all 7.
- **Power draw during testing was about 10–24W.**

| Task | Best pick | Why |
|---|---|---|
| Everyday assistant | Ling-3.0-tiny | Correct on every reasoning question, 21–29 t/s |
| Quick answers | Granite 4.0 H Tiny | Most replies in 2–3 seconds |
| Code | Mellum2 Thinking | 4/4 coding tests passed (compiled and run) |
| Hardest problems | gpt-oss-20b (medium reasoning) | Passed every chat prompt, 12–18 t/s |
| Printed PDFs and tables | DeepSeek-OCR 2 | Word-for-word on clean pages, about 35s per page |
| Handwriting and photos | Gemma 4 12B + vision | 7/7 handwritten lines correct |
| Upscaling and JPEG cleanup | Real-ESRGAN | +0.7 to +0.9 dB PSNR over a plain upscale |

## Architecture

<p align="center"><img src="docs/architecture.svg" width="100%" alt="GGUF model files (mostly 4-bit, 2.5 to 11.6 GB) are loaded by llama-server for chat and code, llama-mtmd-cli for OCR, or Real-ESRGAN for upscaling, all on the Arc B390 integrated GPU through Vulkan with -ngl 99, -t 8, -fa on and -np 1. Test prompts with known answers go in, one fresh chat each. Output is checked: speed, C++ compiled and run, TypeScript type-checked and run, Kotlin read only, OCR line by line, images scored with PSNR. Results go into docs/guide.md. Everything shares 16 GB of memory; the GPU can use about 11.4 GB and models over about 12 GB fail with a Vulkan out-of-memory error."></p>

1. **Models** are downloaded as GGUF files into `~/models` and served one at a time on port 8080 (`scripts/llm-aliases.sh` switches between them with one word).
2. **Prompts** with known answers run in a fresh chat each time, so a long context doesn't slow later runs.
3. **Checks** are done outside the model: generated code is compiled and run, OCR is compared line by line, upscaled images are scored with PSNR.
4. **Memory** is the hard limit: the model, its context and the desktop all share the GPU's ~11.4GB.

**Why this setup**

| Choice | Why |
|---|---|
| llama.cpp, Vulkan build | Runs on the Intel Arc iGPU from an AUR package. The `gfx1151`/`gfx1152` builds are for AMD and don't work here. |
| Small Mixture-of-Experts models | Only 1–4B parameters are used per token, so they reach 20–35 t/s. Dense models under 9B wouldn't beat about 15–20 t/s here. |
| 4-bit files (Q4_K_M, IQ4_NL) | The model must fit in ~11.4GB with its context. A 2-bit file was tried and lost too much quality. |
| All layers on the GPU, 8 threads, one slot | Measured: prompt speed for gpt-oss-20b went from 33 to 193 t/s. The default 2 threads were a bottleneck, and 4 slots split the memory. |
| No CPU expert offload (`--n-cpu-moe`) | The CPU and GPU share the same RAM, so it frees no space. It made generation slower here. |
| Compile and run the code | Several answers looked right but did not compile or had a concurrency bug. |

## How I tested

- **Same conditions every run.** Each prompt ran in a fresh chat, so context length didn't skew speed. Generation speed, prompt speed and wait time were recorded for each run.
- **Prompts with known answers.** The set covered summarizing, following a strict format, two reasoning traps, a practical Linux question, and four coding tasks.
- **Code was executed, not eyeballed.**
  - C++17 was compiled with `g++ -std=c++17 -Wall` and run against edge-case CSV input: quoted commas, escaped quotes, empty fields, trailing commas.
  - TypeScript was type-checked with `tsc --strict` and tested at runtime.
  - The Kotlin answers were reviewed by reading only, because no Kotlin compiler was available.
- **OCR on purpose-built test pages.** I made a three-page PDF (clean text, an invoice table, a two-column layout with a 7pt footnote) and used a real photo of a handwritten note.
- **Image upscaling scored with a metric.** A test chart was degraded three ways, upscaled 4x, and compared with the original using PSNR.
- **Limits stated in the guide.** Four coding prompts is a small sample, and results come from one machine with a fast-moving llama.cpp build.

## Results

All numbers are from [docs/guide.md](docs/guide.md), measured by hand on the test machine (September–October 2026). There are no automated tests or CI in this repo; it is a measurement write-up.

**Coding tests** (code compiled and run, except Kotlin, which was read only)

| Test | Mellum2 Instruct | Mellum2 Thinking | gpt-oss-20b (medium) | North Mini Code (2-bit) |
|---|---|---|---|---|
| TypeScript debounce | ⚠️ doesn't compile | ✅ | ⚠️ same syntax slip | ❌ |
| C++17 CSV parser | ❌ | ✅ | ✅ | ❌ |
| Kotlin coroutines (max 4 at a time) | ❌ | ✅ | ⚠️ broken limiter | ❌ |
| Bug finding | ✅ | ✅ | ✅ | ✅ |
| Generation speed | 33–36 t/s | **32–35 t/s** | 12–18 t/s | 16–20 t/s |

**Tuning gpt-oss-20b** (11.6GB, slightly more than the GPU's free memory)

| Setup | Prompt speed | Generation speed |
|---|---|---|
| `-ngl 99`, default threads, 4 slots | 33 t/s | 15 t/s (very short chat) |
| Auto-fit (no `-ngl`) | 22 t/s | 10 t/s |
| `-ngl 99 --n-cpu-moe 4 -t 8` | 86 t/s | 10.6 t/s |
| `-ngl 99 -t 8 -np 1` | 181 t/s | 11.2 t/s |
| Same + `reasoning_effort: low` | **193 t/s** | **12.2 t/s** |

**Image upscaling** (PSNR against the original, higher is closer): Real-ESRGAN beat a plain 4x upscale by +0.70 dB (clean), +0.92 dB (noise + JPEG) and +0.32 dB ("old photo").

## What worked and what didn't

| | What | Detail |
|---|---|---|
| ✅ | Small MoE models | Ling-3.0-tiny got every reasoning question right at 21–29 t/s. Granite 4.0 H Tiny answered most prompts in 2–3 s. |
| ✅ | Mellum2 Thinking for code | 4/4 coding tests, about 2.5x faster than gpt-oss-20b. |
| ✅ | Tuning flags | `-ngl 99 -t 8 -np 1` plus low reasoning took gpt-oss-20b's prompt speed from 33 to 193 t/s. |
| ✅ | DeepSeek-OCR 2 on printed pages | Clean text word for word; invoice line items and totals correct; 7pt footnote captured. |
| ❌ | Gemma 4 26B-A4B (12.7GB) | Failed to load: Vulkan out-of-memory. |
| ❌ | CPU expert offload | Faster prompts (86 t/s) but slower generation (10.6 t/s). |
| ❌ | 2-bit North Mini Code | Failed 3 of 4 coding tests. |
| ❌ | DeepSeek-OCR 2 on handwriting | 3 of 7 lines correct; Gemma 4 12B's vision got 7/7. Wrong reads look like real words ("Agency's tasks"). |
| ❌ | DRY anti-repetition flags for OCR | Mangled repeated model names (`LFM2.5-8BB-A1B`); removing them fixed it. |
| ❌ | Gemma 4 12B with thinking on | A four-line poem took 7 minutes. |
| ❌ | `realesrgan-ncnn-vulkan-git` (AUR) | Failed to build; the prebuilt release worked. |
| ⚠️ | Limits | Four coding prompts is a small sample, Kotlin was not compiled, and all results come from one machine with a fast-moving llama.cpp build. |

## Skills shown

- **Linux systems work:** Arch Linux / Omarchy, Vulkan driver setup, AUR packages, diagnosing out-of-memory and build failures.
- **LLM inference:** llama.cpp, model quantization (Q4_K_M, IQ4_NL, 2-bit), context and KV-cache sizing, sampling settings, multimodal models.
- **Performance tuning:** GPU layer offload, threads, parallel slots, and how shared memory changes the usual advice.
- **Evaluation:** repeatable prompts, pass/partial/fail scoring, compiling and running generated code, PSNR for image quality.
- **Technical writing:** a guide someone else can follow from install to model choice, including a troubleshooting section built from real errors.

## Repository

| Path | Contents |
|---|---|
| [docs/guide.md](docs/guide.md) | The full guide: setup, downloads, launch commands, test prompts, all results, troubleshooting, upgrade path |
| [scripts/llm-aliases.sh](scripts/llm-aliases.sh) | One-word shortcuts to switch the local model server between the tested models |
| [docs/architecture.svg](docs/architecture.svg) | The test setup diagram used above |

## Test machine

Framework Laptop 13 Pro · Intel Core Ultra X7 (Panther Lake) · Intel Arc B390 (Xe3, integrated) · 16GB LPCAMM2 LPDDR5X-7467, shared · Omarchy 4 (Arch Linux + Hyprland) · llama.cpp Vulkan build
