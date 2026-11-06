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

## Test machine

Framework Laptop 13 Pro · Intel Core Ultra X7 (Panther Lake) · Intel Arc B390 (Xe3, integrated) · 16GB LPCAMM2 LPDDR5X-7467, shared · Omarchy 4 (Arch Linux + Hyprland) · llama.cpp Vulkan build
