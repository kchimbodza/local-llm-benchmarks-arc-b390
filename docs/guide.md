# Local AI on the Framework Laptop 13 Pro

*A hands-on guide to running LLMs locally on the Intel Arc B390 with 16GB of shared memory: setup, copy-paste launch commands, benchmark results, and which model to pick.*

*Tested September–October 2026 on Omarchy 4 with llama.cpp (Vulkan backend). Last updated November 2026.*

---

## TL;DR

- **16GB of shared memory is the real limit.** The Arc B390 has no memory of its own, so the GPU can use about 11.4GB, and the model, its context, and your desktop all compete for it.
- **Small MoE (Mixture-of-Experts) models run best.** They only use 1–4B parameters per token, so they're fast on an integrated GPU. Dense models under 9B were ruled out: none would beat about 15–20 tokens/sec here.
- **Best everyday picks:** Ling-3.0-tiny (fast and smart) or Granite 4.0 H Tiny (quickest answers).
- **Best for coding:** Mellum2 Thinking passed all four coding tests at 32–35 tokens/sec, about 2.5x faster than gpt-oss-20b.
- **Best for documents:** DeepSeek-OCR 2 (3.4B MoE, 2.5GB) reads printed PDFs and tables accurately at about 35 seconds per page. For handwriting and photos, Gemma 4 12B's vision was far more accurate.
- **Image restoration:** Real-ESRGAN (not an LLM) runs on the Arc GPU. It cleanly upscales low-resolution images and removes JPEG noise, but it can't repair old photos and it garbles small text and numbers.
- **Hardest tasks:** gpt-oss-20b is the most capable model that fits, at about 11–18 tokens/sec depending on context length.
- **Power draw while testing: roughly 10–24W.**

---

## Test machine

| Component | Spec |
|---|---|
| Laptop | Framework Laptop 13 Pro |
| CPU | Intel Core Ultra X7 (Panther Lake) |
| GPU | Intel Arc B390 (integrated, Xe3) |
| Memory | 16GB LPCAMM2 LPDDR5X-7467 (shared between CPU and GPU) |
| GPU-visible memory | 11,665 MiB total (about 10,500 MiB free at idle) |
| OS | Omarchy 4 (Arch Linux + Hyprland) |
| Inference engine | llama.cpp, Vulkan build (`llama.cpp-vulkan-git` from the AUR) |
| Power draw during tests | about 10–24W |

### Why memory matters more than anything else

On a laptop with integrated graphics, the GPU borrows system RAM. On this machine Vulkan reports about **11.4GB usable by the GPU**, and roughly 10.25GB of that is free once the desktop is running. The model file, the context (KV cache), and every open app come out of the same pool.

Rule of thumb for 16GB:

| Model file size | Experience |
|---|---|
| Under 7GB | Comfortable. Room for long context and a browser. |
| 7–10GB | Fine if you keep other apps light. |
| 10–12GB | Tight. Close the browser, keep context at 8K. |
| Over 12GB | Won't fit well. Gemma 4 26B-A4B (12.7GB) failed to load with a Vulkan out-of-memory error. |

**A note on CPU offloading.** Moving MoE expert layers to the CPU (`--n-cpu-moe`) is a popular trick for GPUs with their own memory, such as an RTX 4050. On the B390 the CPU and GPU share the same RAM, so offloading frees no extra space. Here it made prompts faster but generation slower (see the gpt-oss-20b tuning table).

---

## Setup

### 1. Install the Intel Vulkan driver and llama.cpp

```bash
sudo pacman -S vulkan-intel vulkan-icd-loader vulkan-tools
yay -S llama.cpp-vulkan-git
```

When yay asks which provider to install, choose **`llama.cpp-vulkan-git`**. The `gfx1151` / `gfx1152` builds are for AMD Strix Halo and won't work on Intel.

### 2. Confirm the GPU is detected

```bash
llama-server --list-devices
```

You should see something like:

```
Vulkan0: Intel(R) Arc(tm) B390 (PTL) (11665 MiB, 10498 MiB free)
```

### 3. Install the Hugging Face download tool

```bash
yay -S python-huggingface-hub
```

### 4. Tools for the document and image tests (optional)

```bash
sudo pacman -S poppler imagemagick
```

`poppler` turns PDF pages into images. `imagemagick` converts WebP images, which llama.cpp can't read.

---

## Download the models

Everything goes into `~/models`. The core set is about 52GB, so check free space first with `df -h ~`.

```bash
# Small, fast MoE models
hf download AlexAtomic/lfm25-8b-a1b-GGUF       --include "*Q4_K_M*" --local-dir ~/models/lfm25-8b-a1b-GGUF
hf download Mike0021/Ling-3.0-tiny-GGUF        --include "*Q4_K_M*" --local-dir ~/models/Ling-3.0-tiny-GGUF
hf download ibm-granite/granite-4.0-h-tiny-GGUF --include "*Q4_K_M*" --local-dir ~/models/granite-4.0-h-tiny-GGUF

# Coding MoE models
hf download JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M --include "*.gguf" --local-dir ~/models/Mellum2-Instruct
hf download JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M --include "*.gguf" --local-dir ~/models/Mellum2-Thinking

# Larger models
hf download unsloth/gpt-oss-20b-GGUF           --include "*Q4_K_M*" --local-dir ~/models/gpt-oss-20b-GGUF
hf download unsloth/gemma-4-12b-it-qat-GGUF    --include "*.gguf"   --local-dir ~/models/gemma-4-12b-it-qat-GGUF

# Document OCR (MoE, runs through llama-mtmd-cli)
hf download SandLogicTechnologies/DeepSeek-OCR-2-GGUF --include "*IQ4_NL*" --include "*mmproj*" --local-dir ~/models/DeepSeek-OCR-2
```

Optional, only to repeat the 2-bit test (not recommended for real use, adds 9.8GB):

```bash
hf download unsloth/North-Mini-Code-1.0-GGUF --include "*UD-IQ2_XXS*" --local-dir ~/models/North-Mini-Code
```

Check the real filenames after downloading:

```bash
command ls -lh ~/models/*/
```

| Model | Type | Active params | File size |
|---|---|---|---|
| LFM2.5-8B-A1B | MoE | ~1.5B | 5.2GB |
| Ling-3.0-tiny | MoE | 1.3B | 4.8GB |
| Granite 4.0 H Tiny | Hybrid MoE (Mamba) | 1B | 4.2GB |
| Mellum2 12B-A2.5B (Instruct and Thinking) | MoE | 2.5B | 8.1GB each |
| Gemma 4 12B (QAT) | Dense | 12B | 6.7GB (+ optional vision file) |
| gpt-oss-20b | MoE | 3.6B | 11.6GB |
| DeepSeek-OCR 2 (document OCR) | MoE (3.4B total) | not checked | 2.5GB (1.6GB model + 0.9GB vision file) |
| North Mini Code 1.0 (2-bit, optional) | MoE (30B total) | 3B | 9.8GB |

---

## Launch commands

Every command below uses the same core flags:

| Flag | What it does |
|---|---|
| `--jinja` | Uses the model's built-in chat template (required for gpt-oss) |
| `-ngl 99` | Puts all layers on the GPU |
| `-t 8` | Uses 8 CPU threads (the default of 2 is a bottleneck) |
| `-c 16384` | 16K context (use `8192` for gpt-oss and North Mini Code to save memory) |
| `-fa on` | Flash attention, which lowers memory use |
| `-np 1` | One chat slot, so memory isn't split four ways |
| `--port 8080` | Web UI at `http://localhost:8080` |

Only one model can use port 8080 at a time. Stop the running one first:

```bash
pkill llama-server
```

### Ling-3.0-tiny (recommended daily driver)

```bash
llama-server -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf \
  --jinja -ngl 99 -t 8 -c 16384 -fa on -np 1 --port 8080
```

### Granite 4.0 H Tiny (fastest answers)

```bash
llama-server -m ~/models/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf \
  --jinja -ngl 99 -t 8 -c 16384 -fa on -np 1 --port 8080 \
  --temp 0.0 --top-p 1.0 --top-k 0
```

### LFM2.5-8B-A1B (fastest writing)

```bash
llama-server -m ~/models/lfm25-8b-a1b-GGUF/lfm25-8b-a1b-Q4_K_M.gguf \
  --jinja -ngl 99 -t 8 -c 16384 -fa on -np 1 --port 8080 \
  --temp 0.2 --top-k 80 --repeat-penalty 1.05
```

### Mellum2 Thinking (best for coding)

The `*.gguf` pattern picks up the single file in the folder, so you don't need the exact filename.

```bash
llama-server -m ~/models/Mellum2-Thinking/*.gguf \
  --jinja -ngl 99 -t 8 -c 16384 -fa on -np 1 --port 8080
```

For the faster but less careful Instruct version, swap the folder to `~/models/Mellum2-Instruct/`.

### gpt-oss-20b (most capable)

```bash
llama-server -m ~/models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf \
  --jinja -ngl 99 -t 8 -c 8192 -fa on -np 1 --port 8080 \
  --temp 1.0 --top-p 1.0 --top-k 0 \
  --chat-template-kwargs '{"reasoning_effort":"low"}'
```

Change `low` to `medium` or `high` for harder problems. It will think longer before answering. `medium` fixed the small slips seen at `low` with similar wait times.

### Gemma 4 12B (accurate but slow here)

```bash
llama-server -m ~/models/gemma-4-12b-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --jinja -ngl 99 -t 8 -c 16384 -fa on -np 1 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 64 --reasoning-budget 0
```

`--reasoning-budget 0` turns thinking off. With thinking on, a four-line poem took 7 minutes on this machine. To read images, add `--mmproj ~/models/gemma-4-12b-it-qat-GGUF/mmproj-F16.gguf` (see the OCR section).

### North Mini Code 1.0 (2-bit, not recommended)

```bash
llama-server -m ~/models/North-Mini-Code/North-Mini-Code-1.0-UD-IQ2_XXS.gguf \
  --jinja -ngl 99 -t 8 -c 8192 -fa on -np 1 --port 8080 \
  --temp 1.0 --top-p 0.95
```

### One-word shortcuts

Add these to `~/.bashrc`, then run `source ~/.bashrc`:

```bash
LLAMA_COMMON="--jinja -ngl 99 -t 8 -fa on -np 1 --port 8080"

alias ling="pkill llama-server; llama-server -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf $LLAMA_COMMON -c 16384"
alias granite="pkill llama-server; llama-server -m ~/models/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf $LLAMA_COMMON -c 16384 --temp 0.0 --top-p 1.0 --top-k 0"
alias lfm="pkill llama-server; llama-server -m ~/models/lfm25-8b-a1b-GGUF/lfm25-8b-a1b-Q4_K_M.gguf $LLAMA_COMMON -c 16384 --temp 0.2 --top-k 80 --repeat-penalty 1.05"
alias mellum="pkill llama-server; llama-server -m ~/models/Mellum2-Thinking/*.gguf $LLAMA_COMMON -c 16384"
alias gptoss="pkill llama-server; llama-server -m ~/models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf $LLAMA_COMMON -c 8192 --temp 1.0 --top-p 1.0 --top-k 0 --chat-template-kwargs '{\"reasoning_effort\":\"low\"}'"
```

Now switching models is one word: `ling`, `granite`, `lfm`, `mellum` or `gptoss`.

---

## The test prompts

Each prompt was run in a fresh chat so context length didn't skew speed. Copy them to test your own setup.

**1. Summarize and rewrite**

```
Summarize this in 3 bullet points, then rewrite it as a friendly 2-sentence text message:

"Hi team, due to the scheduled maintenance on the main server this Friday from 6pm to 11pm, all shared drives and the internal wiki will be unavailable. Please save any work you need locally before 5:30pm. The VPN will still work, but email may be delayed by up to an hour. If you have urgent issues during the outage, contact Sam on the on-call phone at extension 4412."
```

**2. Following instructions**

```
Write a 4-line poem about a laptop overheating. Rules: every line must start with the letter S, don't use the word "hot", and end with a question.
```

**3. Reasoning** (correct answer: 5 minutes)

```
If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets? Explain your reasoning briefly.
```

**4. Practical Linux**

```
On Arch Linux, how do I find which process is using port 8080 and stop it? Give the commands only, with a one-line explanation each.
```

**5. Trick question** (correct answer: $0.05)

```
A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer in one line.
```

**6–9. Coding tests** (see the Coding tests section for results)

- **TypeScript:** a typed `debounce` function with a `cancel()` method that compiles under `tsc --strict`.
- **C++17:** a CSV parser that handles commas inside quotes, escaped quotes and empty fields.
- **Kotlin:** fetch a list of URLs with coroutines, at most 4 at a time, keeping results in order.
- **Bug finding:** spot and fix an off-by-one (`<=` instead of `<`) in a short loop.

---

## Results

✅ pass · ⚠️ partial · ❌ fail

| Model | Generation speed | Typical wait | Summary | Poem | Widgets | Linux | Bat/ball |
|---|---|---|---|---|---|---|---|
| LFM2.5-8B-A1B | ~35 t/s | 9–21s | ✅ | ✅ | ❌ | ✅ | not run |
| Ling-3.0-tiny | ~21–29 t/s | 7–37s | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Granite 4.0 H Tiny | ~16–20 t/s | **0.4–11s** | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Gemma 4 12B (thinking off) | ~5–7 t/s | 2–31s | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Gemma 4 12B (thinking on) | ~5–9 t/s | 55s–7min | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt-oss-20b (low reasoning) | ~6–12 t/s | 9–30s | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt-oss-20b (medium reasoning) | ~12–18 t/s | 9–48s | ✅ | ✅ | ✅ | ✅ | ✅ |
| North Mini Code 1.0 (2-bit) | ~16–20 t/s | 13s | not run | not run | not run | ✅ | not run |

### Notes on each model

**LFM2.5-8B-A1B** was the fastest generator by a wide margin, at about 35 tokens/sec with prompt processing near 480 tokens/sec. Its writing was clean and complete. It failed the widget question, confidently answering 1 minute, so don't trust it with logic or maths.

**Ling-3.0-tiny** was the best all-rounder. It got every reasoning question right and followed the poem rules perfectly. Its text message dropped some details, and it can overthink simple tasks (1,100 tokens of thinking for a four-line poem).

**Granite 4.0 H Tiny** doesn't think before answering, so most replies arrived in 2–3 seconds. It was accurate on everything, with one small invented detail (it assumed Sam's gender). Its raw generation speed was lower than expected, likely because its Mamba layers aren't fully optimized on Vulkan yet.

**Mellum2 (JetBrains)** comes in Instruct and Thinking versions. Thinking was the best coder tested here (see below). Instruct is just as fast but drops hard requirements.

**Gemma 4 12B** gave the most polished answers but is a dense model, so all 12B parameters are used for every token. At 5–9 tokens/sec it's too slow for daily use on this hardware. Turning thinking off helps a lot with wait time, at some cost to quality. Its vision was the best handwriting reader tested.

**gpt-oss-20b** passed every chat prompt and wrote the best poem. At `medium` reasoning it fixed the weak spots seen at `low` (the poem now ends every line in a question, and the text message is exactly two sentences) at similar wait times, and it ran at 12–18 tokens/sec in short chats. On coding it did not beat Mellum2 Thinking: it was about 2.5x slower and made a TypeScript syntax slip and a Kotlin concurrency mistake.

**North Mini Code 1.0** (Cohere, 30B total / 3B active, built for agentic coding) only fits at 2-bit, using the 9.8GB UD-IQ2_XXS file. It generated at 16–20 tokens/sec but failed three of the four coding tests, with the usual signs of heavy compression damage. Cohere's benchmark scores are for the full-precision model, so a 4-bit test on a 32GB machine would be the fair one.

---

## Coding tests

The code was checked, not just read. The C++ was compiled with `g++ -std=c++17 -Wall` and run against a CSV with quoted commas, escaped quotes, empty fields and a trailing comma. The TypeScript was checked with `tsc --strict` and a runtime test. Kotlin was read only, because no Kotlin compiler was available.

| Test | Mellum2 Instruct | Mellum2 Thinking | gpt-oss-20b (medium) | North Mini Code (2-bit) |
|---|---|---|---|---|
| TypeScript debounce | ⚠️ Logic is right, but it doesn't compile: `as T & {...}` sits directly after the arrow function. Wrapping the arrow function in parentheses fixes it. | ✅ Compiles under `tsc --strict`. A burst of three calls fires once, and `cancel()` works. | ⚠️ The same syntax error. The parentheses fix makes it pass `tsc --strict`. | ❌ Three compile errors even without `--strict`. |
| C++17 CSV | ❌ Split `"Paris, France"` into two fields. | ✅ Quoted commas, escaped quotes, empty fields and trailing commas all parse correctly. | ✅ Identical behavior to Mellum2 Thinking. | ❌ Splits on spaces instead of commas. |
| Kotlin coroutines | ❌ No 4-at-a-time limit, and the fetch is faked with `delay(1000)`. | ✅ `Semaphore(4)` with `async` and `awaitAll` gives the limit and keeps order. The fetch is a stub. | ⚠️ Real Ktor fetch with ordered results, but the limiter is broken: it imports the blocking Java `Semaphore` and its `withPermit` helper calls itself. | ❌ Invalid Swift-style syntax and no real limit. |
| Bug finding | ✅ Found the `<=` off-by-one, explained it, and fixed it. | ✅ Correct fix, but no explanation. | ✅ Explained, fixed, and guarded empty input. | ✅ Explained, fixed, and added an empty-array guard. |
| Generation speed | 33–36 t/s | 32–35 t/s | 12–18 t/s | 16–20 t/s |
| Typical wait | 4–10s | 11s–1m14s | 39s–2m8s | 21–39s |

**Takeaway:**

- Mellum2 Thinking was the best coder on this laptop: correct on all four tests, about 2.5x faster than gpt-oss-20b.
- Mellum2 Instruct is fast but drops hard requirements.
- gpt-oss-20b matched Thinking on C++ but made the same TypeScript slip and a subtle Kotlin mistake.
- North Mini Code at 2-bit isn't usable for code.
- **Always compile model code before trusting it.** Four prompts is a small sample, and gpt-oss was only tried at medium reasoning.

---

## Document and image tests (OCR)

**DeepSeek-OCR 2** is a 3.4B-parameter MoE built to read documents. The model plus its vision file is only 2.5GB. It runs through `llama-mtmd-cli` on page images rather than through `llama-server`.

### Test files

The test used a three-page PDF: (1) clean text with numbers, an email and a URL, (2) an invoice with a five-row table and totals, (3) a two-column page with a small table and a 7pt footnote. Use any PDF of your own, and turn it into page images first:

```bash
pdftoppm -png -r 150 yourfile.pdf page      # makes page-1.png, page-2.png, ...
```

### Command

```bash
D=~/models/DeepSeek-OCR-2
llama-mtmd-cli -m $D/DeepSeek-OCR-2-IQ4_NL.gguf \
  --mmproj $D/mmproj-deepseek-ocr-2-bf16.gguf \
  --image page-1.png -p "Free OCR." \
  --chat-template deepseek-ocr --no-jinja --temp 0 --flash-attn off --no-warmup \
  -t 8 -n 2048
```

The model card also suggests `--dry-*` anti-repetition flags. Leave them out for normal pages. With them on, model names that repeat in a list came out mangled (`LFM2.5-8BB-A1B`); with them off the same page was exact. Add them back only if a long table starts looping.

### Printed pages

| Page | Result |
|---|---|
| 1. Clean text | ✅ Word for word, including `11,665 MiB`, the numbered list, the email address and the URL. It adds markdown headings. |
| 2. Invoice | ⚠️ All five line items and the totals ($456.48 / $59.34 / $515.82) are correct, and the table kept its structure. It dropped the large "INVOICE" title, and the totals came out as label-then-value lines instead of table rows. |
| 3. Two columns | ✅ With DRY off, the columns stayed separate, the table is right, and the 7pt footnote was captured. |

**Speed:** about 35 seconds per page with `-t 8` (42 seconds without it). Almost all of that is image encoding, because some vision operations fall back to the CPU on Vulkan.

### Handwriting

The test image was a photo of a handwritten note with seven numbered lines, taken at an angle with a blurry background and an app toolbar. Both models ran on the same PNG.

| Line on the note | DeepSeek-OCR 2 | Gemma 4 12B (vision) |
|---|---|---|
| 1. Front-end: Html - UI Design - Layout | ✅ | ✅ |
| 2. Agentic tasks | ❌ "Agency's tasks" | ✅ |
| 3. Pdf's, dataset extraction, Pictures | ❌ "Paths, datastore exportation" | ✅ |
| 4. Video generation | ✅ | ✅ |
| 5. Image generation | ✅ | ✅ |
| 6. Back-end Coding TS, JS, C, C++, Kotlin e.t.c | ⚠️ Read TS as JS | ✅ (`e.t.c` became `etc.`) |
| 7. Vector graphics, Composition, Visual Detail | ❌ "Couponts, Icones" | ✅ (separators became dashes) |

Both took about 16 seconds. Gemma 4 12B ran in `llama-server` with its vision file, with the image attached in the web UI:

```bash
G=~/models/gemma-4-12b-it-qat-GGUF
llama-server -m $G/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --mmproj $G/mmproj-F16.gguf \
  --jinja -ngl 99 -t 8 -c 16384 -fa on -np 1 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 64 --reasoning-budget 0
```

Prompt used:

```
Transcribe all the handwriting in this image exactly as written, as a numbered list. Ignore the app toolbar and the background.
```

Gemma tidied capitalization and spelling, so "exactly as written" wasn't fully literal. It also left out the note's title, which the prompt didn't ask for.

**Takeaway:** use DeepSeek-OCR 2 for printed pages, tables and PDFs. For handwriting or photos, a general vision model (Gemma 4 12B, which is dense rather than MoE) was far more accurate. OCR mistakes on handwriting look like real words ("Agency's," "Paths"), so always read them through.

---

## Image restoration (Real-ESRGAN)

Image restoration isn't an LLM job, and the good options aren't MoE. **Real-ESRGAN** is a small upscaler that runs through Vulkan, so it works on the Arc GPU (its log shows the B390 detected with fp16 support). The strongest open restorer found, RealRestorer (March 2026), needs about 34GB of GPU memory. SeedVR2 runs through ComfyUI and PyTorch with no Intel testing. Neither is a practical fit here.

### Install

The AUR package `realesrgan-ncnn-vulkan-git` failed to build here. Use the prebuilt release instead, which includes the program and the models:

```bash
mkdir -p ~/realesrgan && cd ~/realesrgan
curl -L -o rg.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip -o rg.zip && chmod +x realesrgan-ncnn-vulkan
```

### Run

```bash
R=~/realesrgan/realesrgan-ncnn-vulkan
M=~/realesrgan/models
$R -i small.png -o big.png -n realesrgan-x4plus -s 4 -m $M
```

`realesrgan-x4plus` is the general photo model (33MB). The `-anime` and `realesr-animevideov3` models are tuned for anime and video frames.

### The test

A 1024x768 test chart (text at five sizes, stripes, a resolution star, a landscape, colour patches and a texture) was shrunk to 256x192 in three ways, then upscaled 4x. Scores are PSNR against the original (higher is closer):

| Image | Plain upscale | Real-ESRGAN | Gain |
|---|---|---|---|
| a. Low-res, clean | 19.15 dB | **19.85 dB** | +0.70 |
| b. Noise + heavy JPEG | 18.17 dB | **19.09 dB** | +0.92 |
| c. "Old photo" (blur, fade, tint, scratches) | 13.90 dB | **14.22 dB** | +0.32 |

Each image finished in a few seconds. The visual cleanup on (a) and (b) is larger than the numbers suggest, but small text and numbers can change.

**Takeaway:** good for upscaling clean low-res images and removing JPEG noise. Don't use it on documents (run OCR on the original instead) or on damaged old photos, where scratches, fade and blur stay.

---

## Which model should you pick?

| If you want... | Use |
|---|---|
| A fast, smart everyday assistant | **Ling-3.0-tiny** |
| Answers in seconds for commands and quick facts | **Granite 4.0 H Tiny** |
| Fast summaries, rewrites and drafts | **LFM2.5-8B-A1B** |
| Code (TypeScript, C++, Kotlin, bug fixing) | **Mellum2 Thinking** |
| The best answer on a hard problem | **gpt-oss-20b** (`medium` reasoning) |
| Reading printed PDFs and tables | **DeepSeek-OCR 2** (about 35s per page) |
| Reading handwriting or photos | **Gemma 4 12B** with its `mmproj` file (about 16s for a short note) |
| Upscaling a clean low-res image or removing JPEG noise | **Real-ESRGAN** `realesrgan-x4plus` (not an LLM) |

---

## Best model by use case

**Tested** means it was run in this guide. **Candidate** means it looks promising on paper but hasn't been tested on this machine.

| Use case | Tested | Candidates (untested) |
|---|---|---|
| Front-end: HTML, UI design, layout | None yet | Vision MoEs like Gemma 4 26B-A4B and Qwen3.6-35B-A3B need a 32GB upgrade (26B-A4B failed to load at 16GB). |
| Agentic tasks (tool calling) | None yet | North Mini Code was built for this, but at 2-bit it wasn't usable. |
| PDFs, data extraction, pictures | **DeepSeek-OCR 2** for printed pages. **Gemma 4 12B** for handwriting. | Unlimited-OCR (3B MoE) needs newer llama.cpp support. |
| Image restoration and upscaling | **Real-ESRGAN** for clean upscaling and JPEG noise. | SeedVR2 (ComfyUI, Intel untested). RealRestorer needs about 34GB, so it won't fit. |
| Video generation | None | No MoE options found that fit in 16GB. |
| Image generation | None | No MoE options. FLUX.2-klein-4B (dense) runs on stable-diffusion.cpp. |
| Back-end coding (TS, JS, C, C++, Kotlin) | **Mellum2 Thinking** passed all four tests. | **Mellum2.1-12B-A2.5B Thinking** (JetBrains, October 2026): same size, reported big gains in coding and agent tests. |
| Vector graphics, composition | None yet | No dedicated small MoE. A vision model could write SVG and check its own render. |
| General chat and writing | Ling-3.0-tiny, Granite 4.0 H Tiny, LFM2.5-8B-A1B | **Nemotron 3.5 Bolt E4B** (NVIDIA, October 2026): about 4GB, Apache 2.0 licence. |
| Reasoning and maths | gpt-oss-20b, Ling-3.0-tiny | |

---

## Tuning gpt-oss-20b: what worked and what didn't

gpt-oss-20b is 11.6GB, slightly more than the GPU's free memory, so it needed tuning. All runs used the same model file.

| Setup | Prompt speed | Generation speed | Verdict |
|---|---|---|---|
| Unsloth Studio (default) | n/a | ~9 t/s | Ran mostly on CPU with disk paging |
| `-ngl 99`, default threads, 4 slots | 33 t/s | 15 t/s* | GPU partly used |
| Auto-fit (no `-ngl`) | 22 t/s | 10 t/s | Slower: left too much on CPU |
| `-ngl 99 --n-cpu-moe 4 -t 8` | 86 t/s | 10.6 t/s | Faster prompts, slower generation |
| `-ngl 99 -t 8 -np 1` | 181 t/s | 11.2 t/s | Best balance |
| Same + `reasoning_effort: low` | 193 t/s | **12.2 t/s** | **Best overall** |

\*Measured on a very short chat. Generation speed drops as context grows, so it isn't directly comparable.

The winning combination was all layers on the GPU, 8 threads, a single slot, and low reasoning effort. Low reasoning doesn't make tokens faster, but it cut one answer from 526 tokens to 209, halving the wait.

---

## Troubleshooting

**`couldn't bind HTTP server socket ... port: 8080`.** Another llama-server is still running. Run `pkill llama-server` or use `--port 8081`.

**`failed to fit params to free device memory: n_gpu_layers already set by user`.** This is harmless. llama.cpp skipped its automatic sizing because you set `-ngl` yourself.

**`ggml_vulkan: ErrorOutOfDeviceMemory`.** This is a real out-of-memory error. The model is too big for the GPU's share of RAM. Use a smaller file, lower `-c`, or close other apps. Models over about 12GB (such as Gemma 4 26B-A4B at 12.7GB) didn't load here.

**`unknown model architecture` (for example `cohere2moe`).** Your llama.cpp is too old for that model. Update with `yay -Syu llama.cpp-vulkan-git`.

**A download fails with a lock-file error.** A previous `hf download` was interrupted. Run the same command again; it resumes.

**Code copied from the chat UI won't compile.** Angle brackets in generics and templates can get mangled on copy. Compare against the model's actual output before blaming the model.

**`llama-mtmd-cli` prints `usage:` and the output file is empty.** A flag was rejected. This build doesn't accept `--dry-penalty-last-n -1`. Drop the `--dry-*` flags or use a positive number such as `2048`.

**OCR output has mangled words that repeat, such as model names.** The DRY repetition penalty is fighting legitimate repeats. Remove the `--dry-*` flags.

**`llama-mtmd-cli` can't read my image.** WebP isn't supported. Convert it with `magick note.webp note.png`.

**A wall of `CLIP graph uses unsupported operators ... backend=Vulkan0` warnings.** These are harmless. Some image-encoding steps fall back to the CPU, which is why `-t 8` helps.

**`yay -S realesrgan-ncnn-vulkan-git` fails during the CMake step.** Skip the AUR package and use the prebuilt release from the Image restoration section.

**`realesrgan-ncnn-vulkan: command not found`.** It isn't on your PATH. Call it by full path (`~/realesrgan/realesrgan-ncnn-vulkan`) and pass `-m ~/realesrgan/models`.

**Speed drops mid-answer, then recovers.** This is usually heat or memory pressure. Plug in, run `powerprofilesctl set performance`, close the browser, and watch `btop` for swap use or temperatures near 90°C and above.

**Everything is slow and the GPU isn't used.** Check that `llama-server --list-devices` shows `Vulkan0`. If not, install `vulkan-intel` and confirm with `vulkaninfo --summary`.

**`ls -lhL` fails with `invalid value for '--level'`.** Omarchy aliases `ls` to `eza`, where `-L` means depth. Use `command ls -lhL` instead.

**Answers get slower over a long chat.** The whole conversation is resent with every message. Start a new chat for each new topic.

---

## Upgrade path

LPCAMM2 memory is replaceable. The Intel board takes up to 64GB (the 96GB option is only for the AMD board).

| Memory | GPU can use about | What it opens up |
|---|---|---|
| 16GB (tested) | 11.4GB | Everything in this guide |
| 32GB | 23GB | Qwen3.6-35B-A3B (about 21GB at 4-bit), Gemma 4 26B-A4B (about 17GB at 4-bit), North Mini Code at 4-bit |
| 64GB | 46–48GB | Larger MoE models at 4-bit, or mid-size ones with long context |

More memory lets bigger models fit, but **it won't make tokens faster.** Generation speed is set by memory bandwidth, which stays the same.

---

*Results are from one machine over a few weeks with a fast-moving llama.cpp build. Your numbers will vary with drivers, thermals, power mode and what else is running.*
