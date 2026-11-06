# One-word shortcuts for switching the local llama.cpp server between the models tested in
# docs/guide.md. Source this from ~/.bashrc:   source /path/to/llm-aliases.sh
# Each one stops the running server, then starts the model with its web UI on http://localhost:8080.
# Models are expected under ~/models (see "Download the models" in the guide).

LLAMA_COMMON="--jinja -ngl 99 -t 8 -fa on -np 1 --port 8080"

# Everyday assistant
alias ling="pkill llama-server; llama-server -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf $LLAMA_COMMON -c 16384"
# Fastest answers (no thinking step)
alias granite="pkill llama-server; llama-server -m ~/models/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf $LLAMA_COMMON -c 16384 --temp 0.0 --top-p 1.0 --top-k 0"
# Fastest writing: summaries, rewrites, drafts
alias lfm="pkill llama-server; llama-server -m ~/models/lfm25-8b-a1b-GGUF/lfm25-8b-a1b-Q4_K_M.gguf $LLAMA_COMMON -c 16384 --temp 0.2 --top-k 80 --repeat-penalty 1.05"
# Best for code
alias mellum="pkill llama-server; llama-server -m ~/models/Mellum2-Thinking/*.gguf $LLAMA_COMMON -c 16384"
# Most capable; 8K context to fit in memory
alias gptoss="pkill llama-server; llama-server -m ~/models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf $LLAMA_COMMON -c 8192 --temp 1.0 --top-p 1.0 --top-k 0 --chat-template-kwargs '{\"reasoning_effort\":\"low\"}'"
