# Antigravity / Gemini Workspace Rules

Please refer to [AGENTS.md](AGENTS.md) for full repository guidelines and protocols.

### Quick Summary for Agent Turns:
1. Reusable library code lives in `src/`.
2. Executable benchmarks live in `experiments/`.
3. Results must be saved in self-contained folders: `results/<exp_id>/` (containing `metrics.json` and plots).
4. Every experiment must be logged in `EXPERIMENTS.md`.
5. Cloud GPU jobs on Modal must enforce `timeout <= 300` and confirm spin-down.
6. Pull exact numbers from `metrics.json`, never guess or hallucinate metrics.
