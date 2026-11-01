# 第三方 skills 來源（2026-11-01 引入）

這些 skill 從 GitHub 原樣複製，只刪除了 ml-paper-writing 範本裡的範例 PDF。
放在 `.claude/skills/`，在這個 repo 裡開 Claude Code（雲端或本機）就會自動載入。

| Skill | 來源 repo @ commit | 授權 |
|---|---|---|
| academic-plotting, ml-paper-writing, presenting-conference-talks, mlflow | Orchestra-Research/AI-Research-SKILLs @ 773a529 | MIT |
| statistical-analysis, scientific-visualization | K-Dense-AI/scientific-agent-skills @ 5f2bc03 | MIT |
| flower（含 sub-skills） | VectorSpaceLab/AREX-Skill @ ac3fe1af | Apache-2.0 |
| differential-privacy-prod | mukul975/Privacy-Data-Protection-Skills @ 9b2ef9e | Apache-2.0 |
| scipilot-figure-skill | Haojae/scipilot-figure-skill @ 43098dd | MIT |

## 使用注意
- `flower` 設了 `disable-model-invocation: true`，只會在輸入 `/flower` 時載入。
- `academic-plotting` 的架構圖功能會呼叫 Google Gemini（需 API key，會把描述文字送到 Google）；只用 matplotlib/seaborn 畫數據圖則不會連網。
- `ml-paper-writing` 建議加裝第三方 Exa MCP（`npx mcp-remote`），不需要就不要裝。
- `statistical-analysis` 建議在獨立 venv 安裝 pingouin/scipy/statsmodels。
