# Learning journal

This is an append-only log, newest entry first. Write one entry per session (it takes about 5 minutes).

## Mental map
Redraw this weekly in your own words. Each box is filled in by the versions listed under it.

```mermaid
flowchart LR
    C["Client<br/>mic, playback, approval card<br/><i>v4 CLI, v5 browser</i>"] -->|WebSocket| G["App: gateway + turn manager<br/><i>v5</i>"]
    G --> ASR["ASR<br/><i>v4, v8 Triton, v13 LoRA, v14 TRT</i>"]
    G --> A["Agent<br/>tools, approvals<br/><i>v1, v6</i>"]
    A --> LLM["LLM server<br/>OpenAI API<br/><i>v0 MLX, v9 vLLM, v10 quant, v11/v13 LoRA</i>"]
    A --> M["Memory<br/>embed + pgvector<br/><i>v2, v8, v14 TRT</i>"]
    A --> T["Tools / MCP<br/><i>v1, v6</i>"]
    G --> TTS["TTS + normaliser<br/><i>v3, v8</i>"]
    M --> DB[("Postgres<br/>memories, approvals, audit<br/><i>v1, v2, v6, v7b</i>")]
    A --> DB
    E["Eval + observability<br/><i>v7a, v7b</i>"] -.-> LLM
    E -.-> ASR
```

## Still don't get (re-ask after 1 week and again after 4 weeks)
| Added | Question | Re-asked (1 week) | Re-asked (4 weeks) |
|---|---|---|---|

## Entry template
```markdown
### YYYY-MM-DD: <version> session N (learn | build | interview)
- **Built or watched:**
- **Predicted vs actual:**
- **Surprised me:**
- **I can now explain:**
- **Still don't get:**
- **Tomorrow's first step:**
```

---

_No entries yet. The first one comes from F, session 1._
