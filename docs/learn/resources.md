# Learning resources

These are video-first and ordered by roadmap version.
- ⭐ marks the one to start with.
- Times are rough.
- Links were checked on 2026-10-07; to re-check, run the loop in [§ Checking links](#checking-links).

## F: Foundations
| | Resource | Format | Time |
|---|---|---|---|
| ⭐ | [Karpathy: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | video | 1 h |
| | [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (the series continues from there) | video | 20 min per episode |
| ⭐ | [3Blue1Brown: Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) | video | 30 min |
| ⭐ | [3Blue1Brown: Attention in transformers, step by step](https://www.youtube.com/watch?v=eMlx5fFNoYc) | video | 30 min |
| ⭐ | [Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | video | 2 h (watch in parts) |
| ⭐ | [Karpathy: Let's build GPT, from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) (required; you code along) | video | 2 h |
| ⭐ | [Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | video | 3.5 h (watch in parts) |
| | [Umar Jamil: LLaMA explained (KV cache, RoPE, RMSNorm, GQA, SwiGLU)](https://www.youtube.com/watch?v=Mn_9W1nCFLo) | video | 1 h |
| | [Umar Jamil: Attention is all you need, with the math](https://www.youtube.com/watch?v=bCz4OMemCcA) | video | 1 h (optional depth) |
| | [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | blog | 40 min |
| | [HF: Chat templates](https://huggingface.co/docs/transformers/chat_templating) | docs | 25 min |
| | [Tiktokenizer](https://tiktokenizer.vercel.app) (paste text, see the tokens) | tool | 10 min |
| | [HF Audio Course, ch.1: Audio data](https://huggingface.co/learn/audio-course/chapter1/audio_data) | course | 30 min |
| | [Pipecat docs](https://docs.pipecat.ai) · [LiveKit Agents docs](https://docs.livekit.io/agents/) (concept pages) | docs | 45 min each |

## v0: LLM hosting I
- ⭐ [mlx-lm README](https://github.com/ml-explore/mlx-lm) · [mlx-lm SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [OpenAI-compatible APIs: Ollama's explainer](https://docs.ollama.com/api/openai-compatibility), for the concept
- [Umar Jamil: Quantization explained with PyTorch](https://www.youtube.com/watch?v=0VdNflU08yA), the first half, for intuition

## v1: Tool calling and agents
- ⭐ [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI: Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [MCP: architecture overview](https://modelcontextprotocol.io/docs/learn/architecture)

## v2: RAG and memory
- ⭐ [Sentence-Transformers quickstart](https://www.sbert.net/docs/quickstart.html) · [pgvector README](https://github.com/pgvector/pgvector)
- [Anthropic: Contextual retrieval](https://www.anthropic.com/news/contextual-retrieval), covering hybrid BM25 plus dense retrieval and reranking
- [Chip Huyen: Building a GenAI platform](https://huyenchip.com/2024/07/25/genai-platform.html), the RAG section

## v3 and v4: Speech
- ⭐ [HF Audio Course, ch.1 preprocessing](https://huggingface.co/learn/audio-course/chapter1/preprocessing)
- ⭐ [HF Audio Course, ch.5: ASR models](https://huggingface.co/learn/audio-course/chapter5/asr_models) · [ch.5: evaluation (WER)](https://huggingface.co/learn/audio-course/chapter5/evaluation)
- ⭐ [HF Audio Course, ch.6: TTS](https://huggingface.co/learn/audio-course/chapter6/introduction) · [pre-trained TTS models](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models)
- [DeepLearning.AI: Open Source Models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/), the audio lessons
- Tools: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) · [mlx-audio](https://github.com/Blaizzy/mlx-audio) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

## v5: Real-time voice
- ⭐ [Pipecat docs](https://docs.pipecat.ai), on pipelines and interruptions
- [LiveKit Agents docs](https://docs.livekit.io/agents/), on turn detection
- [Silero VAD](https://github.com/snakers4/silero-vad)

## v6: Reliability and safety
- ⭐ [DeepLearning.AI: AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/quickstart)

## v7a and v7b: Evaluation and operations
- ⭐ [Hamel Husain: Your AI product needs evals](https://hamel.dev/blog/posts/evals/)
- ⭐ [Google SRE book: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)

## v8: Serving with ONNX and Triton
- ⭐ [Triton Conceptual Guide, part 1: model deployment](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment), then parts 2 and 3 in the [same repo](https://github.com/triton-inference-server/tutorials)
- [ONNX Runtime tutorials](https://onnxruntime.ai/docs/tutorials/)

## v9: LLM serving internals and vLLM
- ⭐ [Anyscale: Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U) (video)
- ⭐ [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) (video)
- [vLLM blog: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [DeepLearning.AI: Efficiently Serving LLMs](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)
- [Lilian Weng: Large transformer model inference optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Umar Jamil: Flash Attention derived and coded](https://www.youtube.com/watch?v=zy8ChVd_oTM) (depth)
- [vllm-metal docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/)

## v10: Quantisation
- ⭐ [Umar Jamil: Quantization explained with PyTorch (PTQ, QAT)](https://www.youtube.com/watch?v=0VdNflU08yA)
- ⭐ [DeepLearning.AI: Quantization Fundamentals](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/), then [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
- [HF: Quantization overview](https://huggingface.co/docs/transformers/main/quantization/overview)

## v11: Fine-tuning I
- ⭐ [Umar Jamil: LoRA explained visually, with PyTorch code](https://www.youtube.com/watch?v=PXWYUTMt-AU)
- ⭐ [Umar Jamil: DPO explained (Bradley-Terry, log-probs, math)](https://www.youtube.com/watch?v=hvGa5Mba4c8)
- [DeepLearning.AI: Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)
- [HF PEFT quicktour](https://huggingface.co/docs/peft/quicktour) · [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

## v12: CUDA and GPU fundamentals
- ⭐ [How GPU Computing Works (GTC 2021, Stephen Jones)](https://www.youtube.com/watch?v=3l10o0DYJXg)
- ⭐ [NVIDIA: An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [Intro to CUDA, part 1: high-level concepts](https://www.youtube.com/watch?v=4APkMJdiudU)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures), for kernel depth
- [Simon Oz: How to write a fast softmax kernel](https://www.youtube.com/watch?v=IpHjDoW4ffw)
- [Karpathy: Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU), the sections on mixed precision, flash attention and DDP

## v13: Fine-tuning II and quantisation at scale
- [HF TRL docs](https://huggingface.co/docs/trl/index) · [TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [llm-compressor (AWQ/GPTQ for vLLM)](https://github.com/vllm-project/llm-compressor)
- [vLLM LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html)

## v14: TensorRT and profiling
- ⭐ [TensorRT quick start guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## v15: Proving it under load
- ⭐ [Gil Tene: How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (on coordinated omission)
- [Locust docs](https://docs.locust.io/en/stable/)

## v16: Interviews
- [Chip Huyen: Designing Machine Learning Systems](https://huyenchip.com/books/) (selected chapters)
- [Chip Huyen: ML Interviews book](https://huyenchip.com/ml-interviews-book/)

## Checking links
```sh
grep -oE 'https?://[^) ]+' docs/learn/resources.md | sort -u | while read u; do
  printf '%s %s\n' "$(curl -sL -o /dev/null -w '%{http_code}' -A 'Mozilla/5.0' "$u")" "$u"; done | grep -v '^200'
```
YouTube links always return 200, so check them through `https://www.youtube.com/oembed?url=<link>&format=json` instead; a dead video returns 404 there.
