# TensorRT-LLM: hardware-dependent experiment

Это схема реального HTTP-подключения, **не отчёт об успешном NVIDIA запуске**. Core application уже выбирает OpenAICompatibleLLM при `LLM_BACKEND=tensorrt_llm`; отдельный SDK adapter и CUDA-код приложению не нужны.

## Environment

Нужны поддерживаемая NVIDIA GPU с достаточной VRAM, совместимые driver/CUDA и TensorRT-LLM version, выбранная supported model и соответствующий container/environment. Точную матрицу проверяйте для выбранного release: [supported hardware](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html), [installation](https://nvidia.github.io/TensorRT-LLM/installation/index.html), [quick start](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html).

GPU, CUDA stack и model weights отсутствуют в core lockfile. Контейнер TensorRT-LLM запускается отдельно от API/PostgreSQL. Не пытайтесь исправить отсутствие hardware добавлением fake CUDA calls в application.

## Runtime flow

В подготовленном NVIDIA environment базовый OpenAI-compatible server запускается через CLI:

```bash
trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 127.0.0.1 --port 8003
curl -f http://127.0.0.1:8003/health
curl -f http://127.0.0.1:8003/v1/models
```

Это illustrative command из documented serving interface; поддержка конкретной архитектуры, flags и объём памяти проверяются в выбранном release через `trtllm-serve --help`. TinyLlama приведена для узнаваемого protocol example, а не как гарантированно оптимальная модель для каждого TensorRT-LLM release.

В `.env` application перенесите значения из соседнего `.env.example`. `LLM_MODEL` должен совпадать с model ID сервера. Затем обычный POST `/api/v1/tickets/{id}/analyze` использует тот же HTTP adapter. Можно отдельно отправить chat request из [INFERENCE_GUIDE](../../docs/INFERENCE_GUIDE.md), заменив URL и model.

## Build vs runtime

Современный TensorRT-LLM включает PyTorch backend; предварительная сборка serialized TensorRT engine не является обязательным шагом каждого запуска `trtllm-serve`. Для TensorRT engine path flow иной: model checkpoint → supported conversion/quantization → engine build под hardware, precision и shape limits → runtime loading → HTTP serving. Build artifacts зависят от version/hardware и не равны переносимому PyTorch state_dict.

Универсальной команды `trtllm-build`, подходящей любой модели/версии, здесь нет: следуйте recipe конкретного release. Для quantization может потребоваться calibration dataset, а для FP8 — поддержка hardware. Эти шаги не имитируются unit tests.

## Проверка и диагностика

Проверьте health/models, затем минимальную chat completion, затем JSON draft через основной graph. Если нужна native LangChain tools integration, отдельно проверьте support JSON schema и tool calling. Только после этого измеряйте latency, memory и throughput. HTTP adapter контрактно протестирован на mock transport; это не проверка TensorRT execution.

Основной CLI reference: [trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html).

## Questions to answer after reading this code

1. Чем engine build отличается от runtime inference?
2. Почему serialized engine может зависеть от hardware и software version?
3. Когда TensorRT-LLM использует PyTorch backend?
4. Почему HTTP compatibility недостаточно для гарантии structured output?
5. Что именно проверяет наш contract test без NVIDIA GPU?
