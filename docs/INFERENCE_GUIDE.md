# Inference guide

`llm/protocols.py` описывает application contract, `schemas.py` — messages/request/response, `providers/openai_compatible.py` — один async HTTP adapter. Backends vLLM, llama_cpp и tensorrt_llm используют этот же adapter, выбранный `llm/factory.py`; им не нужны пустые классы с разными именами.

## Mock и remote

| MockLLM | OpenAICompatibleLLM |
|---|---|
| Читает JSON факты из последнего user message | Отправляет messages на `/v1/chat/completions` |
| Возвращает deterministic draft | Получает реальную generation сервера |
| Нет сети, GPU, API key | Нужен внешний процесс и model weights |
| Проверяет integration flow | Качество/latency зависят от engine и модели |
| Не сообщает выдуманное token usage | Передаёт usage, если server его вернул |

Общий HTTP client живёт в lifespan; timeout задаётся Settings. Transport errors превращаются в domain exceptions с exception chaining. Completion envelope проверяется отдельно от содержимого draft. `healthcheck()` проверяет `/v1/models`, а health serving engine можно проверить его собственным endpoint.

## vLLM — optional integration

В отдельном NVIDIA environment с установленным vLLM и совместимыми drivers:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8002
curl -f http://127.0.0.1:8002/health
curl -f http://127.0.0.1:8002/v1/models
```

Эта небольшая модель — пример подключения, не рекомендация качества support reasoning. Model weights загружаются только в вашем serving environment. Перенесите значения из `experiments/vllm.env.example` в `.env`, затем перезапустите API. Core uv environment не устанавливает vLLM.

vLLM ориентирован на serving throughput и работу с KV cache/batching. Выбор GPU memory, max model length и quantization зависит от hardware/model; здесь не заявлены измерения производительности. [Официальная документация](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).

## llama.cpp — optional integration

Нужны собранный `llama-server` и локальный GGUF instruct-model с корректным chat template. После установки из [официального llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
llama-server -m /absolute/path/to/model.gguf --alias support-local \
  --host 127.0.0.1 --port 8080 -c 4096 -ngl 0
curl -f http://127.0.0.1:8080/health
curl -f http://127.0.0.1:8080/v1/models
```

`-ngl 0` задаёт CPU execution; для конкретной сборки проверьте `llama-server --help`. Путь к model.gguf необходимо заменить реальным. Никаких weights в repository нет. Перенесите `experiments/llama_cpp.env.example` в `.env` и перезапустите API.

llama.cpp использует C/C++ runtime и GGUF, удобен для локальных quantized моделей и разных аппаратных платформ. Он тоже поддерживает batching; противопоставление «vLLM умеет batching, llama.cpp никогда не умеет» было бы неверным. Конкретный выбор зависит от нагрузки, модели, оборудования и latency/throughput. [Server reference](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

## Общий HTTP-запрос

Для vLLM; для другого engine замените URL и model ID на `/v1/models` вашего сервера:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Explain a login token failure briefly."}],"temperature":0,"max_tokens":128}'
```

Основной graph просит JSON через prompt и проверяет его Pydantic: это наиболее переносимый базовый контракт. Строгая constrained decoding и native tool calling зависят от model template, flags и engine version. Пример `scripts/native_langchain.py` требует и tools, и JSON schema; обычная доступность `/chat/completions` этого не гарантирует.

## TensorRT-LLM

[Отдельный experiment](../experiments/tensorrt_llm/README.md) описывает NVIDIA environment, runtime и optional engine build. Не устанавливайте TensorRT-LLM в основной uv environment. HTTP-интеграция готова, hardware setup не считается проверенным запуском.

## Questions to answer after reading this code

1. Зачем скрывать serving engine за Protocol?
2. Чем weights/model architecture отличаются от serving engine?
3. Что OpenAI-compatible API гарантирует, а что не гарантирует?
4. Какие workloads могут выиграть от vLLM и почему?
5. Когда удобен llama.cpp и что такое GGUF?
6. Где находится TensorRT-LLM относительно PyTorch и CUDA?
7. Почему timeout не равен malformed output?
8. Почему temperature=0 не гарантирует побитовую идентичность real GPU inference?
