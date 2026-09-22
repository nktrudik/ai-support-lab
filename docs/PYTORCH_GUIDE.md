# PyTorch and Lightning

Обе реализации используют `deep_learning/dataset.py` и `deep_learning/model.py`. Модель: Embedding → masked mean pooling → Linear → ReLU → Dropout → Linear. Нет transformer download или внешнего tokenizer.

## Tensor flow

| Объект | Shape | dtype / смысл |
|---|---|---|
| token IDs | `[B,64]` | long, индексы embedding |
| embeddings | `[B,64,48]` | float, обучаемые представления |
| mask | `[B,64,1]` | bool, padding исключён из pooling |
| pooled | `[B,48]` | float, среднее по непустым токенам |
| logits | `[B,7]` | float, ненормированные class scores |
| targets | `[B]` | long, class index |
| loss | `[]` | scalar, batch mean |

Vocabulary строится только на train. `<pad>=0`, `<unk>=1`; неизвестные tokens не меняют размер embedding. `Dataset.__getitem__` отдаёт token IDs и target, DataLoader объединяет их в batch. `num_workers=0` выбран для маленького in-memory dataset и простоты Windows/CPU запуска.

## Manual loop → Lightning abstraction

| Pure PyTorch: training.py | Lightning: lightning_module.py |
|---|---|
| `model.to(device)` / `batch.to(device)` | Trainer accelerator/devices |
| `for epoch ...` | Trainer(max_epochs=...) |
| `for batch in loader` | training_step(batch, batch_idx) |
| `model.train()` | Trainer задаёт training mode |
| `optimizer.zero_grad` | automatic optimization |
| forward + criterion | код training_step сохраняется явно |
| `loss.backward()` | automatic backward |
| gradient clipping | Trainer(gradient_clip_val=1.0) |
| `optimizer.step()` | automatic optimizer step |
| AdamW creation | configure_optimizers |
| `model.eval()` + inference_mode | validation loop Trainer |
| accumulation of metrics | self.log с batch_size |
| best loss + torch.save | ModelCheckpoint(monitor='val_loss') |

Lightning не меняет математическую модель и loss. Он управляет циклом, device и checkpoints. Сравнивать удобно после запуска обоих CLI с одинаковым числом epochs.

## Autograd

`forward` строит граф операций, loss.backward вычисляет градиенты параметров и добавляет их в `.grad`. Optimizer читает `.grad` и меняет weights. Следующий batch начинается с `zero_grad(set_to_none=True)`, потому что накопление градиентов по умолчанию не сбрасывается автоматически.

`model.train()` и `.eval()` переключают поведение Dropout; они не включают/выключают autograd. `inference_mode()` отключает отслеживание градиентов и дополнительную bookkeeping при inference. CrossEntropyLoss получает logits, а не probabilities.

## Checkpoints и inference

Manual loop сохраняет лучший validation checkpoint в `artifacts/torch.pt`: CPU state_dict, vocabulary, labels, version. Дополнительный `.resume.pt` показывает состояние optimizer, epoch и CPU RNG для изучения training state; готового CLI возобновления и точного replay sampler/CUDA RNG в lab нет. Lightning `.ckpt` хранит состояние Trainer; после обучения экспортируется та же inference schema в `artifacts/lightning.pt`.

`TorchTicketClassifier` использует `torch.load(..., map_location='cpu', weights_only=True)`, воссоздаёт архитектуру, загружает weights и вызывает eval. `TORCH_ARTIFACT=artifacts/lightning.pt` переключает serving на Lightning-обученные weights без нового inference класса.

Для NVIDIA training потребуются отдельный CUDA-compatible PyTorch environment и `--device cuda` / `--accelerator gpu`; default uv environment намеренно содержит CPU wheel. GPU выполнение здесь не проверено. Default training фиксирует seed и число CPU threads; побитовая воспроизводимость между версиями библиотек/оборудования не обещается.

## Questions to answer after reading this code

1. Что делает loss.backward и где находятся gradients?
2. Что изменится без zero_grad между batches?
3. Чем train/eval отличаются от no_grad/inference_mode?
4. Почему target имеет dtype long, а embedding output float?
5. Почему padding нельзя включать в обычное среднее?
6. Что делает DataLoader сверх Dataset?
7. Какие данные кроме state_dict нужны для корректного inference?
8. Почему best-validation metric нельзя назвать независимой test accuracy?
9. Какие действия Trainer выполняет вместо вашего цикла?
