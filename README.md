# small-jokeLLM

##  Описание проекта

Учебный проект по обучению небольшой языковой модели (Language Model) на русскоязычном датасете анекдотов. Включает полный пайплайн: от обучения собственного токенизатора до генерации текста.

##  Цель работы

1. Реализовать и обучить **Byte-level BPE токенизатор** с нуля
2. Реализовать **Transformer** модель для Causal Language Modeling
3. Обучить модель на датасете русских анекдотов
4. Залить артефакты (токенизатор и модель) на Hugging Face Hub

##  Структура ноутбука

### 1. Установка зависимостей
```
%pip install --quiet datasets livelossplot
```
Дополнительные библиотеки: `datasets`, `livelossplot`, `huggingface_hub`, `regex`, `torch`.

### 2. Датасет

Используется датасет **[IgorVolochay/russian_jokes](https://huggingface.co/datasets/IgorVolochay/russian_jokes)** с русскими анекдотами.

- Исходный размер: **135 497** примеров
- Разбивка: train/test = 90% / 10% (с фиксированным `SEED = 0xC0FFEE`)
- Train: ~121 947 примеров, Test: ~15 056 примеров

### 3. Токенизатор — Byte-level BPE

Реализация с нуля:

| Компонент | Описание |
|-----------|----------|
| `bytes_to_unicode()` | Маппинг 256 байт в Unicode-символы (аналог GPT-2) |
| `merge()` | Слияние самой частой пары токенов с обновлением статистики |
| `train()` | Обучение BPE: подсчёт пар, итеративное слияние до `vocab_size` |
| `ByteLevelBPETokenizer` | Класс с методами `encode`, `decode`, `push_to_hub`, `from_pretrained` |

**Параметры обучения:**
- `vocab_size = 1024`
- `special_tokens = ["[EOS]"]`
- Регулярка для разбиения слов: `WHITESPACE_SPLITTER` (стандартная из GPT-2)

**Статистика по токенизации:**
- Средняя длина последовательности: **73.49** токенов
- Минимум/максимум: **5 / 3418** токенов
- `MAX_SEQ_LEN = 128`

Токенизатор сохраняется в `vocabulary.json` + `merges.json` и пушится на HF Hub.

### 4. Модель — Transformer

Реализованы современные архитектурные решения:

| Компонент | Реализация |
|-----------|------------|
| **Позиционные эмбеддинги** | ALiBi (Attention with Linear Biases) — геометрическая прогрессия slopes |
| **Attention** | Grouped-Query Attention (GQA) — `n_head → n_kv_head` повтор через `repeat_interleave` |
| **Feed-Forward** | SwiGLU (Swish + Gated Linear Unit) |
| **Нормализация** | RMSNorm (pre-norm) |
| **Регуляризация** | Dropout на residual-ветках |
| **Тайинг весов** | `token_emb.weight = lm_head.weight` |

**Конфигурации моделей:**

| Название | n_layer | n_head | n_kv_head | hidden_dim | intermediate_dim |
|----------|---------|--------|-----------|------------|------------------|
| `nano`   | 3       | 4      | 2         | 96         | 256              |
| `mini`   | 6       | 6      | 3         | 384        | 1024             |
| `small`  | 12      | 12     | 6         | 768        | 2048             |

В ноутбуке обучается конфигурация **`nano`** (~0.40M параметров).

**Ключевые классы:**
- `RMSNorm` — Root Mean Square Layer Normalization
- `CausalSelfAttention` — с GQA + ALiBi + causal mask + padding mask
- `SwiGLU` — gated MLP
- `Block` — residual + pre-norm
- `TransformerForCausalLM` — с методом `generate()` (temperature, top_k, do_sample, eos)

### 5. Train Loop

**Класс `Trainer`** — кастомный цикл обучения с:
- `AdamW` оптимизатором (`lr=3e-4`, `weight_decay=0.01`)
- Linear warmup + linear decay scheduler (warmup = 10% шагов)
- Gradient clipping (`clip_grad_norm=1.0`)
- `n_steps = 10 000`
- Валидация каждые **1 000** шагов
- Визуализация через `livelossplot.PlotLosses`

**Дополнительно:**
- `TextDataset` — обёртка над списком текстов
- `data_collator` — паддинг до максимальной длины в батче + `attention_mask`
- `cross_entropy_loss` — с маскированием padding-токенов
- `BATCH_SIZE = 16`

### 6. Результаты

- Финальный train loss: **~3.995**, val loss: **~3.926**
- Пример генерации после обучения:

> *"Заходит в баркий. Продходил, американка:*
> *— Да, с ну, что вчера?*
> *— Да вот, наконец-тра, что выбралась настели*
> *— Все, а ты весь?*
> *Первый год."*

(Текст ещё далёк от идеала — модель маленькая, но структура диалога и синтаксис уже улавливаются.)

### 7. Публикация

Артефакты загружаются в репозиторий:
- `REPO_NAME = f"{username}/llm"` 
- Токенизатор: `vocabulary.json`, `merges.json`
- Модель: `model.safetensors` (через `PyTorchModelHubMixin`)

##  Ключевые особенности реализации

1. **Токенизатор написан полностью с нуля** — без `tokenizers` от HF
2. **Обучение BPE на 256 базовых байтах** — покрытие любого Unicode без `[UNK]`
3. **ALiBi вместо positional embeddings** — лучшая экстраполяция на длинные контексты
4. **GQA** — экономия памяти на KV-кэше
5. **SwiGLU** — более эффективный FFN по сравнению с GELU-MLP
6. **Weight tying** между embedding и lm_head

##  Возможные улучшения

- Увеличить `vocab_size` (2048–4096) — уменьшит среднюю длину последовательности
- Обучать `mini` или `small` конфигурацию для лучшего качества
- Добавить KV-кэш в `generate()` для ускорения инференса
- Использовать больше данных и `n_steps`
- Применить gradient accumulation для больших батчей

##  Ссылки

- Датасет: [IgorVolochay/russian_jokes](https://huggingface.co/datasets/IgorVolochay/russian_jokes)

