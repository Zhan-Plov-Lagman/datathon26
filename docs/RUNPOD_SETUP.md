# Серверный профиль L40S

Рабочий код находится в `/workspace/forte` на арендованном командой Runpod.
Локальная интеграционная копия остаётся в `C:\Users\arman\Desktop\ИТОГ`.
Базовый снимок сервера — `7f4e1078aeb15fa2fdb6643494e05dc48bf5a73e` плюс
серверные инструкции и диагностическая обёртка этого этапа.

Пользователь подтвердил самостоятельный запуск на арендованном сервере.
Модели исполняются внутри этой среды команды. Видео, кадры и их описания
не отправляются в облачные AI API. Скачивание публичных весов — отдельная setup-операция.
На текущем этапе на сервер перенесён только код; банковские данные не загружались.

## Оборудование и окружение

- NVIDIA L40S: драйвер показывает 46068 МиБ; драйвер 595.91.07.
- Шаблон `runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404`.
- Python 3.12.3; штатные torch 2.8.0+cu128 / torchvision 0.23.0+cu128.
- `.venv` использует штатную CUDA-сборку через system-site-packages.
- Transformers 4.57.6, bitsandbytes 0.48.2; версии остальных библиотек — в requirements.
- Persistent volume пользователя: 150 ГБ `/workspace`; container disk: 50 ГБ.
  `df` показывает размер общей сетевой файловой системы, а не квоту этого Pod.

Установка в том же образе:

```bash
cd /workspace/forte
bash scripts/setup_linux.sh
FORTE_PEER_SNAPSHOT="$PWD" PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/worker tests/rules tests/evaluation tests/api tests/integration
```

Использовать pip с исходными ограничениями torch. Первоначальный unconstrained
uv install скачал дополнительную CUDA 13 / torch 2.14.1 в изолированную среду;
этот дополнительный набор удалён из `.venv`, штатная CUDA не изменялась.
Окончательная проверка импорта подтверждает torch из `/usr/local/lib/python3.12/dist-packages`.

Для Ultralytics исправлены отсутствующие зависимости: thop 2.2.2,
nvidia-ml-py 13.615.71, ultralytics-platform 0.1.90. Сам клиент Platform не
используется для обработки видео. В адаптере отключены auto-install и sync.
`pip check` оставляет одно объяснимое сообщение: Ultralytics требует пакет
`opencv-python`, но здесь установлен `opencv-python-headless==4.11.0.86`,
предоставляющий тот же модуль cv2. Не устанавливать две реализации cv2 вместе.

## Выбор больших моделей

Qwen2B остаётся ноутбучным baseline. Пользователь запросил более мощные модели
для сервера. Основной кандидат — 32B NF4; 8B NF4 — сравнительный вариант.
Размер модели не считается доказательством точности на банковских записях.
Окончательный выбор зависит от ошибок и общего лимита 300 с из ТЗ.

| Модель | Ревизия | Веса и конфигурация на диске |
|---|---|---|
| Qwen/Qwen3-VL-32B-Instruct | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | 66,73 ГБ |
| Qwen/Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 17,55 ГБ |

Обе модели скачаны непосредственно на сервер, не занимают место ноутбука.
Сохраняются исходные safetensors, NF4 применяется при загрузке в GPU.
32B FP16 потребовала бы примерно 64+ ГБ только для параметров — поэтому здесь NF4.
Обе карточки моделей заявляют Apache-2.0:
[32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
[8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

Для воспроизведения скачать один раз:

```bash
.venv/bin/python scripts/download_model.py \
  --repo-id Qwen/Qwen3-VL-32B-Instruct \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb \
  --output-dir models/Qwen3-VL-32B-Instruct
.venv/bin/python scripts/download_model.py \
  --repo-id Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --output-dir models/Qwen3-VL-8B-Instruct
```

`download_manifest.json` фиксирует repo/revision после завершения загрузки.
Наличие одного config.json не означает готовность весов.

## Проверка без сетевых соединений

```bash
.venv/bin/python scripts/offline_smoke.py \
  --model-dir models/Qwen3-VL-32B-Instruct \
  --output-dir artifacts/server-setup/qwen32b-offline-smoke
```

Обёртка запускает существующий `smoke_worker.py`, генерирующий синтетический MOV.
Она блокирует Python socket connect/connect_ex/create_connection и включает offline
режим Hugging Face. Это диагностика, не firewall и не новая inference-реализация.
Выводит только метрики. Новый прогон требует нового output-dir; старый отчёт
не перезаписывается. Запускайте большие модели последовательно, чтобы сравнивать
время без конкуренции за GPU. Загрузка модели измеряется отдельно от анализа.

Для повторения проверки стандартного окна, уточнения и JPEG:

```bash
ffmpeg -nostdin -y -hide_banner -loglevel error -f lavfi \
  -i testsrc2=size=1280x720:rate=10:duration=12 -c:v libx264 \
  artifacts/server-setup/synthetic-window.mov
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/benchmark_worker.py \
  --video artifacts/server-setup/synthetic-window.mov \
  --model-dir models/Qwen3-VL-32B-Instruct \
  --output-dir artifacts/server-setup/qwen32b-window-new --refine --timeout 240
```

Фактический результат: загрузка 190,009 с отдельно; обзор + три уточнения +
два исходных JPEG — 77,658 с. Сохраняйте одну модель в процессе между заданиями.
Это 12-секундный синтетический вход, без правил и API, не замер полной службы.

## Серверный профиль runtime

```bash
cd /workspace/forte
set -a
source config/runpod-l40s.env
set +a
```

Этот профиль задаёт `FORTE_MODEL_DIR=models/Qwen3-VL-32B-Instruct` и offline-режим.
Он не запускает сервис, не содержит секретов и не подставляет тестовую фабрику
правил. Не подключать профиль к командам скачивания: для setup требуется сеть.

## Что проверено и что ещё предстоит

Результаты серверного этапа записаны в `INTEGRATION_STATUS.md`.
Успешный синтетический smoke подтверждает загрузку, исполнение, JSON и память;
он не измеряет качество, p95 и полный API-путь. Нужны сопоставимые локальные
проверенные примеры, затем готовая фабрика CV → адресная VLM-проверка → правила →
реальные JPEG → API. Hypothesis-реестр не активирован ради теста.

Сервер не выключается автоматически. По окончании аренды сначала сохранить
результаты, затем Stop; хранение продолжает тарифицироваться. При Terminate
volume disk удаляется. Приватные SSH-ключи и `.env` не включаются в код или отчёты.

## Кандидат для следующего сравнения скорости

Есть официальный [Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8).
Его карточка требует vLLM/SGLang: в текущий Transformers-адаптер веса не подставляются.
[vLLM поддерживает FP8 W8A8 на Ada SM8.9](https://docs.vllm.ai/en/latest/features/quantization/#supported-hardware).
Это возможный отдельный эксперимент с той же 32B, а не установленный backend:
FP8-веса здесь не скачаны, запуск и выигрыш не проверены. Рабочую среду NF4
не обновлять на месте ради этого сравнения. Сначала завершить фактический путь
CV → VLM → правила и получить измеримые ошибки; выбор другой реализации
без сравнения тех же примеров не доказывает улучшение качества.
