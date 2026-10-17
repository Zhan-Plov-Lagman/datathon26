# datathon26

Проект для кейса ForteBank на Datathon 26: локальный AI-сервис анализа видеозаписей кассовых операций с таймкодами, кадрами-доказательствами и нормативным обоснованием.

Команда: три участника, три ноутбука — две RTX 4050 и одна RTX 3050. Арман — роль A. Общие инструкции находятся в `main`, реализация локального видеомодуля — в ветке `feature/ml-worker`. Полный сервис с правилами и API ещё не собран.

Ветка `feature/api-ui` содержит API, сохраняемую очередь, локальный веб-интерфейс и транспорт GPU-обработчика; код A подключён без изменений. Правила B и настоящий внешний прогон ещё необходимы для готового анализа. На ноутбуке C запустите из корня `.\start.ps1`, затем откройте **http://127.0.0.1:8000**. Ключ — в созданном локальном `.env`.

Подробности: [запуск и публичный доступ](docs/DEPLOYMENT.md), [стык A/B/C](docs/WORKER_TRANSPORT.md), [результат роли C](docs/HANDOFF_C.md), [чеклист оригинального кейса](docs/CASE_CHECKLIST_C.md).

Начать здесь:

- [prompts/ROLE_A.md](prompts/ROLE_A.md), [prompts/ROLE_B.md](prompts/ROLE_B.md), [prompts/ROLE_C.md](prompts/ROLE_C.md) — готовые задания для ваших ИИ в открытой папке репозитория.
- [AGENTS.md](AGENTS.md) — общие инструкции помощникам, границы ролей и условия локальности.
- [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) — кто что делает, первые задачи и работа через Git.
- [FORTEBANK_PIPELINE.md](FORTEBANK_PIPELINE.md) — полный пайплайн, расписание и приёмка.
- [docs/CONTRACTS.md](docs/CONTRACTS.md) — общий контракт API и границы модулей до реализации.
- [docs/DATA_SETUP.md](docs/DATA_SETUP.md) — локальные данные на каждом ноутбуке и воспроизводимый аудит.

Для двух остальных участников:

```powershell
git clone https://github.com/Zhan-Plov-Lagman/datathon26.git
cd datathon26
```

Открыть эту папку в редакторе с ИИ. Участник со второй RTX 4050 передаёт помощнику текст `prompts/ROLE_B.md` и работает в `feature/rules-eval`; участник с RTX 3050 — текст `prompts/ROLE_C.md` и ветку `feature/api-ui`. Помощнику нужен доступ ко всей папке, а не только вложенный файл TEAM_WORKFLOW. Роль A уже выполняется на ноутбуке Армана.

Если репозиторий уже клонирован со старого адреса, повторно клонировать не нужно:

```powershell
git remote set-url origin https://github.com/Zhan-Plov-Lagman/datathon26.git
git fetch origin
```

Текущий код A и инструкция подключения: [ветка feature/ml-worker](https://github.com/Zhan-Plov-Lagman/datathon26/tree/feature/ml-worker), [docs/WORKER.md](https://github.com/Zhan-Plov-Lagman/datathon26/blob/feature/ml-worker/docs/WORKER.md), [docs/HANDOFF_A.md](https://github.com/Zhan-Plov-Lagman/datathon26/blob/feature/ml-worker/docs/HANDOFF_A.md). C объединяет ветки после проверки стыков; B и C не должны заново реализовывать видеомодуль A.

Скачать данные из исходной ссылки организаторов, разместить их локально и выполнить инструкцию `docs/DATA_SETUP.md`. Скрипт аудита уже доступен; команды запуска приложения добавляются вместе с работающей реализацией.

Исходные видео, таблица разметки, подробный аудит данных, веса моделей и секреты хранятся локально и исключены из Git. Обработка видео должна выполняться без обращения к внешним AI-сервисам.

Дедлайн: 18 октября 2026 года, 12:00, Кызылорда (UTC+5).
