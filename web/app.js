"use strict";
const $ = id => document.getElementById(id);
let selectedFile = null, videoURL = null, pollTimer = null, pollGeneration = 0;
let currentJob = null, busy = false;
let jobMeta = null, elapsedTimer = null, clockOrigin = null, clockOffset = 0;
let pollActive = false, pollKey = null, readyGeneration = 0, fileDuration = null, jobUnavailable = false;
const sessionJobKey = "forte-job-session";
const labels = {processing: "Анализ", done: "Завершено", failed: "Ошибка"};

function validJobId(value) { return typeof value === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(value); }
function urlJobId() {
  const match = /^#job=([A-Za-z0-9_-]{1,128})$/.exec(location.hash);
  return match ? match[1] : null;
}
function readSavedJob() {
  let saved = null, oldId = null;
  try { saved = JSON.parse(sessionStorage.getItem(sessionJobKey)); } catch (_) { /* Optional storage. */ }
  try { oldId = localStorage.getItem("forte-last-job"); } catch (_) { /* Older UI compatibility. */ }
  const id = urlJobId() || (validJobId(saved?.jobId) ? saved.jobId : null) || (validJobId(oldId) ? oldId : null);
  if (!id) return null;
  if (!saved || saved.jobId !== id) return {jobId: id};
  const number = value => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  return {jobId: id, filename: typeof saved.filename === "string" ? saved.filename.slice(0, 500) : null,
    durationSec: number(saved.durationSec), startedWallMs: number(saved.startedWallMs),
    elapsedMs: number(saved.elapsedMs), finishedWallMs: number(saved.finishedWallMs),
    terminalState: ["done", "failed"].includes(saved.terminalState) ? saved.terminalState : null};
}
function saveJob() {
  if (!validJobId(currentJob) || jobMeta?.jobId !== currentJob) return;
  // Only navigation/timing metadata: never credentials, video bytes or results.
  try { sessionStorage.setItem(sessionJobKey, JSON.stringify(jobMeta)); } catch (_) { /* Optional storage. */ }
  try { localStorage.setItem("forte-last-job", currentJob); } catch (_) { /* Optional storage. */ }
}
function jobHeader() {
  $("empty-state").hidden = true; $("job-info").hidden = false;
  $("job-label").textContent = currentJob ? `ID: ${currentJob}` : "Запуск анализа";
  $("job-id").value = currentJob || "";
  $("job-link").hidden = !currentJob;
  if (currentJob) $("job-link").href = `#job=${encodeURIComponent(currentJob)}`;
}
function elapsedMilliseconds() {
  if (typeof jobMeta?.elapsedMs === "number") return jobMeta.elapsedMs;
  return clockOrigin === null ? null : clockOffset + performance.now() - clockOrigin;
}
function renderTiming() {
  const elapsed = elapsedMilliseconds();
  $("elapsed-status").hidden = elapsed === null;
  if (elapsed !== null) $("elapsed-value").textContent = String(Math.max(1, Math.round(elapsed / 1000)));
  const parts = [];
  if (jobMeta?.filename) parts.push(jobMeta.filename);
  if (typeof jobMeta?.durationSec === "number") parts.push(`видео ${Math.round(jobMeta.durationSec)} с`);
  if (jobMeta?.terminalState) {
    parts.push(elapsed === null ? "время до результата неизвестно" : `до результата ${Math.max(1, Math.round(elapsed / 1000))} с`);
    if (jobMeta.finishedWallMs) parts.push(new Intl.DateTimeFormat("ru-RU", {
      day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit"}).format(jobMeta.finishedWallMs));
  }
  $("result-meta").textContent = parts.join(" · ");
  $("result-meta").hidden = !parts.length;
}
function stopElapsed() { clearInterval(elapsedTimer); elapsedTimer = null; }
function startElapsed() {
  stopElapsed();
  if (!jobMeta?.terminalState && clockOrigin === null && typeof jobMeta?.startedWallMs === "number") {
    clockOffset = Math.max(0, Date.now() - jobMeta.startedWallMs); clockOrigin = performance.now();
  }
  renderTiming();
  if (!jobMeta?.terminalState && clockOrigin !== null) elapsedTimer = setInterval(renderTiming, 250);
}
function finishTiming(state) {
  if (!jobMeta.terminalState) {
    jobMeta.elapsedMs = elapsedMilliseconds(); jobMeta.finishedWallMs = Date.now();
  }
  jobMeta.terminalState = state; stopElapsed(); saveJob(); renderTiming();
}
function stopPolling() { clearTimeout(pollTimer); pollTimer = null; ++pollGeneration; pollActive = false; pollKey = null; }
function resetJob() {
  stopPolling(); stopElapsed(); currentJob = null; jobMeta = null; clockOrigin = null; clockOffset = 0;
  jobUnavailable = false;
  try { sessionStorage.removeItem(sessionJobKey); localStorage.removeItem("forte-last-job"); } catch (_) { /* Optional. */ }
  if (urlJobId()) history.replaceState(null, "", location.pathname + location.search);
  $("events").replaceChildren(); $("job-info").hidden = true; $("empty-state").hidden = false;
  $("job-id").value = ""; setStatus("", ""); renderTiming();
}
function awaitingKey() {
  jobHeader(); startElapsed();
  setStatus("", "Задание сохранено на сервере. Введите ключ доступа: результат загрузится автоматически, повторно отправлять видео не нужно.");
}

async function request(path, options = {}) {
  const key = $("api-key").value.trim();
  if (!key) throw new Error("Введите ключ доступа.");
  let response;
  try {
    response = await fetch(path, {...options, headers: {"X-API-Key": key}, signal: AbortSignal.timeout(options.method === "POST" ? 240000 : 30000)});
  } catch (_) { throw new Error("Нет ответа сервера. Проверьте подключение и повторите запрос."); }
  if (response.status === 401) {
    const error = new Error("Ключ доступа не принят. Проверьте его и повторите запрос."); error.status = 401; throw error;
  }
  if (response.status === 413) throw new Error("Файл превышает допустимый размер загрузки.");
  if (response.status === 422) throw new Error("Выберите непустой файл в формате MOV.");
  if (response.status === 404) { const error = new Error("Задание с таким ID не найдено."); error.status = 404; throw error; }
  if (response.status === 503 && path !== "/api/v1/readiness") throw new Error("Сервис временно занят или хранилище недоступно. Повторите позже.");
  if (!response.ok && !(path === "/api/v1/readiness" && response.status === 503)) throw new Error("Не удалось выполнить запрос. Повторите позже.");
  return response.json();
}

async function checkReady({resume = true} = {}) {
  const box = $("readiness");
  const generation = ++readyGeneration;
  try {
    const state = await request("/api/v1/readiness");
    if (generation !== readyGeneration) return false;
    if (typeof state.ready !== "boolean") throw new Error("Сервер вернул неполный статус готовности. Повторите запрос.");
    box.dataset.ready = String(state.ready);
    box.textContent = state.ready ? "Анализ доступен. Можно загрузить запись." : "Ключ принят. Анализ пока недоступен: обработчик не готов.";
    if (resume && currentJob && !busy) watchJob(currentJob);
    return state.ready;
  } catch (error) {
    if (generation !== readyGeneration) return false;
    box.dataset.ready = "false";
    box.textContent = error.message;
    if (currentJob && error.status === 401) {
      stopPolling(); stopElapsed(); jobHeader();
      setStatus("", "Ключ не принят. Введите действующий ключ: сохранённое задание будет открыто автоматически.");
    }
    return false;
  }
}
$("check-ready").addEventListener("click", checkReady);
$("api-key").addEventListener("change", checkReady);

function selectFile(file) {
  if (busy) return;
  $("upload-message").textContent = "";
  if (!file || !/\.mov$/i.test(file.name) || !file.size) {
    selectedFile = null;
    $("video-section").hidden = true;
    $("upload-message").textContent = "Выберите непустой файл MOV.";
    return;
  }
  if (currentJob && (jobUnavailable || (selectedFile && file !== selectedFile) || (jobMeta?.filename && file.name !== jobMeta.filename))) resetJob();
  selectedFile = file;
  fileDuration = null;
  $("file-name").textContent = file.name;
  $("file-meta").textContent = `${(file.size / 1024 / 1024).toFixed(1)} МБ · MOV`;
  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = URL.createObjectURL(file);
  $("player").src = videoURL;
  $("video-section").hidden = false;
  $("playback-note").hidden = true;
}
$("player").addEventListener("loadedmetadata", () => {
  const duration = $("player").duration;
  if (!selectedFile || !Number.isFinite(duration) || duration <= 0) return;
  fileDuration = duration;
  if (jobMeta?.filename === selectedFile.name) { jobMeta.durationSec = duration; saveJob(); renderTiming(); }
});
$("video-file").addEventListener("change", event => selectFile(event.target.files[0]));
for (const type of ["dragenter", "dragover"]) $("drop-zone").addEventListener(type, event => {
  event.preventDefault(); $("drop-zone").classList.add("dragging");
});
for (const type of ["dragleave", "drop"]) $("drop-zone").addEventListener(type, event => {
  event.preventDefault(); $("drop-zone").classList.remove("dragging");
});
$("drop-zone").addEventListener("drop", event => {
  if (busy) return;
  selectFile(event.dataTransfer.files[0]);
  if (selectedFile) $("video-file").files = event.dataTransfer.files;
});

function setStatus(state, text) {
  $("status-badge").textContent = labels[state] || "Ожидание";
  $("status-badge").dataset.state = state;
  $("status-text").textContent = text;
  $("submit").disabled = busy || !!(currentJob && !jobMeta?.terminalState);
}
function timeText(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}
function timeButton(seconds) {
  const button = document.createElement("button");
  button.type = "button"; button.className = "time-button"; button.textContent = timeText(seconds);
  button.setAttribute("aria-label", `Перейти к ${timeText(seconds)}`);
  button.addEventListener("click", () => {
    const video = $("player");
    if (!selectedFile) { $("playback-note").hidden = false; return; }
    if (Number.isFinite(video.duration)) {
      video.currentTime = Math.min(seconds, video.duration);
      video.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center"});
    } else {
      $("playback-note").textContent = "Браузер не смог открыть этот MOV. Исходные кадры доступны ниже; для воспроизведения используйте локальный видеоплеер.";
      $("playback-note").hidden = false;
    }
  });
  return button;
}
function renderEvents(events) {
  $("events").replaceChildren();
  for (const [index, event] of events.entries()) {
    const card = document.createElement("article"); card.className = "event";
    const heading = document.createElement("div"); heading.className = "event-heading";
    const title = document.createElement("h3"); title.textContent = `${index + 1}. ${event.type}`;
    heading.append(title, timeButton(event.timecode_sec));
    const description = document.createElement("p"); description.textContent = event.description;
    const frames = document.createElement("div"); frames.className = "frames";
    for (const frame of event.evidence_frames) {
      const figure = document.createElement("figure");
      const image = document.createElement("img"); image.src = `data:image/jpeg;base64,${frame.image_base64}`;
      image.alt = `Исходный кадр, ${timeText(frame.timestamp_sec)}`; image.loading = "lazy";
      const caption = document.createElement("figcaption"); caption.append(timeButton(frame.timestamp_sec));
      figure.append(image, caption); frames.append(figure);
    }
    card.append(heading, description, frames); $("events").append(card);
  }
}
function watchJob(jobId) {
  if (!validJobId(jobId)) return;
  const key = $("api-key").value.trim();
  if (pollActive && currentJob === jobId && pollKey === key) return;
  stopPolling();
  jobUnavailable = false;
  if (jobMeta?.jobId !== jobId) {
    stopElapsed(); jobMeta = {jobId}; clockOrigin = null; clockOffset = 0;
  }
  currentJob = jobId; jobHeader(); saveJob();
  history.replaceState(null, "", `#job=${encodeURIComponent(jobId)}`);
  if (!key) { awaitingKey(); return; }
  const generation = pollGeneration; pollActive = true; pollKey = key;
  $("events").replaceChildren();
  startElapsed();
  let errors = 0;
  async function poll() {
    try {
      const result = await request(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
      if (generation !== pollGeneration) return;
      errors = 0;
      if (result.status === "processing") {
        setStatus("processing", "Видео принято. Ожидаем результат сервера; задание в очереди или анализируется. После обновления страницы введите ключ, чтобы продолжить просмотр.");
        pollTimer = setTimeout(poll, 2000);
      } else if (result.status === "failed") {
        pollActive = false; finishTiming("failed");
        setStatus("failed", "Анализ не завершён из-за технической ошибки или истечения времени. Результат о наличии нарушений не получен.");
      } else if (result.status === "done") {
        pollActive = false; finishTiming("done");
        setStatus("done", result.verdict === "violation" ? `Обнаружены признаки возможных нарушений: ${result.violations.length}. Проверьте кадры и основания ниже.` : "В рамках выполненной проверки признаки возможных нарушений не обнаружены.");
        renderEvents(result.violations);
        $("playback-note").hidden = !!selectedFile || !result.violations.length;
      } else throw new Error("Сервер вернул неизвестный статус задания. Повторите проверку.");
    } catch (error) {
      if (generation !== pollGeneration) return;
      if (error.status === 401) {
        pollActive = false; stopElapsed();
        setStatus("", "Ключ не принят. Введите действующий ключ: сохранённое задание будет открыто автоматически.");
      } else if (error.status !== 404 && ++errors < 3) {
        $("status-text").textContent = `${error.message} Задание сохранено; повторим запрос через 5 секунд.`;
        pollTimer = setTimeout(poll, 5000);
      } else {
        pollActive = false; stopElapsed();
        jobUnavailable = error.status === 404;
        $("status-text").textContent = `${error.message} Задание сохранено. Нажмите «Проверить подключение», чтобы повторить запрос.`;
      }
    }
  }
  setStatus("processing", "Получаем статус задания…"); poll();
}
$("upload-form").addEventListener("submit", async event => {
  event.preventDefault(); if (busy || !selectedFile || (currentJob && !jobMeta?.terminalState)) return;
  resetJob();
  jobMeta = {jobId: null, filename: selectedFile.name, durationSec: fileDuration,
    startedWallMs: Date.now(), elapsedMs: null, finishedWallMs: null, terminalState: null};
  clockOrigin = performance.now(); clockOffset = 0; jobHeader(); startElapsed();
  busy = true; $("submit").disabled = true; $("video-file").disabled = true;
  $("upload-message").textContent = "";
  setStatus("processing", "Проверяем доступность анализа…");
  try {
    if (!await checkReady({resume: false})) {
      stopElapsed(); setStatus("", "Запуск не начат. Проверьте ключ и готовность обработчика."); return;
    }
    const form = new FormData(); form.append("file", selectedFile);
    $("upload-message").textContent = "Загружаем файл…";
    setStatus("processing", "Загружаем видео на сервер. Дождитесь подтверждения приёма; до него обновлять страницу не следует.");
    const result = await request("/api/v1/analyze", {method: "POST", body: form});
    if (!validJobId(result.job_id)) throw new Error("Сервер не вернул ID задания. Приём видео не подтверждён.");
    currentJob = result.job_id; jobMeta.jobId = result.job_id; saveJob();
    $("upload-message").textContent = "Видео принято. ID сохранён; после обновления страницы введите ключ для просмотра результата.";
    watchJob(result.job_id);
  } catch (error) { stopElapsed(); $("upload-message").textContent = error.message; setStatus("failed", error.message); }
  finally { busy = false; $("submit").disabled = !!(currentJob && !jobMeta?.terminalState); $("video-file").disabled = false; }
});
$("resume-form").addEventListener("submit", event => {
  event.preventDefault(); if (busy) return;
  const jobId = $("job-id").value.trim();
  if (jobId) watchJob(jobId);
});
const restored = readSavedJob();
if (restored) { jobMeta = restored; currentJob = restored.jobId; awaitingKey(); }
window.addEventListener("hashchange", () => {
  const id = urlJobId();
  if (id && id !== currentJob && !busy) watchJob(id);
});
window.addEventListener("beforeunload", () => { stopPolling(); stopElapsed(); if (videoURL) URL.revokeObjectURL(videoURL); });
