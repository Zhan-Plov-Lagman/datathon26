"use strict";
const $ = id => document.getElementById(id);
let selectedFile = null, videoURL = null, pollTimer = null, pollGeneration = 0;
let currentJob = null, busy = false;
const labels = {processing: "Анализ", done: "Завершено", failed: "Ошибка"};

async function request(path, options = {}) {
  const key = $("api-key").value.trim();
  if (!key) throw new Error("Введите ключ доступа.");
  let response;
  try {
    response = await fetch(path, {...options, headers: {"X-API-Key": key}, signal: AbortSignal.timeout(options.method === "POST" ? 240000 : 30000)});
  } catch (_) { throw new Error("Нет ответа сервера. Проверьте подключение и повторите запрос."); }
  if (response.status === 401) throw new Error("Ключ доступа не принят. Проверьте его и повторите запрос.");
  if (response.status === 413) throw new Error("Файл превышает допустимый размер загрузки.");
  if (response.status === 422) throw new Error("Выберите непустой файл в формате MOV.");
  if (response.status === 404) throw new Error("Задание с таким ID не найдено.");
  if (response.status === 503 && path !== "/api/v1/readiness") throw new Error("Сервис временно занят или хранилище недоступно. Повторите позже.");
  if (!response.ok && path !== "/api/v1/readiness") throw new Error("Не удалось выполнить запрос. Повторите позже.");
  return response.json();
}

async function checkReady() {
  const box = $("readiness");
  try {
    const state = await request("/api/v1/readiness");
    box.dataset.ready = String(state.ready);
    box.textContent = state.ready ? "Анализ доступен. Можно загрузить запись." : "Ключ принят. Анализ пока недоступен: нет готового обработчика с проверенными правилами.";
    return state.ready;
  } catch (error) {
    box.dataset.ready = "false";
    box.textContent = error.message;
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
  if (currentJob && selectedFile && file !== selectedFile) {
    clearTimeout(pollTimer); ++pollGeneration; currentJob = null;
    $("events").replaceChildren(); $("job-info").hidden = true; $("empty-state").hidden = false;
    setStatus("", "");
  }
  selectedFile = file;
  $("file-name").textContent = file.name;
  $("file-meta").textContent = `${(file.size / 1024 / 1024).toFixed(1)} МБ · MOV`;
  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = URL.createObjectURL(file);
  $("player").src = videoURL;
  $("video-section").hidden = false;
  $("playback-note").hidden = true;
}
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
  clearTimeout(pollTimer); const generation = ++pollGeneration; currentJob = jobId;
  $("empty-state").hidden = true; $("job-info").hidden = false;
  $("job-label").textContent = `ID: ${jobId}`; $("job-id").value = jobId;
  $("events").replaceChildren();
  try { localStorage.setItem("forte-last-job", jobId); } catch (_) { /* Storage is optional. */ }
  let errors = 0;
  async function poll() {
    try {
      const result = await request(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
      if (generation !== pollGeneration) return;
      errors = 0;
      if (result.status === "processing") {
        setStatus("processing", "Запись ожидает свободный обработчик или анализируется. Можно оставить страницу открытой.");
        pollTimer = setTimeout(poll, 2000);
      } else if (result.status === "failed") {
        setStatus("failed", "Анализ не завершён из-за технической ошибки или истечения времени. Результат о наличии нарушений не получен.");
      } else {
        setStatus("done", result.verdict === "violation" ? `Подтверждённых событий: ${result.violations.length}. Проверьте кадры и основания ниже.` : "В рамках выполненной проверки нарушения не обнаружены.");
        renderEvents(result.violations);
        $("playback-note").hidden = !!selectedFile || !result.violations.length;
      }
    } catch (error) {
      if (generation !== pollGeneration) return;
      $("status-text").textContent = `${error.message} ID задания сохранён. Опрос можно возобновить через «Открыть задание по ID».`;
      if (++errors < 3) pollTimer = setTimeout(poll, 5000);
    }
  }
  setStatus("processing", "Получаем статус задания…"); poll();
}
$("upload-form").addEventListener("submit", async event => {
  event.preventDefault(); if (busy || !selectedFile) return;
  busy = true; $("submit").disabled = true; $("video-file").disabled = true;
  $("upload-message").textContent = "";
  try {
    if (!await checkReady()) return;
    const form = new FormData(); form.append("file", selectedFile);
    $("upload-message").textContent = "Загружаем файл…";
    const result = await request("/api/v1/analyze", {method: "POST", body: form});
    $("upload-message").textContent = "Запись принята. Сохраните ID задания для повторного просмотра.";
    watchJob(result.job_id);
  } catch (error) { $("upload-message").textContent = error.message; }
  finally { busy = false; $("submit").disabled = false; $("video-file").disabled = false; }
});
$("resume-form").addEventListener("submit", event => {
  event.preventDefault(); const jobId = $("job-id").value.trim();
  if (jobId) watchJob(jobId);
});
try { $("job-id").value = localStorage.getItem("forte-last-job") || ""; } catch (_) { /* Optional. */ }
window.addEventListener("beforeunload", () => { clearTimeout(pollTimer); if (videoURL) URL.revokeObjectURL(videoURL); });
