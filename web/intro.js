"use strict";

// Optional branding must never prevent access to the working application.
(() => {
  const dialog = document.getElementById("brand-intro");
  const slot = document.getElementById("intro-video-wrap");
  const skip = document.getElementById("skip-intro");
  const replay = document.getElementById("replay-intro");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let video = null, closing = false, loadTimer, maximumTimer, fadeTimer;

  if (!dialog || typeof dialog.showModal !== "function") return;
  replay.hidden = false;

  function cleanup() {
    clearTimeout(loadTimer);
    clearTimeout(maximumTimer);
    clearTimeout(fadeTimer);
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
      video.remove();
      video = null;
    }
    document.documentElement.classList.remove("intro-open");
    dialog.classList.remove("is-leaving");
    closing = false;
  }

  function finish() {
    if (!dialog.open || closing) return;
    closing = true;
    video?.pause();
    dialog.classList.add("is-leaving");
    fadeTimer = setTimeout(() => dialog.close(), motion.matches ? 0 : 400);
  }

  function start() {
    if (dialog.open) return;
    cleanup();
    video = document.createElement("video");
    video.className = "intro-video";
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.setAttribute("aria-hidden", "true");
    video.setAttribute("disablepictureinpicture", "");
    video.src = "/assets/media/forte-intro.mp4?v=2";
    video.addEventListener("ended", finish);
    video.addEventListener("error", finish);
    video.addEventListener("playing", () => clearTimeout(loadTimer), {once: true});
    slot.append(video);
    try {
      dialog.showModal();
    } catch (_) { cleanup(); return; }
    document.documentElement.classList.add("intro-open");
    skip.focus({preventScroll: true});
    loadTimer = setTimeout(finish, 3000);
    maximumTimer = setTimeout(finish, 12000);
    const playback = video.play();
    if (playback) playback.catch(finish);
  }

  skip.addEventListener("click", finish);
  dialog.addEventListener("cancel", event => { event.preventDefault(); finish(); });
  dialog.addEventListener("close", cleanup);
  replay.addEventListener("click", start);
  motion.addEventListener("change", event => { if (event.matches) finish(); });
  window.addEventListener("pagehide", () => { if (dialog.open) dialog.close(); cleanup(); });
  if (!motion.matches) start();
})();
