(function initializePlayLens() {
  "use strict";

  const core = globalThis.PlayLensCore;
  if (!core || globalThis.__playLensController) return;
  const ownerId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  if (!core.claimControllerOwnership(document.documentElement, ownerId)) return;

  const API_BASE = `${location.origin}/api/v1`;
  const DASHBOARD_BASE = "http://localhost:3001";
  const SAMPLE_INTERVAL_MS = 250;
  const CAPTURE_WIDTH = 160;
  const CAPTURE_HEIGHT = 160;
  const SIGNATURE_SIZE = 16;

  class PlayLensController {
    constructor() {
      this.armed = false;
      this.sessionId = null;
      this.captureTimer = null;
      this.frameInFlight = false;
      this.detector = new core.RunDetector();
      this.previousSignature = null;
      this.root = null;
      this.shadow = null;
      this.recorder = null;
      this.recordingChunks = [];
      this.collapsed = false;
      this.lastPrediction = null;
      this.launcher = null;
      this.collectionRunNumber = null;
      this.collectionTarget = 50;
      this.sessionMode = "collection";
      this.paused = false;
      this.activeAccumulatedMs = 0;
      this.activeSegmentStartedAt = null;
      this.visibilityHandler = () => void this.sample();
      this.pageHideHandler = () => { if (this.sessionId) void this.finishSession("navigation"); };
      this.pageShowHandler = (event) => {
        if (!event.persisted) return;
        this.resetRunState();
        if (this.armed) {
          this.render({ status: "unavailable", unavailableReason: "Waiting for a new run" });
          void this.sample();
        }
      };
      this.captureFailureSince = null;
      this.startRequestedAt = null;
      this.gameStartHandler = (event) => {
        if (event.type === "keydown") {
          if (event.key !== "Enter" || !this.isVisible("#startBtn")) return;
        } else if (!(event.target instanceof Element) || !event.target.closest("#startBtn")) {
          return;
        }
        this.startRequestedAt = Date.now();
      };
      this.captureCanvas = document.createElement("canvas");
      this.captureCanvas.width = CAPTURE_WIDTH;
      this.captureCanvas.height = CAPTURE_HEIGHT;
      this.captureContext = this.captureCanvas.getContext("2d");
      this.signatureCanvas = document.createElement("canvas");
      this.signatureCanvas.width = SIGNATURE_SIZE;
      this.signatureCanvas.height = SIGNATURE_SIZE;
      this.signatureContext = this.signatureCanvas.getContext("2d", { willReadFrequently: true });
    }

    async toggle() {
      if (this.armed) await this.disarm();
      else await this.arm();
      return { armed: this.armed };
    }

    async arm() {
      if (this.armed) return;
      this.armed = true;
      this.launcher?.remove();
      this.launcher = null;
      this.mountOverlay();
      this.render({ status: "unavailable", unavailableReason: "Waiting for a new run" });
      document.addEventListener("visibilitychange", this.visibilityHandler);
      document.addEventListener("mousedown", this.gameStartHandler, true);
      document.addEventListener("touchstart", this.gameStartHandler, true);
      document.addEventListener("keydown", this.gameStartHandler, true);
      window.addEventListener("pagehide", this.pageHideHandler);
      window.addEventListener("pageshow", this.pageShowHandler);
      this.captureTimer = window.setInterval(() => void this.sample(), SAMPLE_INTERVAL_MS);
      await this.sample();
    }

    async disarm() {
      this.armed = false;
      window.clearInterval(this.captureTimer);
      this.captureTimer = null;
      document.removeEventListener("visibilitychange", this.visibilityHandler);
      document.removeEventListener("mousedown", this.gameStartHandler, true);
      document.removeEventListener("touchstart", this.gameStartHandler, true);
      document.removeEventListener("keydown", this.gameStartHandler, true);
      window.removeEventListener("pagehide", this.pageHideHandler);
      window.removeEventListener("pageshow", this.pageShowHandler);
      if (this.sessionId) await this.finishSession("disarmed");
      this.stopRecording();
      this.resetRunState();
      this.root?.remove();
      this.root = null;
      this.shadow = null;
      this.mountLauncher();
    }

    resetRunState() {
      this.detector.reset();
      this.previousSignature = null;
      this.paused = false;
      this.activeAccumulatedMs = 0;
      this.activeSegmentStartedAt = null;
      this.collectionRunNumber = null;
      this.lastPrediction = null;
      this.captureFailureSince = null;
      this.startRequestedAt = null;
    }

    mountLauncher() {
      if (this.armed || this.launcher) return;
      const host = document.createElement("div");
      host.id = "playlens-extension-launcher";
      host.style.cssText = "position:fixed;right:20px;top:20px;z-index:2147483647;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;";
      const shadow = host.attachShadow({ mode: "open" });
      shadow.innerHTML = `
        <style>
          button { display:flex; align-items:center; gap:9px; padding:10px 13px; border:1px solid rgba(255,255,255,.22); border-radius:12px; color:#f7f7f2; background:rgba(13,17,19,.86); box-shadow:0 12px 36px rgba(0,0,0,.25); backdrop-filter:blur(12px); cursor:pointer; font:750 11px/1 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
          button:hover { background:rgba(24,31,33,.94); }
          i { width:7px; height:7px; border-radius:50%; background:#5dd39e; box-shadow:0 0 0 4px rgba(93,211,158,.12); }
        </style>
        <button type="button" aria-label="Enable PlayLens capture"><i></i>Enable PlayLens</button>`;
      shadow.querySelector("button").addEventListener("click", () => void this.arm());
      document.documentElement.appendChild(host);
      this.launcher = host;
    }

    findCanvas() {
      const canvas = document.querySelector("canvas#canvas");
      if (!(canvas instanceof HTMLCanvasElement) || canvas.width === 0 || canvas.height === 0) return null;
      return canvas;
    }

    isVisible(selector) {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) return false;
      const style = getComputedStyle(element);
      return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity || 1) > 0.05;
    }

    gameIsOver() {
      return this.isVisible("#gameoverscreen");
    }

    gameIsPaused() {
      if (document.hidden) return true;
      const pauseButton = document.querySelector("#pauseBtn");
      if (!(pauseButton instanceof HTMLImageElement)) return false;
      return pauseButton.src.includes("btn_resume.svg") && this.isVisible("#pauseBtn");
    }

    gameIsInProgress() {
      return this.isVisible("#pauseBtn") && !this.gameIsOver();
    }

    prepareCapture(canvas) {
      const context = this.captureContext;
      context.fillStyle = "#000";
      context.fillRect(0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
      const scale = Math.min(CAPTURE_WIDTH / canvas.width, CAPTURE_HEIGHT / canvas.height);
      const width = Math.round(canvas.width * scale);
      const height = Math.round(canvas.height * scale);
      context.drawImage(canvas, Math.round((CAPTURE_WIDTH - width) / 2), Math.round((CAPTURE_HEIGHT - height) / 2), width, height);
      this.signatureContext.drawImage(canvas, 0, 0, SIGNATURE_SIZE, SIGNATURE_SIZE);
      const rgba = this.signatureContext.getImageData(0, 0, SIGNATURE_SIZE, SIGNATURE_SIZE).data;
      const gray = new Uint8Array(SIGNATURE_SIZE * SIGNATURE_SIZE);
      for (let source = 0, target = 0; source < rgba.length; source += 4, target += 1) {
        gray[target] = Math.round(0.299 * rgba[source] + 0.587 * rgba[source + 1] + 0.114 * rgba[source + 2]);
      }
      return gray;
    }

    frameData() {
      return this.captureCanvas.toDataURL("image/jpeg", 0.72);
    }

    activeElapsedMs(now = Date.now()) {
      return Math.max(0, Math.round(this.activeAccumulatedMs + (this.activeSegmentStartedAt === null ? 0 : now - this.activeSegmentStartedAt)));
    }

    async sample() {
      if (!this.armed || this.frameInFlight) return;
      const canvas = this.findCanvas();
      if (!canvas) {
        this.render({ status: "unavailable", unavailableReason: "Game canvas not found" });
        this.captureFailureSince ??= Date.now();
        if (this.sessionId && Date.now() - this.captureFailureSince >= 3_000) await this.finishSession("capture_error");
        return;
      }
      this.captureFailureSince = null;

      let currentSignature;
      try {
        currentSignature = this.prepareCapture(canvas);
      } catch (_) {
        this.render({ status: "unavailable", unavailableReason: "Canvas capture blocked" });
        this.captureFailureSince ??= Date.now();
        if (this.sessionId && Date.now() - this.captureFailureSince >= 3_000) await this.finishSession("capture_error");
        return;
      }

      const motion = core.averageMotion(this.previousSignature, currentSignature);
      this.previousSignature = currentSignature;
      const event = this.detector.update({
        motion,
        hidden: document.hidden,
        paused: this.gameIsPaused(),
        gameOver: this.gameIsOver(),
        playing: this.gameIsInProgress()
          || (this.startRequestedAt !== null && Date.now() - this.startRequestedAt < 2_000),
      });

      if (event === "started") {
        this.startRequestedAt = null;
        await this.startSession(canvas);
      }
      if (event === "paused" && this.sessionId) await this.setPaused(true);
      if (event === "resumed" && this.sessionId) await this.setPaused(false);
      if (event === "ended" && this.sessionId) {
        await this.finishSession("game_over");
        return;
      }
      if (!this.sessionId || this.paused || document.hidden) return;

      this.frameInFlight = true;
      const timestampMs = Date.now();
      try {
        const response = await fetch(`${API_BASE}/sessions/${this.sessionId}/frames`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            timestampMs,
            activeElapsedMs: this.activeElapsedMs(timestampMs),
            imageDataUrl: this.frameData(),
            observedMotion: motion,
          }),
        });
        if (!response.ok) throw new Error(`Inference returned ${response.status}`);
        const prediction = await response.json();
        this.lastPrediction = prediction;
        this.render(prediction);
      } catch (_) {
        this.render({ status: "unavailable", unavailableReason: "Local service disconnected" });
      } finally {
        this.frameInFlight = false;
      }
    }

    async startSession(canvas) {
      this.lastPrediction = null;
      this.activeAccumulatedMs = 0;
      this.activeSegmentStartedAt = Date.now();
      this.paused = false;
      try {
        const response = await fetch(`${API_BASE}/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            game: "hextris",
            sourceUrl: location.href,
            canvasWidth: canvas.width,
            canvasHeight: canvas.height,
            overlayExpanded: !this.collapsed,
          }),
        });
        if (!response.ok) throw new Error(`Session start returned ${response.status}`);
        const session = await response.json();
        this.sessionId = session.id;
        this.collectionRunNumber = session.collectionRunNumber;
        this.collectionTarget = session.collectionTarget || 50;
        this.sessionMode = session.mode || "collection";
        this.render(this.sessionMode === "collection" ? this.collectionStatus() : { status: "warming_up" });
        this.startRecording(canvas);
      } catch (_) {
        this.activeSegmentStartedAt = null;
        this.detector.reset();
        this.render({ status: "unavailable", unavailableReason: "Start the local PlayLens service" });
      }
    }

    collectionStatus() {
      return {
        status: "collecting",
        collectionRunNumber: this.collectionRunNumber,
        collectionTarget: this.collectionTarget,
        modelSource: "collection_only",
      };
    }

    async setPaused(paused) {
      if (this.paused === paused) return;
      const now = Date.now();
      if (paused) {
        if (this.activeSegmentStartedAt !== null) this.activeAccumulatedMs += now - this.activeSegmentStartedAt;
        this.activeSegmentStartedAt = null;
        if (this.recorder?.state === "recording") this.recorder.pause();
      } else {
        this.activeSegmentStartedAt = now;
        this.previousSignature = null;
        if (this.recorder?.state === "paused") this.recorder.resume();
      }
      this.paused = paused;
      this.render(paused ? { status: "paused" } : this.sessionMode === "collection" ? this.collectionStatus() : { status: "warming_up" });
      try {
        await fetch(`${API_BASE}/sessions/${this.sessionId}/lifecycle`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ state: paused ? "paused" : "active", timestampMs: now, activeElapsedMs: this.activeElapsedMs(now) }),
        });
      } catch (_) {
        // Pause behavior remains local if lifecycle telemetry is temporarily unavailable.
      }
    }

    async finishSession(reason) {
      const sessionId = this.sessionId;
      if (!sessionId) return;
      const now = Date.now();
      const activeElapsedMs = this.activeElapsedMs(now);
      if (this.activeSegmentStartedAt !== null) this.activeAccumulatedMs = activeElapsedMs;
      this.activeSegmentStartedAt = null;
      this.sessionId = null;
      this.stopRecording(sessionId);
      this.detector.reset();
      this.previousSignature = null;
      this.startRequestedAt = null;
      let completedPrediction = this.lastPrediction;
      try {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}/finish`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          keepalive: reason === "navigation",
          body: JSON.stringify({ reason, timestampMs: now, activeElapsedMs }),
        });
        completedPrediction = response.ok ? await response.json() : completedPrediction;
      } catch (_) {}
      if (this.sessionId) return;
      if (reason === "navigation") {
        this.paused = false;
        this.activeAccumulatedMs = 0;
        return;
      }
      this.render({ ...(completedPrediction || {}), status: "game_over", sessionId });
      this.paused = false;
      this.activeAccumulatedMs = 0;
    }

    startRecording(canvas) {
      if (!canvas.captureStream || typeof MediaRecorder === "undefined") return;
      try {
        const stream = canvas.captureStream(15);
        const preferred = "video/webm;codecs=vp9";
        const mimeType = MediaRecorder.isTypeSupported(preferred) ? preferred : "video/webm";
        this.recordingChunks = [];
        this.recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 900_000 });
        this.recorder.ondataavailable = (event) => { if (event.data.size) this.recordingChunks.push(event.data); };
        this.recorder.start(1000);
      } catch (error) {
        console.warn("PlayLens recording unavailable", error);
      }
    }

    stopRecording(sessionId = null) {
      const recorder = this.recorder;
      this.recorder = null;
      if (!recorder || recorder.state === "inactive") return;
      recorder.onstop = async () => {
        if (!sessionId || this.recordingChunks.length === 0) return;
        const blob = new Blob(this.recordingChunks, { type: recorder.mimeType || "video/webm" });
        this.recordingChunks = [];
        try {
          await fetch(`${API_BASE}/sessions/${sessionId}/recording`, { method: "POST", headers: { "Content-Type": blob.type }, body: blob });
        } catch (error) {
          console.warn("PlayLens could not save the local recording", error);
        }
      };
      recorder.stop();
      recorder.stream.getTracks().forEach((track) => track.stop());
    }

    async setCollapsed(collapsed) {
      this.collapsed = collapsed;
      this.root?.toggleAttribute("data-collapsed", collapsed);
      this.shadow.querySelector(".pl-collapse").textContent = collapsed ? "+" : "−";
      if (!this.sessionId) return;
      try {
        await fetch(`${API_BASE}/sessions/${this.sessionId}/overlay`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expanded: !collapsed, timestampMs: Date.now() }),
        });
      } catch (_) {
        // The visual state remains local even when event logging is unavailable.
      }
    }

    mountOverlay() {
      const host = document.createElement("div");
      host.id = "playlens-extension-root";
      host.style.cssText = "position:fixed;right:22px;top:22px;z-index:2147483647;width:250px;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;";
      const shadow = host.attachShadow({ mode: "open" });
      shadow.innerHTML = `
        <style>
          * { box-sizing:border-box; }
          .card { --signal:#5dd39e; color:#f7f7f2; background:rgba(13,17,19,.88); border:1px solid color-mix(in srgb,var(--signal) 42%,transparent); border-radius:16px; box-shadow:0 18px 54px rgba(0,0,0,.34); backdrop-filter:blur(14px); overflow:hidden; user-select:none; }
          .card[data-state="warning"] { --signal:#e0a24a; }
          .card[data-state="imminent"] { --signal:#e05d55; }
          .card[data-state="paused"] { --signal:#82a8e8; }
          .head { min-height:42px; display:flex; align-items:center; gap:8px; padding:0 12px; cursor:grab; border-bottom:1px solid rgba(255,255,255,.10); }
          .brand { font-size:11px; letter-spacing:.13em; font-weight:800; }
          .mode { margin-left:auto; padding:5px 7px; border:1px solid rgba(255,255,255,.14); border-radius:99px; color:#aeb5b2; font-size:8px; font-weight:800; letter-spacing:.06em; }
          button { appearance:none; border:0; color:inherit; background:transparent; cursor:pointer; font:inherit; }
          .pl-collapse { width:24px; height:24px; color:#aeb5b2; font-size:18px; }
          .body { min-height:98px; padding:17px 14px; display:grid; align-content:center; gap:7px; }
          :host-context([data-collapsed]) .body { display:none; }
          .headline { color:var(--signal); font-size:18px; line-height:1.12; font-weight:880; letter-spacing:.025em; }
          .headline.large { font-size:22px; }
          .detail { color:#aeb5b2; font-size:10px; font-weight:750; letter-spacing:.075em; text-transform:uppercase; }
          .status { color:#c7cecb; font-size:12px; line-height:1.45; }
          .report { width:100%; margin-top:3px; padding:9px 10px; border-radius:9px; background:#f2f0e8; color:#141817; font-size:11px; font-weight:800; }
        </style>
        <section class="card" data-state="clear" aria-live="polite" aria-label="PlayLens loss-window forecast">
          <header class="head"><span class="brand">PLAYLENS</span><span class="mode">COLLECTION</span><button class="pl-collapse" type="button" aria-label="Collapse PlayLens">−</button></header>
          <div class="body"><div class="status">Waiting for a new run</div></div>
        </section>`;
      document.documentElement.appendChild(host);
      this.root = host;
      this.shadow = shadow;
      shadow.querySelector(".pl-collapse").addEventListener("click", () => void this.setCollapsed(!this.collapsed));
      this.enableDrag(shadow.querySelector(".head"), host);
    }

    enableDrag(handle, host) {
      let origin = null;
      handle.addEventListener("pointerdown", (event) => {
        if (event.target.closest("button")) return;
        const rect = host.getBoundingClientRect();
        origin = { x:event.clientX, y:event.clientY, left:rect.left, top:rect.top };
        handle.setPointerCapture(event.pointerId);
      });
      handle.addEventListener("pointermove", (event) => {
        if (!origin) return;
        host.style.left = `${Math.max(8, Math.min(window.innerWidth - host.offsetWidth - 8, origin.left + event.clientX - origin.x))}px`;
        host.style.top = `${Math.max(8, Math.min(window.innerHeight - host.offsetHeight - 8, origin.top + event.clientY - origin.y))}px`;
        host.style.right = "auto";
      });
      handle.addEventListener("pointerup", () => { origin = null; });
    }

    render(prediction) {
      if (!this.shadow) return;
      const body = this.shadow.querySelector(".body");
      const card = this.shadow.querySelector(".card");
      const mode = this.shadow.querySelector(".mode");
      mode.textContent = prediction.modelSource === "validated_personal_model" || this.sessionMode === "live" ? "VALIDATED" : "COLLECTION";

      if (prediction.status === "paused") {
        card.dataset.state = "paused";
        body.innerHTML = '<div class="headline">PAUSED</div><div class="detail">Monitoring suspended</div>';
        return;
      }
      if (prediction.status === "collecting") {
        card.dataset.state = "clear";
        const current = prediction.collectionRunNumber || this.collectionRunNumber || 1;
        const target = prediction.collectionTarget || this.collectionTarget;
        body.innerHTML = `<div class="headline large">CAPTURING RUN ${this.escape(current)} / ${this.escape(target)}</div><div class="detail">Personal training data</div>`;
        return;
      }
      if (prediction.status === "warming_up") {
        card.dataset.state = "clear";
        body.innerHTML = '<div class="status">Building eight seconds of gameplay context…</div>';
        return;
      }
      if (prediction.status === "unavailable") {
        card.dataset.state = "clear";
        body.innerHTML = `<div class="status">${this.escape(prediction.unavailableReason || "Not enough context")}</div>`;
        return;
      }

      if (prediction.status === "game_over") {
        const receipt = core.completionReceipt(prediction);
        card.dataset.state = receipt.accepted ? "clear" : "imminent";
        body.innerHTML = `<div class="headline">${this.escape(receipt.headline)}</div><div class="detail">${this.escape(receipt.detail)}</div><button class="report" type="button">Open session report</button>`;
        body.querySelector(".report")?.addEventListener("click", () => window.open(`${DASHBOARD_BASE}/sessions/${prediction.sessionId}`, "_blank", "noopener,noreferrer"));
        return;
      }

      const window = prediction.lossWindow || core.deriveLossWindow(core.HORIZONS_SECONDS.map((horizon) => prediction.failureProbabilities?.[String(horizon)] || 0));
      const labels = {
        none: "NO FAILURE SIGNAL",
        "45_to_60s": "LOSS LIKELY IN 45–60s",
        "30_to_45s": "LOSS LIKELY IN 30–45s",
        "20_to_30s": "LOSS LIKELY IN 20–30s",
        "10_to_20s": "LOSS LIKELY IN 10–20s",
        under_10s: "LOSS IMMINENT · <10s",
      };
      card.dataset.state = window === "under_10s" ? "imminent" : window === "none" ? "clear" : "warning";
      body.innerHTML = `<div class="headline">${this.escape(labels[window] || labels.none)}</div><div class="detail">${this.escape((prediction.certainty || "low").toUpperCase())} CERTAINTY</div>`;
    }

    escape(value) {
      const span = document.createElement("span");
      span.textContent = String(value);
      return span.innerHTML;
    }
  }

  const controller = new PlayLensController();
  globalThis.__playLensController = controller;
  const isLocalController = document.currentScript?.dataset.playlensLocalController === "true";
  if (isLocalController) void controller.arm();
  else controller.mountLauncher();
  if (globalThis.chrome?.runtime?.onMessage) {
    globalThis.chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
      if (message?.type !== "PLAYLENS_TOGGLE") return false;
      void controller.toggle().then(sendResponse);
      return true;
    });
  }
})();
