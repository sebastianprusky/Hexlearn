(function installPlayLensTrainingBridge() {
  "use strict";

  const API = "http://127.0.0.1:8000/api/v1/training";
  const params = new URLSearchParams(location.search);
  const automated = params.get("playlensBot") === "1";
  const requestedRuns = Number.parseInt(params.get("runs") || "10", 10);
  const requestedSeed = Number.parseInt(params.get("seed") || "7", 10);
  const baseSeed = Number.isFinite(requestedSeed) ? Math.max(0, requestedSeed) : 7;
  const policies = ["survivor", "balanced", "explorer", "adversarial"];
  const targetRuns = Math.max(1, Math.min(200, Number.isFinite(requestedRuns) ? requestedRuns : 10));
  let trainingId = null;
  let previousState = null;
  let uploadInFlight = false;
  let statePollInFlight = false;
  let completedRuns = 0;
  let restarting = false;
  let currentPolicy = policies[0];
  let currentSeed = baseSeed;
  let random = Math.random;

  const seededRandom = (seed) => {
    let value = seed >>> 0;
    return () => {
      value += 0x6d2b79f5;
      let mixed = value;
      mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
      mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
      return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
  };

  const configureRun = () => {
    currentPolicy = policies[completedRuns % policies.length];
    currentSeed = baseSeed + completedRuns * 9973;
    random = seededRandom(currentSeed);
  };

  const status = document.createElement("aside");
  status.id = "playlens-training-status";
  Object.assign(status.style, {
    position: "fixed",
    top: "16px",
    right: "16px",
    zIndex: "2147483647",
    minWidth: "210px",
    padding: "12px 14px",
    border: "1px solid rgba(255,255,255,.28)",
    borderRadius: "12px",
    color: "#eef7f3",
    background: "rgba(18,23,22,.86)",
    boxShadow: "0 12px 35px rgba(0,0,0,.2)",
    font: "12px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif",
    pointerEvents: "none",
  });
  document.body.appendChild(status);

  const renderStatus = (message) => {
    const runLabel = automated ? `${Math.min(completedRuns + 1, targetRuns)} / ${targetRuns} · ${currentPolicy}` : "manual";
    status.innerHTML = `<strong style="display:block;margin-bottom:3px">PlayLens collector</strong>
      <span style="color:#aebdb7">Run ${runLabel} &middot; ${message}</span>`;
  };

  const capture = () => {
    const source = document.querySelector("canvas#canvas");
    if (!(source instanceof HTMLCanvasElement)) return null;
    const target = document.createElement("canvas");
    target.width = 160;
    target.height = 160;
    const context = target.getContext("2d");
    if (!context) return null;
    context.fillStyle = "#000";
    context.fillRect(0, 0, 160, 160);
    const scale = Math.min(160 / source.width, 160 / source.height);
    const width = source.width * scale;
    const height = source.height * scale;
    context.drawImage(source, (160 - width) / 2, (160 - height) / 2, width, height);
    return target.toDataURL("image/jpeg", 0.72);
  };

  const startTrainingSession = async () => {
    const response = await fetch(`${API}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "instrumented_hextris",
        automated,
        policy: automated ? currentPolicy : null,
        seed: automated ? currentSeed : null,
      }),
    });
    if (!response.ok) throw new Error(`PlayLens training start failed: ${response.status}`);
    trainingId = (await response.json()).id;
    renderStatus("capturing labels");
  };

  const finishTrainingSession = async () => {
    const id = trainingId;
    trainingId = null;
    if (!id) return;
    const response = await fetch(`${API}/sessions/${id}/finish`, { method: "POST" });
    if (!response.ok) throw new Error(`PlayLens training finish failed: ${response.status}`);
    completedRuns += 1;
    renderStatus(completedRuns >= targetRuns ? "collection complete" : "run saved");
  };

  const beginBotRun = () => {
    if (!automated || restarting || completedRuns >= targetRuns) return;
    restarting = true;
    try {
      configureRun();
      if (typeof clearSaveState === "function") clearSaveState();
      if (Number(window.gameState) === 0 && typeof resumeGame === "function") {
        resumeGame();
      } else if (typeof init === "function") {
        init(1);
        if (window.jQuery) window.jQuery("#gameoverscreen").hide();
      }
      renderStatus("starting");
    } finally {
      window.setTimeout(() => { restarting = false; }, 700);
    }
  };

  const chooseTargetLane = (fallingBlock) => {
    if (!window.MainHex || !Array.isArray(window.MainHex.blocks)) return null;
    const profile = {
      survivor: { exploration: 0.02, heightWeight: 7.0, matchWeight: 9.0 },
      balanced: { exploration: 0.10, heightWeight: 5.0, matchWeight: 6.0 },
      explorer: { exploration: 0.25, heightWeight: 3.5, matchWeight: 3.5 },
      adversarial: { exploration: 0.45, heightWeight: -1.5, matchWeight: 1.0 },
    }[currentPolicy];
    if (random() < profile.exploration) return Math.floor(random() * window.MainHex.sides);
    let bestLane = 0;
    let bestScore = Number.POSITIVE_INFINITY;
    window.MainHex.blocks.forEach((rawStack, lane) => {
      const stack = rawStack.filter((block) => block && block.deleted !== 2);
      let matchingTop = 0;
      for (let index = stack.length - 1; index >= 0; index -= 1) {
        if (stack[index].color !== fallingBlock.color) break;
        matchingTop += 1;
      }
      const landingIndex = stack.length;
      let matchingNeighbors = matchingTop;
      for (const offset of [-1, 1]) {
        const adjacent = window.MainHex.blocks[(lane + offset + window.MainHex.sides) % window.MainHex.sides]
          .filter((block) => block && block.deleted !== 2);
        if (adjacent[landingIndex]?.color === fallingBlock.color) matchingNeighbors += 1;
      }
      const laneScore =
        stack.length * profile.heightWeight -
        matchingNeighbors * profile.matchWeight +
        random() * 0.35;
      if (laneScore < bestScore) {
        bestScore = laneScore;
        bestLane = lane;
      }
    });
    return bestLane;
  };

  window.setInterval(() => {
    if (!automated || Number(window.gameState) !== 1 || !window.MainHex || !Array.isArray(window.blocks)) return;
    const fallingBlock = window.blocks
      .filter((block) => block && !block.settled && !block.removed && !block.deleted)
      .sort((left, right) => left.distFromHex - right.distFromHex)[0];
    if (!fallingBlock) return;
    const targetLane = chooseTargetLane(fallingBlock);
    if (targetLane === null) return;
    const projectedLane = (window.MainHex.sides - fallingBlock.fallingLane + window.MainHex.position) % window.MainHex.sides;
    const clockwiseDistance = (targetLane - projectedLane + window.MainHex.sides) % window.MainHex.sides;
    if (clockwiseDistance === 0) return;
    window.MainHex.rotate(clockwiseDistance <= window.MainHex.sides / 2 ? 1 : -1);
  }, 110);

  window.setInterval(async () => {
    if (statePollInFlight) return;
    statePollInFlight = true;
    try {
      const state = Number(window.gameState);
      if (automated && completedRuns < targetRuns && state === 0 && !trainingId) beginBotRun();
      if (state === 1 && previousState !== 1 && !trainingId) await startTrainingSession();
      if (previousState === 1 && state === 2 && trainingId) {
        await finishTrainingSession();
        if (automated && completedRuns < targetRuns) window.setTimeout(beginBotRun, 900);
      }
      previousState = state;

      if (!trainingId || state !== 1 || uploadInFlight) return;
      const imageDataUrl = capture();
      if (!imageDataUrl) return;
      uploadInFlight = true;
      try {
        const response = await fetch(`${API}/sessions/${trainingId}/frames`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            timestampMs: Date.now(),
            imageDataUrl,
            gameState: state,
            score: Math.max(0, Number(window.score) || 0),
          }),
        });
        if (!response.ok) throw new Error(`PlayLens training frame failed: ${response.status}`);
      } finally {
        uploadInFlight = false;
      }
    } catch (error) {
      console.warn(error);
      renderStatus("API unavailable");
    } finally {
      statePollInFlight = false;
    }
  }, 250);

  renderStatus(automated ? "waiting to start" : "ready for play");
})();
