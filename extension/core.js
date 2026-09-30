(function attachPlayLensCore(globalScope) {
  "use strict";

  const HORIZONS_SECONDS = [5, 10, 15, 20, 30, 45, 60];
  const CONTROLLER_LOCK_ATTRIBUTE = "data-playlens-controller-owner";

  const claimControllerOwnership = (root, ownerId) => {
    if (!root || typeof root.hasAttribute !== "function" || typeof root.setAttribute !== "function") {
      return false;
    }
    if (root.hasAttribute(CONTROLLER_LOCK_ATTRIBUTE)) return false;
    root.setAttribute(CONTROLLER_LOCK_ATTRIBUTE, String(ownerId || "playlens"));
    return true;
  };

  const monotonicProbabilities = (values) => {
    const probabilities = values.map((value) => Math.max(0, Math.min(1, Number(value) || 0)));
    return probabilities.map((value, index) => Math.max(value, ...probabilities.slice(0, index)));
  };

  const deriveLossWindow = (values, threshold = 0.5) => {
    const probabilities = monotonicProbabilities(values);
    const crossing = probabilities.findIndex((value) => value >= threshold);
    if (crossing < 0) return "none";
    const horizon = HORIZONS_SECONDS[crossing];
    if (horizon <= 10) return "under_10s";
    if (horizon <= 20) return "10_to_20s";
    if (horizon <= 30) return "20_to_30s";
    if (horizon <= 45) return "30_to_45s";
    return "45_to_60s";
  };

  const lossWindowSeconds = (window) => ({
    under_10s: 5,
    "10_to_20s": 15,
    "20_to_30s": 25,
    "30_to_45s": 37.5,
    "45_to_60s": 52.5,
    none: 60,
  })[window] ?? 60;

  const supportedGameUrl = (value) => {
    try {
      const url = new URL(value);
      return (
        url.protocol === "http:" &&
        url.port === "8000" &&
        (url.hostname === "127.0.0.1" || url.hostname === "localhost")
      );
    } catch (_) {
      return false;
    }
  };

  const completionReceipt = (result = {}) => {
    const coverage = Number(result.captureCoverage);
    const coverageText = Number.isFinite(coverage) ? ` · ${Math.round(coverage * 100)}% capture` : "";
    if (result.usableForTraining) {
      return {
        accepted: true,
        headline: "RUN ACCEPTED",
        detail: `${Number(result.usableRuns) || 0} / ${Number(result.collectionTarget) || 40} usable${coverageText}`,
      };
    }
    const reason = String(result.qualityReason || "Run did not pass capture checks")
      .replace(/^excluded:\s*/i, "")
      .replace(/^legacy:\s*/i, "");
    return { accepted: false, headline: "RUN EXCLUDED", detail: reason };
  };

  class RunDetector {
    constructor({ startThreshold = 1.8 } = {}) {
      this.startThreshold = startThreshold;
      this.state = "idle";
      this.motionStreak = 0;
    }

    update({ motion, hidden = false, paused = false, gameOver = false, playing = false }) {
      if (this.state === "idle") {
        if (hidden || paused || gameOver) {
          this.motionStreak = 0;
          return null;
        }
        if (playing) {
          this.state = "active";
          this.motionStreak = 0;
          return "started";
        }
        this.motionStreak = motion >= this.startThreshold ? this.motionStreak + 1 : 0;
        if (this.motionStreak >= 3) {
          this.state = "active";
          this.motionStreak = 0;
          return "started";
        }
        return null;
      }

      if (gameOver) {
        this.state = "idle";
        this.motionStreak = 0;
        return "ended";
      }
      if (hidden || paused) {
        if (this.state !== "paused") {
          this.state = "paused";
          return "paused";
        }
        return null;
      }
      if (this.state === "paused") {
        this.state = "active";
        return "resumed";
      }
      return null;
    }

    reset() {
      this.state = "idle";
      this.motionStreak = 0;
    }
  }

  const averageMotion = (previous, current) => {
    if (!previous || previous.length !== current.length) return 0;
    let difference = 0;
    for (let index = 0; index < current.length; index += 1) {
      difference += Math.abs(current[index] - previous[index]);
    }
    return difference / current.length;
  };

  const core = {
    HORIZONS_SECONDS,
    CONTROLLER_LOCK_ATTRIBUTE,
    averageMotion,
    claimControllerOwnership,
    completionReceipt,
    deriveLossWindow,
    lossWindowSeconds,
    monotonicProbabilities,
    supportedGameUrl,
    RunDetector,
  };
  globalScope.PlayLensCore = core;
  if (typeof module !== "undefined" && module.exports) module.exports = core;
})(typeof globalThis !== "undefined" ? globalThis : this);
