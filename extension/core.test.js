const test = require("node:test");
const assert = require("node:assert/strict");
const {
  HORIZONS_SECONDS,
  averageMotion,
  claimControllerOwnership,
  completionReceipt,
  deriveLossWindow,
  lossWindowSeconds,
  monotonicProbabilities,
  supportedGameUrl,
  RunDetector,
} = require("./core.js");

test("only one controller can claim a page across execution worlds", () => {
  const attributes = new Map();
  const root = {
    hasAttribute: (name) => attributes.has(name),
    setAttribute: (name, value) => attributes.set(name, value),
  };
  assert.equal(claimControllerOwnership(root, "extension"), true);
  assert.equal(claimControllerOwnership(root, "local-fallback"), false);
});

test("completion receipts distinguish accepted and excluded runs", () => {
  assert.deepEqual(
    completionReceipt({ usableForTraining: true, usableRuns: 3, collectionTarget: 50, captureCoverage: 0.96 }),
    { accepted: true, headline: "RUN ACCEPTED", detail: "3 / 50 usable · 96% capture" },
  );
  assert.deepEqual(
    completionReceipt({ usableForTraining: false, qualityReason: "excluded: capture coverage below 75%" }),
    { accepted: false, headline: "RUN EXCLUDED", detail: "capture coverage below 75%" },
  );
});

test("seven-horizon survival probabilities are monotonic", () => {
  const values = monotonicProbabilities([0.4, 0.2, 0.5, 0.45, 0.8, 0.7, 0.9]);
  assert.equal(values.length, HORIZONS_SECONDS.length);
  assert.deepEqual(values, [0.4, 0.4, 0.5, 0.5, 0.8, 0.8, 0.9]);
});

test("loss window uses the first 50 percent crossing", () => {
  assert.equal(deriveLossWindow([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]), "10_to_20s");
  assert.equal(deriveLossWindow([0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.49]), "none");
  assert.equal(lossWindowSeconds("20_to_30s"), 25);
});

test("motion is the mean absolute pixel difference", () => {
  assert.equal(averageMotion([0, 10], [10, 20]), 10);
});

test("run detector pauses and resumes one session", () => {
  const detector = new RunDetector({ startThreshold: 2 });
  assert.equal(detector.update({ motion: 3 }), null);
  assert.equal(detector.update({ motion: 3 }), null);
  assert.equal(detector.update({ motion: 3 }), "started");
  assert.equal(detector.update({ motion: 0, paused: true }), "paused");
  assert.equal(detector.state, "paused");
  assert.equal(detector.update({ motion: 0, paused: true }), null);
  assert.equal(detector.update({ motion: 2 }), "resumed");
  assert.equal(detector.state, "active");
});

test("visible in-progress UI starts immediately without a motion threshold", () => {
  const detector = new RunDetector({ startThreshold: 100 });
  assert.equal(detector.update({ motion: 0, playing: true }), "started");
  assert.equal(detector.state, "active");
});

test("reset after navigation permits a restored page to start again", () => {
  const detector = new RunDetector();
  assert.equal(detector.update({ motion: 0, playing: true }), "started");
  detector.reset();
  assert.equal(detector.update({ motion: 0, playing: true }), "started");
});

test("stillness never ends a run; visible game over does", () => {
  const detector = new RunDetector({ startThreshold: 2 });
  detector.update({ motion: 3 });
  detector.update({ motion: 3 });
  detector.update({ motion: 3 });
  for (let index = 0; index < 30; index += 1) assert.equal(detector.update({ motion: 0 }), null);
  assert.equal(detector.state, "active");
  assert.equal(detector.update({ motion: 0, gameOver: true }), "ended");
  assert.equal(detector.state, "idle");
});

test("hidden active tabs pause instead of ending", () => {
  const detector = new RunDetector({ startThreshold: 2 });
  detector.update({ motion: 3 });
  detector.update({ motion: 3 });
  detector.update({ motion: 3 });
  assert.equal(detector.update({ motion: 0, hidden: true }), "paused");
  assert.equal(detector.update({ motion: 0, hidden: false }), "resumed");
});

test("only the clean local game origin is supported", () => {
  assert.equal(supportedGameUrl("http://127.0.0.1:8000/"), true);
  assert.equal(supportedGameUrl("http://localhost:8000/index.html"), true);
  assert.equal(supportedGameUrl("http://127.0.0.1:8001/"), false);
  assert.equal(supportedGameUrl("https://hextris.github.io/hextris/"), false);
});
