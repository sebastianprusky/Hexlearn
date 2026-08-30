export const horizonsSeconds = [5, 10, 15, 20, 30, 45, 60] as const;

export type ModelSource =
  | "collection_only"
  | "validated_personal_model"
  | "development_heuristic"
  | "validated_model"
  | "personalized_model";

export type LossWindow =
  | "none"
  | "under_10s"
  | "10_to_20s"
  | "20_to_30s"
  | "30_to_45s"
  | "45_to_60s";

export type Prediction = {
  timestampMs: number;
  activeElapsedMs: number | null;
  status: "ready";
  failureProbabilities: Record<string, number>;
  lossWindow: LossWindow | null;
  certainty: "low" | "medium" | "high" | null;
  estimatedRemainingSeconds: number | null;
  uncertaintyLowSeconds: number | null;
  uncertaintyHighSeconds: number | null;
  displayedValue: string | null;
  modelSource: ModelSource;
  calibrated: boolean;
  latencyMs: number;
};

export type Insight = {
  kind: "loss_signal" | "imminent" | "forecast_shift";
  startMs: number;
  endMs: number;
  startActiveMs: number | null;
  endActiveMs: number | null;
};

export type OverlayEvent = { timestampMs: number; expanded: boolean };
export type LifecycleEvent = {
  timestampMs: number;
  activeElapsedMs: number;
  state: "active" | "paused";
};

export type SessionSummary = {
  id: string;
  game: "hextris";
  sourceUrl: string;
  startedAt: string;
  endedAt: string | null;
  status: "active" | "complete";
  canvasWidth: number;
  canvasHeight: number;
  overlayExpandedInitial: boolean;
  endReason: string | null;
  hasRecording: boolean;
  modelSource: ModelSource;
  activeDurationMs: number;
  usableForTraining: boolean;
  qualityReason: string | null;
  collectionIndex: number | null;
  modelVersion: string | null;
  validFrameCount: number;
  expectedFrameCount: number;
  captureCoverage: number;
};

export type SessionDetail = SessionSummary & {
  predictions: Prediction[];
  insights: Insight[];
  overlayEvents: OverlayEvent[];
  lifecycleEvents: LifecycleEvent[];
  baselineDurationSeconds: number | null;
};

export type EvaluationMetrics = {
  meanBrier?: number;
  meanBrierImprovement?: number;
  worstCalibrationError?: number;
  timeToFailureMae?: number;
  windowAccuracy?: number;
  usefulWarningRateAt15s?: number;
  medianWarningLeadSeconds?: number;
  falseEarlyWarnings?: number;
  final10Mae?: number;
  final30Mae?: number;
  latencyMsP95?: number;
  evaluatedRuns?: number;
  brierImprovements?: Record<string, number>;
  timeToFailureImprovements?: Record<string, number>;
  horizons?: Record<string, {
    auroc?: number;
    brier?: number;
    ece?: number;
    strongestBaseline?: string;
    strongestBaselineBrier?: number;
    brierImprovement?: number;
  }>;
};

export type ModelStatus = {
  collector: {
    totalSessions: number;
    completeSessions: number;
    automatedSessions: number;
    manualSessions: number;
    totalFrames: number;
    trainingUse: "smoke_tests_only";
  };
  dataset: {
    ready: boolean;
    sessions: number;
    windows: number;
    failureRates: Record<string, number>;
    horizonsSeconds: number[];
    contextSeconds: number;
    minimumSessions: number;
    readyForTraining: boolean;
    source: "personal_clean_runs_only";
  };
  evaluations: Array<{
    id: string;
    model: string;
    split: string | null;
    eligibleForLive: boolean;
    metrics: EvaluationMetrics;
    gates: Record<string, boolean>;
    updatedAt: number;
  }>;
  personalCollection: {
    usableRuns: number;
    targetRuns: number;
    remainingRuns: number;
    developmentRuns: number;
    developmentTarget: number;
    lockedTestRuns: number;
    lockedTestTarget: number;
    phase: "development" | "locked_test" | "evaluation";
    excludedRuns: number;
    exclusionReasons: Record<string, number>;
    archivedLegacyRuns: number;
    cohortVersion: "collection-v3";
    recentRuns: Array<{
      id: string;
      startedAt: string;
      usable: boolean;
      reason: string;
      legacy: boolean;
      officialV3: boolean;
      collectionIndex: number | null;
      activeDurationMs: number;
      validFrameCount: number;
      expectedFrameCount: number;
      captureCoverage: number;
      pauseCount: number;
      hasRecording: boolean;
    }>;
  };
  trainingJob: {
    status: "idle" | "running" | "complete" | "failed";
    usableRuns?: number;
    step?: string;
    error?: string;
    completedSteps?: string[];
  };
  checkpoints: Array<{
    usableRuns: number;
    createdAt: string | null;
    sessionCount: number;
    path: string;
  }>;
  liveModel: {
    source: ModelSource;
    calibrated: boolean;
    artifactPresent: boolean;
    collectionOnly: boolean;
  };
};
