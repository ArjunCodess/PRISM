export type Factor = {
  feature: string;
  direction: "higher" | "lower" | string;
  contribution: number;
  label?: string;
};

export type Prediction = {
  eventId?: string;
  predictedFinalRiskLog10: number;
  predictedFinalPc: number;
  interval90Log10: [number, number] | number[];
  configuredHighRiskProbability: number;
  highRiskThresholdLog10: number;
  riskBand: string;
  abstained: boolean;
  topFactors: Factor[];
  explanation?: string;
  modelVersion: string;
  disclaimer: string;
};

export type CdmMessage = {
  timeToTcaDays: number;
  riskLog10: number;
  missDistanceM: number;
  relativeSpeedMps: number;
  tSigmaR?: number;
  cSigmaR?: number;
  tObsUsed?: number;
  cObsUsed?: number;
  cObjectType?: string;
};

export type DemoCase = {
  id: string;
  story: string;
  missionAlias: string;
  title: string;
  prediction: Prediction;
  baselineRiskLog10: number;
  actualFinalRiskLog10: number;
  messages: CdmMessage[];
  futureMessages: CdmMessage[];
};

export type MetricsFile = {
  ensemble: Record<string, number>;
  persistence: Record<string, number>;
  improvement: Record<string, number>;
  warning: Record<string, number>;
  splits: Record<string, number>;
  calibration?: Array<{ mid: number; predicted: number; observed: number; n: number }>;
  ablation?: Record<string, number>;
  featureGroups?: Array<{ group: string; gain: number }>;
  failures?: {
    worstUnderpredictions: Array<Record<string, number>>;
    worstOverpredictions: Array<Record<string, number>>;
    missedHighRisk: Array<Record<string, number>>;
    falseEscalations: Array<Record<string, number>>;
  };
};
