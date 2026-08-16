CUTOFF_DAYS = 2.0
CUTOFF_HOURS = 48
HORIZON_HOURS = (72, 48, 24, 12)
# The −6 class follows the ESA challenge definition. The persistence guard and
# 1.25 disagreement threshold are fixed design choices, not test-tuned.
HIGH_RISK_THRESHOLD = -6.0
LOW_RISK_CLIP = -6.001
NEGLIGIBLE_RISK = -30.0
RANDOM_STATE = 42
MODEL_VERSION = "prism-0.2.0"
ABSTENTION_DISAGREEMENT = 1.25
DISCLAIMER = (
    "Research prototype for offline, explainable conjunction-risk forecasting. "
    "Not flight software. Not an operational decision system."
)
ABSTENTION_RULE = (
    "PRISM abstains when the 90% bootstrap band crosses the ESA challenge class "
    "log10(Pc) ≥ −6, when current risk or miss distance is missing, or when "
    "bootstrap disagreement exceeds 1.25 log-risk units. The −6 class follows "
    "the ESA challenge definition. The persistence guard and 1.25 disagreement "
    "threshold were fixed design choices before evaluating the test split."
)
FALSE_REASSURANCE_DEFINITION = (
    "An accepted forecast (no abstention) with predicted log10(Pc) < −6 while "
    "the final reported value is ≥ −6."
)
ESA_LOSS_DEFINITION = (
    "ESA-style loss is high-risk MSE divided by F2 (F-beta with β=2), so recall "
    "of the log10(Pc) ≥ −6 class is weighted more than precision."
)
RESEARCH_QUESTION = (
    "Do pre-T−48 conjunction histories contain enough predictive signal to "
    "improve forecasts of later reported log10(Pc) over persistence?"
)

TREND_COLUMNS = [
    "risk",
    "miss_distance",
    "relative_speed",
    "max_risk_estimate",
    "t_sigma_r",
    "t_sigma_t",
    "t_sigma_n",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "log_t_position_covariance_det",
    "log_c_position_covariance_det",
    "t_obs_used",
    "c_obs_used",
]

SNAPSHOT_COLUMNS = [
    "risk",
    "max_risk_estimate",
    "max_risk_scaling",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
    "azimuth",
    "elevation",
    "geocentric_latitude",
    "c_object_type",
    "mission_id",
    "time_to_tca",
    "F10",
    "F3M",
    "AP",
    "SSN",
    "t_sigma_r",
    "t_sigma_t",
    "t_sigma_n",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "t_sigma_rdot",
    "t_sigma_tdot",
    "t_sigma_ndot",
    "c_sigma_rdot",
    "c_sigma_tdot",
    "c_sigma_ndot",
    "t_span",
    "c_span",
    "t_rcs_estimate",
    "c_rcs_estimate",
    "t_ecc",
    "c_ecc",
    "t_j2k_inc",
    "c_j2k_inc",
    "t_j2k_sma",
    "c_j2k_sma",
    "t_h_apo",
    "c_h_apo",
    "t_h_per",
    "c_h_per",
    "t_obs_available",
    "c_obs_available",
    "t_obs_used",
    "c_obs_used",
    "t_actual_od_span",
    "c_actual_od_span",
    "t_recommended_od_span",
    "c_recommended_od_span",
    "t_weighted_rms",
    "c_weighted_rms",
    "t_cd_area_over_mass",
    "c_cd_area_over_mass",
]

FEATURE_DICTIONARY = {
    "risk": "today's reported chance",
    "risk_change": "whether chance rose or fell",
    "risk_slope": "whether chance is climbing",
    "risk_delta_last3": "change across the last three updates",
    "miss_distance": "predicted miss distance",
    "relative_speed": "closing speed",
    "normalized_separation": "miss size versus uncertainty",
    "t_obs_used": "satellite observations used",
    "c_obs_used": "other-object observations used",
    "t_position_covariance_det": "satellite position uncertainty",
    "c_position_covariance_det": "other-object position uncertainty",
    "hours_before_cutoff": "hours left before the 48-hour line",
    "n_messages": "early messages available",
    "max_risk_estimate": "most pessimistic chance in the message",
    "derived_miss_distance": "miss distance from the geometry",
    "t_sigma_r": "satellite radial uncertainty",
    "c_sigma_r": "other-object radial uncertainty",
}

STORY_COPY = {
    "low": {
        "title": "Easy correct prediction",
        "blurb": "Wide miss. Persistence and the forecast both stay calm, and both are right.",
    },
    "escalate": {
        "title": "Hard correct prediction",
        "blurb": "The history is moving. The forecast beats carrying the latest report forward.",
    },
    "deescalate": {
        "title": "Early scare, then calmer",
        "blurb": "First alerts looked high; later geometry looks safer.",
    },
    "uncertain": {
        "title": "The model refuses to claim",
        "blurb": "Ensemble spread crosses the ESA class, so PRISM asks for human review.",
    },
    "failure": {
        "title": "Confident failure",
        "blurb": "The model did not abstain, and the later update shows it was wrong.",
    },
}
