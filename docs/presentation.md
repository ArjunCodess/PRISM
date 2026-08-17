# PRISM presentation notes

Print this. Keep it next to the keyboard. Numbers stay the same even if your brain does not.

Model version `prism-0.3.0`. Selected policy: hurdle residual (MAE XGBoost on `Δ = y − today's risk`) mixed with a collapse-to-floor classifier, persistence guard at −6.

## If you blank

Say this and then open the abstention case:

"Two objects might get close. The reported collision chance keeps changing as new tracking data arrives. Waiting gives a better number, and it also eats planning time. PRISM stops the clock 48 hours early and asks: can the history of those reports beat just copying today's value? On average, yes: MAE falls from 5.080 to 2.800. On the ESA high-risk score, it ties at 0.167, because if today's report is already at −6 or worse, PRISM copies it on purpose. The ranges on screen are split-conformal bands. They cover about 91 percent of outcomes. This is a school research exhibit. A person still has to review it. Two of nine high-risk test events still slipped through as accepted safe forecasts."

Then click Reveal. Breathe. Keep going.

## What this project actually is

PRISM is Predictive Risk Intelligence for Space Monitoring. Fancy name. Simple job.

It forecasts the later reported `log10(Pc)` from messages you would have had 48 hours before closest approach. That later number is a report from the conjunction process. It is not the true physical probability of a crash, and PRISM does not compute orbits from first principles.

The baseline is persistence. Persistence means: take the latest pre-cutoff risk and copy it forward. If a learned model cannot beat that, it is not adding much.

Data comes from the public ESA Collision Avoidance Challenge archive (2015 to 2019). Real CDMs. Anonymized events. 162,634 rows. 13,154 events. After the cutoff rule, 8,293 events are eligible. The frozen test split has 1,659 events.

T-48 exists because the ESA challenge test file only has messages with `time_to_tca` of at least 2 days. Hence we freeze there. We also report T-72, T-24, and T-12 so you can see what waiting does.

The high-risk line `log10(Pc) >= -6` is the ESA scoring class. It is not an ISRO operations rule. Only 66 eligible events meet it. Nine of those sit in the test split. Therefore that probability meter is a scarce-label estimate. Treat it that way when you talk.

## How the pipeline works

1. Keep only messages at or before the cutoff.
2. The later update is the label. It never goes into features.
3. Train, validation, calibration, and test use different event IDs. Validation is not reused to pick the test set.
4. Each event becomes one row of inspectable summaries: latest snapshot (including geometry and object-type flags), plus slopes, changes, ranges, and observation-count history of those same snapshot fields.
5. Fit a residual XGBoost with MAE loss. Also fit median, Ridge, persistence, a collapse-to-floor classifier, and a warning classifier.
6. Mix: if collapse probability is high, predict the floor of −30; otherwise add the residual to today's risk. If today's report is already at or above −6, copy that report. Invented highs are clamped just below −6.
7. A calibration split builds split-conformal bands (floor vs moving) and maps a warning score to "chance this later report is high-risk."
8. Abstain when the 90 percent conformal band crosses −6, a critical field is missing, bootstrap models disagree by more than 1.25 log units, the model dumps to the floor while today's report is still elevated, or the warning head says high-risk while the point forecast stays safe.

You can say "hurdle residual XGBoost on named features" if someone asks why it is not a transformer. 8,293 events. Lots of missing cells. We wanted to read the reasons. A sequence model can wait until it wins on the same frozen split.

## The result you should lead with

Held-out test, 1,659 events. MAE is in `log10(Pc)` units.

- Persistence MAE: 5.080
- PRISM MAE: 2.800 (about 45 percent lower)
- ESA-style loss: 0.167 for both
- F2: 0.361 for both
- 90 percent conformal coverage: 91.4 percent

ESA-style loss is high-risk MSE divided by F2. F2 is F-beta with beta 2, so missing a high-risk event hurts more than a false alarm.

The exact tie is not a bug. The guard copies today's report once it is already at or above −6. That is exactly the region ESA-style loss cares about. Therefore PRISM matches persistence there on purpose. The MAE win is mostly the boring middle of the risk range, including learning when reports collapse to −30.

One sentence that carries the whole talk:

"The model has useful signal beyond persistence, especially when the forecast is early, but that signal does not automatically become better high-risk decisions."

Say that. Then show the horizon table.

## Horizon table (put this on a slide)

Waiting helps persistence more than it helps PRISM. Learned forecasting is most useful when information is sparse. These are hurdle numbers, not the unguarded booster.

| Horizon | Hurdle MAE | Persistence MAE |
| --- | ---: | ---: |
| T-72 | 4.578 | 7.748 |
| T-48 | 2.800 | 5.080 |
| T-24 | 2.199 | 2.634 |
| T-12 | 1.385 | 1.444 |

At T-72 the gap is huge. At T-12 it is basically gone. That is the finding people remember.

## Other numbers worth having in a pocket

- Unguarded MAE XGBoost: 2.550 MAE, F2 = 0. Median baseline: 3.002.
- History vs snapshot (MAE residual ablation): 2.509 snapshot, 2.535 after history, 2.526 after covariance trends. History does not cut average MAE under this loss.
- Abstention: 89.0 percent coverage, 183 sent to review, accepted MAE 2.085. Seven of nine test high-risk events go to review.
- False reassurance: an accepted forecast below −6 while the final report is at or above −6. Count is **two**.
- Nominal 90 percent conformal band covers 91.4 percent of outcomes. 50 percent band covers 82.0 percent.
- Mission holdout overall MAE: 3.588 vs persistence 4.843. One held-out high-risk event; high-risk MAE 0.114, tied.
- `mission_id` changes unguarded MAE from 2.550 to 2.570. We left it out of the exhibit.
- Failures: floor collapse 135, under-prediction 107, over-prediction 41, moderate 38, sparse history 28, false high-risk 20, close approach 13, late high-risk jumps 2. Accurate within 0.5: 1,275.
- On large errors, risk trend and tracking completeness carry more mean |SHAP| than today's reported risk. That is association in the explanation, not a physics proof.

## What to click

Use the abstention case for the short path. Use the lab if they want receipts.

Start FastAPI and Next before anyone sits down. `NEXT_PUBLIC_API_URL` must be set. If the API is down, the page errors. That is intended.

### 90 seconds

Open the abstention case.

"These six cases are real ESA events: two low, one that needs a person, three high. The page is frozen before T-48. You see today's reported risk, the live forecast of the later report, the high-risk estimate, and a conformal band. If the band crosses −6, or the model dumps to the floor while today's number is still worrying, PRISM says review required. SHAP tells us why the residual booster moved. It does not tell us what physically caused the encounter. The future is hidden on purpose."

Click Reveal.

"Later messages and the final reported value show up now. Reveal is the only way to load that truth."

If there is time: "In the lab, MAE drops from 5.080 to 2.800. ESA-style loss ties at 0.167 because of the guard. The 90 percent band covers 91.4 percent of outcomes. Two of nine high-risk events still got through as accepted safe forecasts."

### 3 minutes

Add the data path (162,634 to 8,293 to 1,659), the leakage rule, persistence as the baseline, and the horizon story. Then the lab. Then false reassurance = 2. Then the mission holdout. Then sit down.

### 7 minutes

Walk the same path slower. Open the lab tables in this order:

1. Horizon. "Useful early. Persistence catches up late."
2. History ablation. "Under MAE loss, extra history does not buy average error. Geometry and object type are in the snapshot."
3. Abstention. "89 percent coverage. Define false reassurance. Two of those."
4. Spread coverage. "91.4 percent on a 90 percent conformal band. We measured it."
5. Failures and SHAP contrast.

Close with the one-sentence finding. Credit ESA, ISRO NETRA, NASA CARA, and CCSDS as context. They did not endorse this project. You built it at City Montessori School, Kanpur Road Campus.

## How the live website works

The six cases are curated real events, not made-up rows. The site requires `NEXT_PUBLIC_API_URL` and talks to FastAPI.

- The queue and the lab load cases and metrics from the API.
- Opening a case runs `POST /v1/risk/predict` on the cutoff-safe messages. That is a live hurdle inference call.
- Reveal still hides the later messages until you ask. That part is a research control, not a fake forecast.

If the API is down, the site fails in the open. There is no frozen JSON fallback. For a Wi-Fi-off rehearsal, keep FastAPI running on the laptop.

## Answers when they interrupt you

**Did you beat the 2019 ESA winners?** We cannot say. Official test labels are still hidden. We report our own frozen split.

**Why is MAE so big?** Some events jump between the dataset floor of −30 and a finite risk. A few of those wreck the mean. Median absolute error on the selected policy is 0. That is why both numbers exist.

**Is the 90 percent band honest now?** Coverage is 91.4 percent on this split. It is split conformal, localized by floor vs moving. Still not an operational probability.

**Why keep persistence?** It protects the high-risk tail. Dropping the guard would make average error look nicer and would weaken the part ESA actually scores.

**Why a hurdle instead of one big ensemble?** Direct regression was mostly learning the −30 floor. Residual plus collapse classifier is the same story, told in two heads.

**Why XGBoost?** Named features. Missing values. 8,293 rows. We wanted explanations we can read.

**Does this fly a satellite?** It is a research exhibit. A qualified person has to review anything that looks like a decision.

**Would it work on a mission you have never seen?** We tried holding out four missions. Overall MAE still beats persistence. The rare high-risk tail is one event. That is not proof.

**You said you never miss high-risk on accepted cases.** That was the old bootstrap exhibit. This split has two false-reassurance events. Say that before they find it in the lab.

## Rehearsal still on you

Wi-Fi off is fine if FastAPI and Next are both local. Projector at 1920x1080. Time a cold start. Time a case load. Time one live predict. Record a 90-second backup video on the presentation laptop. The repo cannot do that part for you.
