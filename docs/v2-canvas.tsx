import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
} from "cursor/canvas";

type Tab = "finding" | "literature" | "work" | "venues" | "plan";

const HORIZON_MAE = {
  categories: ["T−72", "T−48", "T−24", "T−12"],
  series: [
    { name: "Single XGBoost MAE (log10 Pc)", data: [3.214, 2.808, 2.11, 1.384], tone: "info" as const },
    { name: "Persistence MAE (log10 Pc)", data: [7.748, 5.08, 2.634, 1.444], tone: "neutral" as const },
  ],
};

const HORIZON_GAIN = {
  categories: ["T−72", "T−48", "T−24", "T−12"],
  series: [
    {
      name: "MAE advantage over persistence (log10 Pc)",
      data: [4.534, 2.271, 0.524, 0.06],
      tone: "success" as const,
    },
  ],
};

export default function PrismBestVersion() {
  const [tab, setTab] = useCanvasState<Tab>("tab", "finding");

  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>PRISM v2 — a citable measurement</H1>
        <Text tone="secondary">
          Best version: a leakage-safe study of how conjunction-message
          information decays. One policy: the API, the website, and the paper
          all serve it. Persistence and the August ensemble are baselines,
          not a second product.
        </Text>
        <Row gap={8} wrap>
          <Pill active>branch v2/information-decay</Pill>
          <Pill>
            PR: v2: Measure conjunction information decay (horizon, floor,
            official test)
          </Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="42 days" label="to CJSJ (30 Sep 2026)" />
        <Stat value="9 vs ~150" label="high-risk events, local split vs official test" tone="warning" />
        <Stat value="0.000" label="persistence median AE at T−48" tone="warning" />
        <Stat value="2021" label="year ESA released official-test labels" tone="success" />
      </Grid>

      <Callout tone="success" title="Highest-leverage fact found after the first pass">
        Zenodo record 4463683 (25 Jan 2021) released the labels that were hidden
        in 2019. v1 only schema-checks Kelvins test_data.csv. The official test
        over-represents high-risk events (~150 positives). That is the split
        where ESA-style loss is estimable. Score it once, after every choice is
        frozen. Never train on it.
      </Callout>

      <Row gap={8} wrap>
        <Pill active={tab === "finding"} onClick={() => setTab("finding")}>
          The finding
        </Pill>
        <Pill active={tab === "literature"} onClick={() => setTab("literature")}>
          Related work
        </Pill>
        <Pill active={tab === "work"} onClick={() => setTab("work")}>
          v2 work
        </Pill>
        <Pill active={tab === "venues"} onClick={() => setTab("venues")}>
          Venues
        </Pill>
        <Pill active={tab === "plan"} onClick={() => setTab("plan")}>
          Sequence
        </Pill>
      </Row>

      {tab === "finding" && <Finding />}
      {tab === "literature" && <Literature />}
      {tab === "work" && <Work />}
      {tab === "venues" && <Venues />}
      {tab === "plan" && <Plan />}
    </Stack>
  );
}

function Finding() {
  return (
    <Stack gap={16}>
      <H2>What a journal can actually publish</H2>
      <Text>
        Learned forecasting of later reported log10(Pc) helps when the clock is
        stopped early, and almost not at all once later tracking has arrived.
        Typical events often do not move (persistence median AE = 0). Mean
        error is a rare-jump statistic, especially collapses to the dataset
        floor of −30.
      </Text>

      <Grid columns="1.2fr 0.8fr" gap={16}>
        <Stack gap={8}>
          <H3>MAE by forecast horizon (log10 Pc)</H3>
          <LineChart
            categories={HORIZON_MAE.categories}
            series={HORIZON_MAE.series}
            height={220}
            beginAtZero
            valueSuffix=" log"
          />
          <Text tone="tertiary" size="small">
            Source: ml/artifacts/metrics.json, single XGBoost vs persistence.
            T−48 test n = 1,659. T−72 / T−12 eligible sets overlap but are not
            identical.
          </Text>
        </Stack>
        <Stack gap={8}>
          <H3>Learned advantage vs persistence</H3>
          <BarChart
            categories={HORIZON_GAIN.categories}
            series={HORIZON_GAIN.series}
            height={220}
            beginAtZero
            valueSuffix=" log"
          />
          <Text tone="tertiary" size="small">
            T−12 gap is 0.06 log units. Without a confidence interval it is not
            a result.
          </Text>
        </Stack>
      </Grid>

      <Card>
        <CardHeader trailing={<Text size="small">process, not model</Text>}>
          Hypotheses
        </CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>
              H1. Extra information beyond copying the latest report is largest
              at long horizons and approaches zero near TCA.
            </Text>
            <Text>
              H2. Most of the mean-error win is floor jumps, not typical events
              getting more accurate.
            </Text>
            <Text>
              H3. Extra information does not automatically improve high-risk
              decisions. Local n = 9 positives. Official test ~150.
            </Text>
            <Text>
              H4. The snapshot gap max_risk_estimate − risk, and large
              covariance, predict later |Δrisk| and floor collapse (Alfano
              dilution / ellipse contraction). Space weather is the negative
              control.
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Stack gap={8}>
        <H3>Frozen local split — what the numbers actually say</H3>
        <Table
          headers={["System", "MAE", "Median AE", "Within 0.5", "High-risk MAE", "ESA loss / F2"]}
          columnAlign={["left", "right", "right", "right", "right", "right"]}
          rows={[
            ["Persistence", "5.080", "0.000", "63.1%", "0.89", "0.167 / 0.361"],
            ["Training-set median", "3.002", "0.000", "81.6%", "24.8", "∞ / 0"],
            ["Unguarded XGBoost", "2.808", "0.557", "48.8%", "13.69", "∞ / 0"],
            ["Exhibit ensemble + −6 guard", "3.059", "0.474", "50.5%", "3.69", "0.167 / 0.361"],
          ]}
          rowTone={["neutral", "warning", "info", "success"]}
        />
        <Text tone="tertiary" size="small">
          1,659 events. The exhibit policy ties persistence on ESA’s metric,
          loses to the constant median on MAE, and is less often close than
          persistence. Source: metrics.json.
        </Text>
      </Stack>
    </Stack>
  );
}

function Literature() {
  return (
    <Stack gap={16}>
      <H2>Where v2 sits in the real literature</H2>
      <Text tone="secondary">
        A reviewer who has read Uriot will reject “we used XGBoost on the ESA
        challenge.” The paper has to name the gap those papers left open.
      </Text>

      <Table
        headers={["Work", "What they did", "What they did not", "PRISM move"]}
        rows={[
          [
            "Uriot et al. 2021",
            "Kelvins design. LRP = 0.694. 12 of 96 teams beat it. Test over-represents high-risk (150 vs 66).",
            "No multi-horizon decay table on a naturalistic split. No floor-vs-median decomposition.",
            "Cite as the problem statement. Do not rerun the leaderboard as the contribution.",
          ],
          [
            "Zenodo 4463683 (2021)",
            "Released the hidden test labels plus raw CDMs.",
            "Most student projects still treat the test file as unlabeled.",
            "External eval, one shot, after freeze. This is how ESA-style loss becomes estimable.",
          ],
          [
            "Duncan / Wysack / Frisbee 2014 AMOS",
            "Forecast Pc by Monte Carlo with expected covariance reduction.",
            "Needs covariance time-history files. Not a public CDM benchmark.",
            "We are the statistical analogue on public messages, not a physics propagator.",
          ],
          [
            "Alfano; NASA CARA / Hejduk",
            "Dilution: large covariance can make Pc look safely small. Ellipse usually contracts toward TCA.",
            "Not an ML forecast of later reported log Pc.",
            "H4: use ESA’s max_risk_estimate − risk as a dilution probe.",
          ],
          [
            "Stauch / Olson et al. 2024–26 JAS",
            "Predict which conjunctions keep high covariance, for extra tracking, up to T−5 days.",
            "Different target: residual uncertainty, not final reported Pc.",
            "Cite and distinguish. Do not compete on their metric.",
          ],
        ]}
        rowTone={["warning", "success", "info", "info", "neutral"]}
      />

      <Callout tone="info" title="Community value is not the high-school journal">
        SSA researchers read AMOS, JAS, and Uriot. They will not find CJSJ.
        Ship a Zenodo protocol (code, metrics, figures, checksums — not the ESA
        zip). The journal is the credential. The archive is the contribution
        they can cite.
      </Callout>
    </Stack>
  );
}

function Work() {
  return (
    <Stack gap={16}>
      <H2>Ranked work (full checklist in v2.md)</H2>
      <Text tone="secondary">
        Medium-to-small phases live in the repo file v2.md. This table is the
        priority order. UI work is last.
      </Text>

      <Table
        headers={["#", "Phase", "Why it changes the paper", "Blocker?"]}
        columnAlign={["right", "left", "left", "left"]}
        rows={[
          ["0", "Branch + living paper", "paper/main.tex, references.bib, main.pdf. Close-out after every phase.", "Start"],
          ["1", "Honest metrics + CIs", "Stops the 39.8% MAE headline. Median AE and floor-excluded MAE.", "Yes"],
          ["2", "Residual target y − r", "Direct test that anything besides persistence exists.", "Yes"],
          ["3", "Two-part floor model", "Attacks −30 collapses that wreck mean error.", "Yes"],
          ["4", "Split-conformal intervals", "v1 90% bands cover 47.7%. Repair is a result.", "Yes"],
          ["5", "Official-test labels", "External split with ~150 positives. Comparable to LRP 0.694.", "Yes"],
          ["6", "Dilution / max-risk probe", "JEI-legal physics hypothesis. Space weather as negative control.", "Wanted"],
          ["7", "Five grouped redraws + LOO", "History +0.053 and T−12 +0.06 need a range, not a seed.", "Yes"],
          ["8", "Threshold sweep −8…−4", "Shows H3 is not an artifact of ESA’s scoring class.", "Wanted"],
          ["9", "Four paper figures", "Horizon, error anatomy, coverage, dilution.", "Yes"],
          ["10", "Replace the project", "Selected validation policy becomes API, site, README, PRD, paper.", "Yes"],
          ["11–12", "CJSJ cut from paper/ + Zenodo", "Export the living tex. No second manuscript tree.", "Deadline"],
        ]}
        rowTone={[
          "info",
          "danger",
          "danger",
          "danger",
          "warning",
          "danger",
          "info",
          "warning",
          "info",
          "success",
          "neutral",
          "success",
        ]}
      />

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Do not do on this branch</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>Queue redesign, what-if sliders, Vercel, LEDs.</Text>
              <Text>mission_id (already slightly worse).</Text>
              <Text>Transformer as the hero. GRU only as a later negative control.</Text>
              <Text>Retuning after seeing official-test scores.</Text>
              <Text>Claiming NASA/ISRO operational use.</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Reviewer attacks still live until the phase lands</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>You reran 2019 — fixed by H1–H4 + official test as secondary, not the contribution.</Text>
              <Text>n = 9 — fixed by official test + leave-one-positive-out.</Text>
              <Text>MAE shopping — fixed by median AE, floor split, ESA-loss tie as a finding.</Text>
              <Text>Uncalibrated 90% — fixed by conformal.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function Venues() {
  return (
    <Stack gap={16}>
      <H2>Where to send it (checked 19 Aug 2026)</H2>
      <Text tone="secondary">
        CJSJ allows preprints. Data may be reused for fairs. The submitted
        manuscript cannot be republished if CJSJ accepts it. Quarterfinalists
        are named 30 Oct — one day before PSHSJ closes.
      </Text>

      <Table
        headers={["Venue", "Cost", "Deadline", "Fit", "Move"]}
        rows={[
          [
            "CJSJ",
            "Free",
            "30 Sep 2026; quarterfinalists 30 Oct; finalists 11 Dec; issue 26 Mar 2027",
            "2–3 page original research. IEEE citations. ~selective. Gap-year ineligible. Finalists may expand pages.",
            "Primary. One paper per cycle.",
          ],
          [
            "PSHSJ",
            "Free",
            "1 Nov 2026, 23:59 EST",
            "3–5 pages. Allows later republication. Newer (2024).",
            "Only if CJSJ does not name you a quarterfinalist on 30 Oct. Do not dual-submit before that.",
          ],
          [
            "JEI",
            "$49",
            "Rolling",
            "DOI, adult submits. Hypothesis must be about the process, not model accuracy. No AI-written prose. Slow.",
            "Backup with H1–H4. Not the copilot story.",
          ],
          [
            "JHSS",
            "$65",
            "Rolling",
            "Quantitative STEAM, DOI. Accepts computational work. Paid.",
            "Tertiary. After CJSJ/PSHSJ/JEI decision.",
          ],
          [
            "Zenodo + GitHub",
            "Free",
            "Anytime",
            "Citable code. Dataset DOI 10.5281/zenodo.4463683. Do not upload the zip.",
            "Do this regardless.",
          ],
          [
            "IRIS → ISEF / AMOS 2027",
            "Free / travel",
            "Next cycle",
            "ISEF 2026 and AMOS 2026 student paper (14 Aug) already passed.",
            "Park until figures exist.",
          ],
        ]}
        rowTone={["success", "info", "warning", "neutral", "success", "neutral"]}
      />

      <Callout tone="warning" title="Skip">
        IJHSR charges about $250 on acceptance. Journal of Student Research is
        a weaker signal. Do not pay for a less selective venue while CJSJ is
        still open.
      </Callout>

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Working title</CardHeader>
          <CardBody>
            <Text>
              Predictive information in conjunction data-message histories
              decays as time to closest approach decreases
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Desk-reject titles</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>PRISM: an explainable AI copilot for T−48 risk</Text>
              <Text>We hypothesize that XGBoost can predict Pc</Text>
              <Text>Beating the ESA Collision Avoidance Challenge</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function Plan() {
  return (
    <Stack gap={16}>
      <H2>Six weeks, mapped onto v2.md phases</H2>
      <Text tone="secondary">
        Today is 19 August 2026. If official-test ingest slips, submit CJSJ on
        horizon + floor from the local split. CJSJ finalists may add omitted
        sections later.
      </Text>

      <Table
        headers={["When", "v2.md phases", "Done looks like"]}
        rows={[
          [
            "19–21 Aug",
            "0, 1",
            "Branch exists. CIs, floor-excluded MAE, residual MAE in metrics.json.",
          ],
          [
            "22–25 Aug",
            "2, 3",
            "Residual model and two-part floor model compared on the frozen split.",
          ],
          [
            "26–28 Aug",
            "4, 6",
            "Conformal coverage near 90%. Dilution-gap table with Spearman / logistic.",
          ],
          [
            "29 Aug–2 Sep",
            "5, 7, 8",
            "Official-test block written once. Five-split range. Threshold sweep.",
          ],
          [
            "3–6 Sep",
            "9, 10",
            "Four figures. README and lab match the numbers.",
          ],
          [
            "7–30 Sep",
            "11, 12",
            "CJSJ template + form. Tag v2.0.0-paper. Zenodo code archive.",
          ],
        ]}
        rowTone={["danger", "danger", "warning", "warning", "info", "success"]}
      />

      <Callout tone="success" title="PR ready to merge">
        Leakage tests pass. metrics.json has CIs, floor metrics, residual rows,
        conformal coverage, and either officialTest or a written reason it
        could not be built. API, website, and docs all serve the one selected
        policy.
      </Callout>
    </Stack>
  );
}
