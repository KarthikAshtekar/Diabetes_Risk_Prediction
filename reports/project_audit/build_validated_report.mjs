import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const inputPath = resolve(process.argv[2] ?? "reports/project_audit/artifact.json");
const outputPath = resolve(
  process.argv[3] ?? "reports/project_audit/diabetes_project_validation.html"
);
const candidatePath = `${outputPath}.candidate.html`;
const failureScreenshot = resolve(
  dirname(outputPath),
  "portable_report_verification_failure.png"
);

const pluginCache = join(
  homedir(),
  ".codex",
  "plugins",
  "cache",
  "openai-curated-remote",
  "data-analytics"
);
const versions = readdirSync(pluginCache, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => join(pluginCache, entry.name))
  .filter((path) =>
    existsSync(join(path, "skills", "build-report", "scripts", "build_portable_artifact.mjs"))
  )
  .sort((left, right) => statSync(right).mtimeMs - statSync(left).mtimeMs);

if (!versions.length) {
  throw new Error(`No installed Data Analytics report builder found under ${pluginCache}`);
}

const builderScripts = join(versions[0], "skills", "build-report", "scripts");
const { buildPortableArtifact } = await import(
  pathToFileURL(join(builderScripts, "build_portable_artifact.mjs")).href
);
const { extractPortableChartSvgs } = await import(
  pathToFileURL(join(builderScripts, "extract_portable_chart_svgs.mjs")).href
);
const { verifyPortableArtifact } = await import(
  pathToFileURL(join(builderScripts, "verify_portable_artifact.mjs")).href
);

const overflowFix = `
<style id="project-audit-portable-overflow-fix">
html, body { max-width: 100%; overflow-x: clip; }
.analytics-top-bar,
.portable-page-header {
  width: 100%;
  margin-right: 0;
  margin-left: 0;
}
</style>`;

function applyOverflowFix(html) {
  if (!html.includes("</head>")) {
    throw new Error("Portable report has no closing head tag for the scoped CSS correction.");
  }
  return html.replace("</head>", `${overflowFix}\n</head>`);
}

const artifact = JSON.parse(readFileSync(inputPath, "utf8"));
mkdirSync(dirname(outputPath), { recursive: true });

try {
  let html = applyOverflowFix(buildPortableArtifact(artifact));
  writeFileSync(candidatePath, html, "utf8");

  const staticCharts = await extractPortableChartSvgs({
    actionTimeoutMs: 5_000,
    htmlPath: candidatePath,
    readyTimeoutMs: 15_000
  });
  html = applyOverflowFix(buildPortableArtifact(artifact, { staticCharts }));
  writeFileSync(candidatePath, html, "utf8");

  const verification = await verifyPortableArtifact({
    actionTimeoutMs: 5_000,
    artifactPath: inputPath,
    htmlPath: candidatePath,
    readyTimeoutMs: 15_000,
    screenshotPath: failureScreenshot,
    timeoutMs: 30_000
  });

  renameSync(candidatePath, outputPath);
  console.log(
    JSON.stringify(
      {
        ok: true,
        html: outputPath,
        pluginRoot: versions[0],
        sourceDialog: verification.sourceDialog,
        sourceInteraction: verification.sourceInteraction,
        viewports: verification.viewports,
        counts: verification.counts,
        timings: verification.timings
      },
      null,
      2
    )
  );
} finally {
  rmSync(candidatePath, { force: true });
}
