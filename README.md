# PathSnap

PathSnap is a distributed GPS map-matching and route-reconstruction project. This workspace currently implements **Stage 1 only**: raw-data readers and validation, local metric road geometry, candidate generation, monotonic alignment of GPS observations to the supplied ordered reference route, and training-trip diagnostics.

No ML models, API, UI, or evaluator-trip substitution is included yet. The supplied archive contains `train` and `student_test`; the separate reserved evaluator partition described by the official instructions is absent.

## Data layout

Keep the original ZIPs unchanged. The existing extracted source files are under `data/source/`:

```text
data/source/
  materials/                    # official PDFs and extracted text
  dataset/Student folder/
    train/<trip-id>/<trip-id>.*
    student_test/<trip-id>/<trip-id>.*
```

The reader treats `.track`, `.nodes`, `.arcs`, and `.route` as headerless whitespace-separated files. `.route` values index rows of that trip's `.arcs` file; arc IDs are trip-local.

## Environment setup (Windows PowerShell)

From the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Stage 1 commands

Run the unit tests:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Generate the training-only diagnostics and three interactive maps (two low-coverage cases and one well-covered long trip):

```powershell
$env:PYTHONPATH = "src"
python -m pathsnap.stage1
```

Optional parameters:

```powershell
python -m pathsnap.stage1 --data-root "data/source/dataset/Student folder" `
  --output-root reports/stage1 --candidate-radius-m 80 `
  --timestamp-gap-threshold-s 5 --map-count 3
```

The runner reads training files only; it checks partition folder counts and ID overlap without opening student-test trip files, and requires 80 training folders plus 10 student-test folders. It writes `partition_manifest.json`, `trip_diagnostics.csv`, `trip_diagnostics.json`, one per-observation alignment CSV per training trip, and the requested Folium HTML maps under `reports/stage1/`. The raw source files are read-only inputs. `maps/index.json` lists the selected examples.

## Stage 1 method and assumptions

- Each trip is projected independently from WGS84 longitude/latitude to a local azimuthal-equidistant metric CRS centered on that trip's GPS observations. Distances and candidate radii are in metres; Folium output uses the original coordinates.
- Candidate roads are all trip-local arcs whose straight edge geometry falls within the configured 80 m radius of a GPS point. No global arc vocabulary is created.
- Alignment uses a Viterbi-style dynamic program over route-occurrence candidates plus an explicit unmatched state. It prioritizes every nearby candidate arc that repeats in the reference route, then fills up to 20 candidate-arc slots with the nearest remaining arcs; repeated arcs are retained even if they take the total above 20. Transitions preserve route-position order (and nondecreasing chainage on one occurrence); a final path check selects and flags a forward candidate when the decoded path would move backward after unmatched observations. For timestamp steps over five seconds or nonpositive steps, displacement is not used as evidence; route order still applies.
- Local alignment ambiguity is flagged when the two closest reference-route candidate distances differ by at most 5 m. Candidate misses and observations assigned to the unmatched state remain explicit in CSV outputs.
- Supplied route discontinuities are reported and never repaired. Direct decoder transitions across a discontinuity are disallowed; when decoding resumes after an unmatched observation, the resumed label is marked if it spans a supplied break. Nonpositive timestamp intervals are recorded in the trip diagnostics and alignment CSV; they are not silently reordered.
- These are transparent Stage 1 parameters, not learned model parameters. They are not fitted on student-test reference routes. Diagnostic runs do not open `student_test/`.

Alignment remains an approximation of observation-to-route correspondence. The route sequence is not per-GPS ground truth, and outlier GPS points, repeated route arcs, very large gaps, candidate misses, and disconnected supplied route transitions can leave observations unmatched or ambiguous. Review those diagnostics before using the alignments to build a training set.

## Academic integrity

Codex assisted with the Stage 1 implementation and test drafting. The project owner should review and understand the code and disclose AI assistance as required by the course policy. All reported diagnostics are generated from this workspace's supplied training files; no model experiments or performance claims are included.

## Generated outputs

Generated files are kept under `reports/stage1/` and are reproducible by rerunning the command. Maps contain GPS observations, nearby candidate road edges, ordered reference-route arcs, and the inferred monotonic alignment. The maps are generated from the supplied training data rather than hand-drawn.
