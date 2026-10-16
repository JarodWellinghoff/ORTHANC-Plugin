# CTPro statistics and rerun review

## Statistics report

The existing `/plotting` route remains the statistics workspace; `/results` is
still the individual-exam catalog. Stored DICOM protocol metadata is not changed.
The report groups by the first underscore-separated token, except an exact `WIP`
first token retains the full string. The protocol filter uses these same groups.

Box plots are on the left, initially selecting the ten groups with the most
series. Ties and display order are alphabetical. Each histogram on the right
selects one protocol, initially the most frequent. Per-plot names were removed.

Number of patients counts distinct nonblank Patient IDs, including their issuer
when available, rather than series. Missing IDs are excluded and disclosed. This
metric uses count bars in either column: a single cohort count has no per-series
box-plot or histogram distribution. Other metrics retain their usual chart type.

Export CSV replaces the old bottom table and preserves its scope: every protocol
in the current query for each configured metric, independent of chart selections.
The export includes series/patient counts, missing-ID counts, n, mean, sample SD,
median, minimum and maximum. Patient-count rows have an aggregate count instead
of fabricated per-series statistics. CSV uses UTF-8, CRLF, quoted fields and
formula-leading text protection. Processing time and image count are no longer
report metrics; dose labels are DLP and DLP_SSDE.

## Site layout

Home uses the requested CTPro title and no Getting Started section. Batch Analysis
replaces Bulk Runs / Calculation in Local Server, with Run Selected above its
table. Dashboard DataGrids have a stable, scrollable 560px viewport, without
rows-per-page controls. Server pagination is retained for large catalogs. Plain
table containers have a bounded scroll area and sticky headers.

## Rerun review

The calculation wrapper checks for existing measurements before starting; a
metadata-only placeholder is not an analyzed result. First analyses retain their
existing save policy. Reruns force saveResults=false and
deleteAfterCompletion=false, keeping accepted measurements and their coronal
image unchanged during calculation and review.

A global dialog compares saved/new metrics and curves and asks whether to
Overwrite saved results or Discard new results. Review later keeps the candidate
available through a banner, including after a browser refresh. A second run of
the same series is blocked until the first run/review is resolved. Discard only
removes the candidate. Overwrite compares the saved record with its baseline
under a PostgreSQL row lock and then calls the existing save/upsert pathway.
Changed or deleted saved records cause a conflict instead of a stale overwrite;
a failed database save retains the candidate for retry. No schema migration is
required. Image persistence after acceptance uses the existing MinIO behavior.

GET /cho-result-reviews lists pending reviews. GET /cho-result-reviews/<id>
retrieves a comparison. POST to that URL accepts
{"decision":"overwrite"} or {"decision":"discard"}. These APIs use the existing
Orthanc authentication/CORS deployment boundary and disable response caching.

Drafts are process-local, like progress tracking. Browser refreshes recover them;
restarting Orthanc discards unaccepted drafts, never the previously saved result.
The five-minute progress cleanup does not expire a pending review. Complete
reviews before service restarts. No automatic overwrite is scheduled.

## Validation and deployment

Run `cd frontend && npm test` and, from the repository root,
`python -m unittest discover -s tests -p 'test_analysis_review*.py'`.
The tests use synthetic records and storage/Orthanc doubles; they are not a
substitute for deployment-level validation.

Before merging/deploying, run `npm ci && npm run build` and exercise the UI with
Orthanc, PostgreSQL and MinIO. Check protocol counts/WIP names, single histograms,
CSV download, long-table scrolling, and rerun preview/discard/overwrite, including
a browser refresh during review and a full-analysis to global-noise rerun.
