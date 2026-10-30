import * as React from "react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { RESULT_METRICS } from "./metadata/metadataSections.utils";
import FormattedMetricText from "./metadata/FormattedMetricText";
import { buildComparisonRows } from "../utils/resultComparison";

const formatDelta = (value, decimals = 2) => {
  if (!Number.isFinite(value)) return "N/A";
  const normalized = Math.abs(value) < 1e-12 ? 0 : value;
  const fixed = normalized.toFixed(decimals);
  return normalized > 0 ? `+${fixed}` : fixed;
};

const getMetricSource = (data) => data?.results ?? data ?? null;

const DATA_DENSITY_FIELDS = [
  { key: "location", label: "CTDIvol samples" },
  { key: "location_sparse", label: "Noise samples" },
  { key: "spatial_frequency", label: "NPS samples" },
];

const getArrayLength = (data, key) => {
  const source = getMetricSource(data);
  const value = source?.[key];
  return Array.isArray(value) ? value.length : null;
};

const ResultsComparison = ({ baseline, current }) => {
  const baselineMetrics = React.useMemo(
    () => getMetricSource(baseline),
    [baseline]
  );
  const currentMetrics = React.useMemo(
    () => getMetricSource(current),
    [current]
  );

  const metricRows = React.useMemo(
    () => buildComparisonRows(baselineMetrics, currentMetrics, RESULT_METRICS),
    [baselineMetrics, currentMetrics],
  );

  const densityRows = React.useMemo(() => {
    return DATA_DENSITY_FIELDS.map(({ key, label }) => {
      const baselineCount = getArrayLength(baseline, key);
      const currentCount = getArrayLength(current, key);
      if (baselineCount === null && currentCount === null) {
        return null;
      }
      const delta =
        baselineCount !== null && currentCount !== null
          ? formatDelta(currentCount - baselineCount, 0)
          : "N/A";
      return {
        label,
        baselineCount:
          baselineCount !== null ? baselineCount.toString() : "N/A",
        currentCount: currentCount !== null ? currentCount.toString() : "N/A",
        delta,
      };
    }).filter(Boolean);
  }, [baseline, current]);

  if (metricRows.length === 0 && densityRows.length === 0) {
    return null;
  }

  return (
    <Stack
      spacing={2}
      sx={
        {
          // width: "50%",
        }
      }>
      {/* <Box>
          <Typography variant='subtitle2' sx={{ fontWeight: 600 }}>
            Stored vs New Results
          </Typography>
          <Typography variant='body2' color='text.secondary'>
            Stored values reference the last saved analysis. New values reflect
            this run. Differences account for length mismatches so mixed global
            and full tests stay comparable.
          </Typography>
        </Box> */}

      {metricRows.length > 0 && (
        <Box>
          <Typography variant='subtitle1' sx={{ fontWeight: 600, mb: 1 }}>
            Results
          </Typography>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "repeat(4, minmax(80px, 1fr))",
                sm: "minmax(160px, 1fr) repeat(3, minmax(120px, 1fr))",
              },
              columnGap: { xs: 1, sm: 2 },
              rowGap: 1,
              alignItems: "center",
            }}>
            <Typography variant='body2' sx={{ fontWeight: 600 }}>
              Metric
            </Typography>
            <Typography variant='body2' sx={{ fontWeight: 600 }}>
              Stored
            </Typography>
            <Typography variant='body2' sx={{ fontWeight: 600 }}>
              New
            </Typography>
            <Typography variant='body2' sx={{ fontWeight: 600 }}>
              Δ
            </Typography>
            {metricRows.map(({ key, label, unit, baseline, current, delta }) => (
              <React.Fragment key={key}>
                <Typography variant='body2' color='text.secondary'>
                  <FormattedMetricText value={label} />
                </Typography>

                <Stack direction='row' spacing={1}>
                  <Typography variant='body2'>{baseline}</Typography>
                  {unit ? (
                    <Typography variant='body2' color='text.secondary'>
                      <FormattedMetricText value={unit} />
                    </Typography>
                  ) : null}
                </Stack>

                <Stack direction='row' spacing={1}>
                  <Typography variant='body2'>{current}</Typography>
                  {unit ? (
                    <Typography variant='body2' color='text.secondary'>
                      <FormattedMetricText value={unit} />
                    </Typography>
                  ) : null}
                </Stack>
                <Stack direction='row' spacing={1}>
                  <Typography
                    variant='body2'
                    color={
                      delta === "N/A"
                        ? "text.secondary"
                        : delta.startsWith("+")
                        ? "success.main"
                        : delta.startsWith("-")
                        ? "error.main"
                        : "text.primary"
                    }>
                    {delta}
                  </Typography>
                  {unit ? (
                    <Typography variant='body2' color='text.secondary'>
                      <FormattedMetricText value={unit} />
                    </Typography>
                  ) : null}
                </Stack>
              </React.Fragment>
            ))}
          </Box>
        </Box>
      )}

      {/* {densityRows.length > 0 && (
        <>
          <Divider flexItem />
          <Box>
            <Typography
              variant='caption'
              color='text.secondary'
              sx={{ display: "block", mb: 1 }}>
              Sample counts
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: {
                  xs: "repeat(4, minmax(80px, 1fr))",
                  sm: "minmax(160px, 1fr) repeat(3, minmax(120px, 1fr))",
                },
                columnGap: { xs: 1, sm: 2 },
                rowGap: 1,
                alignItems: "center",
              }}>
              <Typography variant='body2' sx={{ fontWeight: 600 }}>
                Series
              </Typography>
              <Typography variant='body2' sx={{ fontWeight: 600 }}>
                Stored
              </Typography>
              <Typography variant='body2' sx={{ fontWeight: 600 }}>
                New
              </Typography>
              <Typography variant='body2' sx={{ fontWeight: 600 }}>
                Δ
              </Typography>
              {densityRows.map(
                ({ label, baselineCount, currentCount, delta }) => (
                  <React.Fragment key={label}>
                    <Typography variant='body2' color='text.secondary'>
                      {label}
                    </Typography>
                    <Typography variant='body2'>{baselineCount}</Typography>
                    <Typography variant='body2'>{currentCount}</Typography>
                    <Typography
                      variant='body2'
                      color={
                        delta === "N/A"
                          ? "text.secondary"
                          : delta.startsWith("+")
                          ? "success.main"
                          : delta.startsWith("-")
                          ? "error.main"
                          : "text.primary"
                      }>
                      {delta}
                    </Typography>
                  </React.Fragment>
                )
              )}
            </Box>
          </Box>
        </>
      )} */}
    </Stack>
  );
};

export default ResultsComparison;
