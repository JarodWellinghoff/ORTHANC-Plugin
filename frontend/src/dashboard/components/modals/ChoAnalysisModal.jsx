import * as React from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import Grid from "@mui/material/Grid";
import InputLabel from "@mui/material/InputLabel";
import LinearProgress from "@mui/material/LinearProgress";
import MenuItem from "@mui/material/MenuItem";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

import { useDashboard } from "../../context/DashboardContext";
import ChoPlots from "../plots/ChoPlots";
import MetadataSections from "../metadata/MetadataSections";

const stageOrder = [
  "initialization",
  "loading",
  "preprocessing",
  "analysis",
  "finalizing",
];

const ChoAnalysisModal = () => {
  const { choModal, actions } = useDashboard();
  const {
    open,
    params,
    stage,
    progress,
    results,
    pollError,
    seriesSummary,
    saving,
  } = choModal;

  const handleFieldChange = (name) => (event) => {
    actions.updateChoParam(name, event.target.value);
  };

  const currentStageIndex = stageOrder.indexOf(
    progress?.stage ?? "initialization"
  );

  return (
    <Dialog open={open} onClose={actions.closeChoModal} maxWidth='xl' fullWidth>
      <DialogTitle>CHO Analysis Configuration</DialogTitle>
      <DialogContent dividers sx={{ bgcolor: "background.default" }}>
        {seriesSummary && (
          <Box
            sx={{
              mb: 3,
              borderRadius: 2,
              border: "1px solid",
              borderColor: "divider",
              p: 2,
              bgcolor: "background.paper",
            }}>
            <Typography variant='subtitle1' sx={{ fontWeight: 600 }}>
              Selected Series
            </Typography>
            <Typography variant='body2' color='text.secondary'>
              {seriesSummary.patient_name ?? "N/A"} -{" "}
              {seriesSummary.protocol_name ?? "N/A"} -{" "}
              {seriesSummary.institution_name ?? "N/A"}
            </Typography>
          </Box>
        )}

        {stage === "config" && (
          <Stack spacing={3}>
            <Stack spacing={1}>
              <Typography variant='subtitle1' sx={{ fontWeight: 600 }}>
                Analysis Type
              </Typography>
              <RadioGroup
                row
                value={params.testType}
                onChange={(event) =>
                  actions.updateChoParam("testType", event.target.value)
                }>
                <FormControlLabel
                  value='global'
                  control={<Radio />}
                  label='Global Noise Analysis'
                />
                <FormControlLabel
                  value='full'
                  control={<Radio />}
                  label='Full CHO Analysis'
                />
              </RadioGroup>
            </Stack>

            <Box
              sx={{
                borderRadius: 2,
                border: "1px solid",
                borderColor: "divider",
                p: 2,
                bgcolor: "background.paper",
              }}>
              <Typography variant='subtitle1' sx={{ fontWeight: 600, mb: 2 }}>
                Common Parameters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    label='Number of Resamples'
                    type='number'
                    value={params.resamples}
                    onChange={handleFieldChange("resamples")}
                    fullWidth
                    inputProps={{ min: 100, max: 2000, step: 100 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    label='Internal Noise'
                    type='number'
                    value={params.internalNoise}
                    onChange={handleFieldChange("internalNoise")}
                    fullWidth
                    inputProps={{ min: 0, max: 10, step: 0.25 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel id='cho-resampling-method-label'>
                      Resampling Method
                    </InputLabel>
                    <Select
                      labelId='cho-resampling-method-label'
                      value={params.resamplingMethod}
                      label='Resampling Method'
                      onChange={handleFieldChange("resamplingMethod")}>
                      <MenuItem value='Bootstrap'>Bootstrap</MenuItem>
                      <MenuItem value='Shuffle'>Shuffle</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>

            {params.testType === "global" && (
              <Box
                sx={{
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  p: 2,
                  bgcolor: "background.paper",
                }}>
                <Typography variant='subtitle1' sx={{ fontWeight: 600, mb: 2 }}>
                  Global Noise Parameters
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={4}>
                    <TextField
                      label='ROI Size (cm)'
                      type='number'
                      value={params.roiSize}
                      onChange={handleFieldChange("roiSize")}
                      fullWidth
                      inputProps={{ min: 3, max: 20, step: 0.5 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6} md={4}>
                    <TextField
                      label='Lower Threshold (HU)'
                      type='number'
                      value={params.thresholdLow}
                      onChange={handleFieldChange("thresholdLow")}
                      fullWidth
                      inputProps={{ min: -100, max: 200, step: 5 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6} md={4}>
                    <TextField
                      label='Upper Threshold (HU)'
                      type='number'
                      value={params.thresholdHigh}
                      onChange={handleFieldChange("thresholdHigh")}
                      fullWidth
                      inputProps={{ min: 100, max: 300, step: 5 }}
                    />
                  </Grid>
                </Grid>
              </Box>
            )}

            {params.testType === "full" && (
              <Box
                sx={{
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  p: 2,
                  bgcolor: "background.paper",
                }}>
                <Typography variant='subtitle1' sx={{ fontWeight: 600, mb: 2 }}>
                  Full Analysis Parameters
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={3}>
                    <TextField
                      label='Window Length (cm)'
                      type='number'
                      value={params.windowLength}
                      onChange={handleFieldChange("windowLength")}
                      fullWidth
                      inputProps={{ min: 10, max: 25, step: 2.5 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <TextField
                      label='Step Size (cm)'
                      type='number'
                      value={params.stepSize}
                      onChange={handleFieldChange("stepSize")}
                      fullWidth
                      inputProps={{ min: 2.5, max: 10, step: 2.5 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <FormControl fullWidth>
                      <InputLabel id='cho-channel-type-label'>
                        Channel Type
                      </InputLabel>
                      <Select
                        labelId='cho-channel-type-label'
                        value={params.channelType}
                        label='Channel Type'
                        onChange={handleFieldChange("channelType")}>
                        <MenuItem value='Gabor'>Gabor</MenuItem>
                        <MenuItem value='Laguerre-Gauss'>
                          Laguerre-Gauss
                        </MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <FormControl fullWidth>
                      <InputLabel id='cho-lesion-set-label'>
                        Lesion Set
                      </InputLabel>
                      <Select
                        labelId='cho-lesion-set-label'
                        value={params.lesionSet}
                        label='Lesion Set'
                        onChange={handleFieldChange("lesionSet")}>
                        <MenuItem value='standard'>
                          Standard (-30, -30, -10, -30, -50 HU)
                        </MenuItem>
                        <MenuItem value='low-contrast'>
                          Low Contrast (-10, -15, -20 HU)
                        </MenuItem>
                        <MenuItem value='high-contrast'>
                          High Contrast (-50, -75, -100 HU)
                        </MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Stack>
        )}

        {stage === "progress" && (
          <Stack spacing={2}>
            <Typography variant='subtitle1' sx={{ fontWeight: 600 }}>
              Analysis Progress
            </Typography>
            <Stack spacing={1}>
              <LinearProgress
                variant='determinate'
                value={progress?.value ?? 0}
              />
              <Typography variant='body2' color='text.secondary'>
                {progress?.value ?? 0}% - {progress?.message ?? "Processing..."}
              </Typography>
            </Stack>
            <Stack direction='row' spacing={1}>
              {stageOrder.map((stageName, index) => {
                const isActive = stageName === progress?.stage;
                const isCompleted = index < currentStageIndex;
                return (
                  <Box
                    key={stageName}
                    sx={{
                      px: 1.5,
                      py: 0.5,
                      borderRadius: 1,
                      bgcolor: isActive
                        ? "primary.light"
                        : isCompleted
                        ? "success.light"
                        : "action.hover",
                      color: isActive
                        ? "primary.contrastText"
                        : "text.secondary",
                      fontSize: 12,
                      fontWeight: isActive ? 600 : 500,
                    }}>
                    {stageName}
                  </Box>
                );
              })}
            </Stack>
            {pollError && <Alert severity='error'>{pollError}</Alert>}
          </Stack>
        )}

        {stage === "results" && (
          <Stack spacing={3}>
            {pollError && <Alert severity='error'>{pollError}</Alert>}
            {results ? (
              <>
                <ChoPlots data={results} mode='results' />
                {/* <Stack direction={{ xs: "column", lg: "row" }} spacing={3}> */}
                <Box sx={{ flex: 1 }}>
                  <Typography
                    variant='subtitle1'
                    sx={{ fontWeight: 600, mb: 1 }}>
                    Analysis Summary
                  </Typography>
                  <MetadataSections data={results} />
                </Box>
                {/* <Box
                    sx={{
                      flex: 1,
                      borderRadius: 2,
                      border: "1px solid",
                      borderColor: "divider",
                      bgcolor: "background.paper",
                      p: 2,
                      maxHeight: 300,
                      overflow: "auto",
                      fontFamily: "Consolas,Menlo,monospace",
                      fontSize: 12,
                    }}>
                    <Typography
                      variant='subtitle1'
                      sx={{ fontWeight: 600, mb: 1 }}>
                      Raw Data (JSON)
                    </Typography>
                    <pre style={{ margin: 0 }}>
                      {JSON.stringify(results, null, 2)}
                    </pre>
                  </Box> */}
                {/* </Stack> */}
              </>
            ) : (
              <Typography variant='body2' color='text.secondary'>
                No cached results available.
              </Typography>
            )}
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={actions.closeChoModal}>Close</Button>
        <Button
          variant='contained'
          onClick={actions.startChoAnalysis}
          disabled={stage === "progress"}>
          {stage === "progress" ? "Analysis Running" : "Start Analysis"}
        </Button>
        <Button
          variant='contained'
          color='success'
          onClick={actions.saveChoResults}
          disabled={!results || stage !== "results" || saving}>
          {saving ? "Saving..." : "Save Results"}
        </Button>
        <Button
          variant='outlined'
          color='error'
          onClick={actions.discardChoResults}
          disabled={!results || stage !== "results"}>
          Discard Results
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ChoAnalysisModal;
