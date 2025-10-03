import * as React from "react";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";

import { useDashboard } from "../../context/DashboardContext";
import ChoPlots from "../plots/ChoPlots";
import MetadataSections from "../metadata/MetadataSections";

const SeriesDetailsModal = () => {
  const { detailsModal, actions } = useDashboard();
  const { open, loading, data, seriesId } = detailsModal;

  return (
    <Dialog open={open} onClose={actions.closeDetails} maxWidth='xl' fullWidth>
      <DialogTitle>Detailed CHO Results</DialogTitle>
      <DialogContent dividers sx={{ bgcolor: "background.default" }}>
        {loading ? (
          <Stack alignItems='center' sx={{ py: 6 }} spacing={2}>
            <CircularProgress size={36} />
            <Typography variant='body2' color='text.secondary'>
              Loading series details...
            </Typography>
          </Stack>
        ) : data ? (
          <Stack spacing={3}>
            <ChoPlots data={data} mode='details' />
            {/* <Stack direction={{ xs: "column", lg: "row" }} spacing={3}> */}
            <Box sx={{ flex: 1 }}>
              <Typography variant='subtitle1' sx={{ fontWeight: 600, mb: 1 }}>
                Analysis Summary
              </Typography>
              <MetadataSections data={data} />
            </Box>
            {/* <Box
                sx={{
                  flex: 1,
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  bgcolor: "background.paper",
                  p: 2,
                  maxHeight: 400,
                  overflow: "auto",
                  fontFamily: "Consolas,Menlo,monospace",
                  fontSize: 12,
                }}
              >
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Raw Data (JSON)
                </Typography>
                <pre style={{ margin: 0 }}>
                  {JSON.stringify(data, null, 2)}
                </pre>
              </Box>
            </Stack> */}
          </Stack>
        ) : (
          <Typography variant='body2' color='text.secondary'>
            No details available for this series.
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button
          onClick={() => actions.exportSeries(seriesId)}
          disabled={!seriesId || loading}>
          Export
        </Button>
        <Button onClick={actions.closeDetails}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default SeriesDetailsModal;
