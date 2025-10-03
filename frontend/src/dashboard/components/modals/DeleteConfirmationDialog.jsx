import * as React from "react";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

import { useDashboard } from "../../context/DashboardContext";

const DeleteConfirmationDialog = () => {
  const { deleteDialog, actions } = useDashboard();

  const { open, loading, seriesId, calculationType, patientName } = deleteDialog;
  const typeText = calculationType
    ? calculationType === "global_noise"
      ? "Global Noise"
      : "Full Analysis"
    : "All Results";

  return (
    <Dialog open={open} onClose={loading ? undefined : actions.closeDeleteDialog} maxWidth="xs" fullWidth>
      <DialogTitle>Confirm Delete</DialogTitle>
      <DialogContent>
        <Stack spacing={1} sx={{ my: 1 }}>
          <Typography variant="body2">Are you sure you want to delete this CHO analysis result?</Typography>
          {seriesId && (
            <Typography variant="body2" color="text.secondary">
              Series: {seriesId}
            </Typography>
          )}
          <Typography variant="body2" color="text.secondary">
            Type: {typeText}
          </Typography>
          {patientName && (
            <Typography variant="body2" color="text.secondary">
              Patient: {patientName}
            </Typography>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={actions.closeDeleteDialog} disabled={loading}>
          Cancel
        </Button>
        <Button color="error" onClick={actions.confirmDelete} disabled={loading}>
          {loading ? "Deleting..." : "Delete"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DeleteConfirmationDialog;
