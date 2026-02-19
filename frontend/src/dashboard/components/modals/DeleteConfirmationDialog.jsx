import * as React from "react";
import Alert from "@mui/material/Alert";
import Button from "@mui/material/Button";
import Checkbox from "@mui/material/Checkbox";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
import FormControlLabel from "@mui/material/FormControlLabel";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import StorageRoundedIcon from "@mui/icons-material/StorageRounded";
import MedicalServicesRoundedIcon from "@mui/icons-material/MedicalServicesRounded";

import { useDashboard } from "../../context/DashboardContext";

const DeleteConfirmationDialog = () => {
  const { deleteDialog, actions } = useDashboard();

  const {
    open,
    loading,
    seriesId,
    patientName,
    hasResults,
    hasDicom,
    deleteResults,
    deleteDicom,
  } = deleteDialog;

  const neitherSelected = !deleteResults && !deleteDicom;

  // Cross-prompt: user selected one but the other also exists and isn't selected
  const suggestDicom = deleteResults && hasDicom && !deleteDicom;
  const suggestResults = deleteDicom && hasResults && !deleteResults;

  return (
    <Dialog
      open={open}
      onClose={loading ? undefined : actions.closeDeleteDialog}
      maxWidth='sm'
      fullWidth>
      <DialogTitle>Confirm Delete</DialogTitle>

      <DialogContent>
        <Stack spacing={2} sx={{ mt: 0.5 }}>
          {/* Series / patient info */}
          <Stack spacing={0.5}>
            {patientName && (
              <Typography variant='body2' color='text.secondary'>
                Patient: <strong>{patientName}</strong>
              </Typography>
            )}
            {seriesId && (
              <Typography
                variant='body2'
                color='text.secondary'
                sx={{ wordBreak: "break-all" }}>
                Series UID: {seriesId}
              </Typography>
            )}
          </Stack>

          <Divider />

          {/* What to delete */}
          <Typography variant='subtitle2'>
            What would you like to delete?
          </Typography>

          {/* Delete results checkbox */}
          <Stack
            direction='row'
            alignItems='flex-start'
            spacing={1}
            sx={{
              p: 1.5,
              borderRadius: 1,
              border: "1px solid",
              borderColor: deleteResults ? "error.main" : "divider",
              bgcolor: deleteResults ? "error.50" : "transparent",
              opacity: hasResults ? 1 : 0.5,
            }}>
            <StorageRoundedIcon
              fontSize='small'
              color={deleteResults ? "error" : "disabled"}
              sx={{ mt: 0.3 }}
            />
            <Stack spacing={0.25} flexGrow={1}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={deleteResults}
                    disabled={loading || !hasResults}
                    onChange={() => actions.toggleDeleteOption("deleteResults")}
                    color='error'
                    size='small'
                  />
                }
                label={
                  <Typography variant='body2' fontWeight={500}>
                    Delete analysis results from database
                  </Typography>
                }
                sx={{ m: 0 }}
              />
              <Typography
                variant='caption'
                color='text.secondary'
                sx={{ pl: 3.5 }}>
                {hasResults
                  ? "Removes the CHO analysis results stored in PostgreSQL. The DICOM images are unaffected."
                  : "No analysis results found in the database for this series."}
              </Typography>
            </Stack>
          </Stack>

          {/* Delete DICOM checkbox */}
          <Stack
            direction='row'
            alignItems='flex-start'
            spacing={1}
            sx={{
              p: 1.5,
              borderRadius: 1,
              border: "1px solid",
              borderColor: deleteDicom ? "error.main" : "divider",
              bgcolor: deleteDicom ? "error.50" : "transparent",
              opacity: hasDicom ? 1 : 0.5,
            }}>
            <MedicalServicesRoundedIcon
              fontSize='small'
              color={deleteDicom ? "error" : "disabled"}
              sx={{ mt: 0.3 }}
            />
            <Stack spacing={0.25} flexGrow={1}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={deleteDicom}
                    disabled={loading || !hasDicom}
                    onChange={() => actions.toggleDeleteOption("deleteDicom")}
                    color='error'
                    size='small'
                  />
                }
                label={
                  <Typography variant='body2' fontWeight={500}>
                    Delete DICOM images from Orthanc server
                  </Typography>
                }
                sx={{ m: 0 }}
              />
              <Typography
                variant='caption'
                color='text.secondary'
                sx={{ pl: 3.5 }}>
                {hasDicom
                  ? "Permanently removes the DICOM files from Orthanc storage. Analysis results are unaffected."
                  : "This series is not currently stored in Orthanc."}
              </Typography>
            </Stack>
          </Stack>

          {/* Cross-prompt alerts */}
          {suggestDicom && (
            <Alert
              severity='info'
              sx={{ py: 0.5 }}
              action={
                <Button
                  size='small'
                  color='inherit'
                  onClick={() => actions.toggleDeleteOption("deleteDicom")}>
                  Also delete
                </Button>
              }>
              DICOM images also exist for this series. Would you like to delete
              them from Orthanc as well?
            </Alert>
          )}

          {suggestResults && (
            <Alert
              severity='info'
              sx={{ py: 0.5 }}
              action={
                <Button
                  size='small'
                  color='inherit'
                  onClick={() => actions.toggleDeleteOption("deleteResults")}>
                  Also delete
                </Button>
              }>
              Analysis results also exist in the database for this series. Would
              you like to delete them as well?
            </Alert>
          )}

          {neitherSelected && (
            <Alert severity='warning' sx={{ py: 0.5 }}>
              Select at least one item to delete.
            </Alert>
          )}
        </Stack>
      </DialogContent>

      <DialogActions>
        <Button onClick={actions.closeDeleteDialog} disabled={loading}>
          Cancel
        </Button>
        <Button
          color='error'
          variant='contained'
          onClick={actions.confirmDelete}
          disabled={loading || neitherSelected}>
          {loading ? "Deleting…" : "Delete"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DeleteConfirmationDialog;
