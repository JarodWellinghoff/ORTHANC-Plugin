import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

import { useDashboard } from "../context/DashboardContext";

const severityToColor = {
  idle: "default",
  loading: "info",
  success: "success",
  error: "error",
};

const StatusHeader = () => {
  const { status } = useDashboard();
  const chipColor = severityToColor[status.severity] ?? "default";

  return (
    <Stack
      direction='row'
      sx={{
        display: { xs: "none", md: "flex" },
        width: "100%",
        alignItems: { xs: "flex-start", md: "center" },
        justifyContent: "space-between",
        maxWidth: { sm: "100%", md: "1700px" },
        pt: 1.5,
      }}
      spacing={2}>
      <div>
        <Typography variant='h4' component='h1'>
          CHO Analysis Dashboard
        </Typography>
        <Typography variant='body2' color='text.secondary'>
          Advanced filtering and analysis of CT image quality metrics
        </Typography>
      </div>
      <Stack direction='row' spacing={1} alignItems='center'>
        {status.message ? (
          <Chip label={status.message} color={chipColor} variant='outlined' />
        ) : (
          <Chip label='Ready' color='default' variant='outlined' />
        )}
        {/* <Button variant='contained' onClick={actions.refresh}>
          Refresh
        </Button> */}
      </Stack>
    </Stack>
  );
};

export default StatusHeader;
