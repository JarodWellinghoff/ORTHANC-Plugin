import { useMemo, useReducer } from "react";
import AddIcon from "@mui/icons-material/Add";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

import { groupRecords } from "../../utils/aggregateMetrics";
import ProtocolPlotCard from "./ProtocolPlotCard";
import ProtocolPlotStatsTable from "./ProtocolPlotStatsTable";
import {
  createInitialPlotState,
  PLOT_COLUMNS,
  PROTOCOL_GROUP_KEY,
  protocolPlotReducer,
} from "./protocolPlotState";

const PlottingWorkspace = ({ records, loading }) => {
  const [state, dispatch] = useReducer(
    protocolPlotReducer,
    undefined,
    createInitialPlotState,
  );
  const protocolGroups = useMemo(
    () => groupRecords(records, PROTOCOL_GROUP_KEY),
    [records],
  );
  const protocolOptions = useMemo(
    () => protocolGroups.map((group) => group.label),
    [protocolGroups],
  );
  const columns = useMemo(
    () =>
      PLOT_COLUMNS.map((column) => ({
        ...column,
        plots: state.columns[column.mode].map((plot, index) => ({
          ...plot,
          title: `${column.plotLabel} ${index + 1}`,
        })),
      })),
    [state.columns],
  );
  const allPlots = useMemo(
    () => columns.flatMap((column) => column.plots),
    [columns],
  );

  return (
    <Stack spacing={3}>
      <Stack spacing={2}>
        <Typography variant='h5' component='h2' fontWeight={600}>
          Plots
        </Typography>
        <Typography variant='body2' color='text.secondary'>
          Every plot is grouped by protocol. Choose a metric and up to 10
          protocols per plot, and add plots to either column independently.
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "minmax(0, 1fr)",
              lg: "repeat(2, minmax(0, 1fr))",
            },
            gap: 3,
            alignItems: "start",
          }}>
          {columns.map((column) => (
            <Stack
              key={column.mode}
              component='section'
              aria-label={column.title}
              spacing={2}
              sx={{ minWidth: 0 }}>
              <Stack
                direction='row'
                alignItems='center'
                justifyContent='space-between'
                spacing={1}>
                <Typography variant='h6' component='h3'>
                  {column.title}
                </Typography>
              </Stack>
              {column.plots.map((plot) => (
                <ProtocolPlotCard
                  key={plot.id}
                  config={plot}
                  protocolGroups={protocolGroups}
                  protocolOptions={protocolOptions}
                  loading={loading}
                  canRemove={column.plots.length > 1}
                  onRemove={() =>
                    dispatch({ type: "remove", mode: column.mode, id: plot.id })
                  }
                  onConfigChange={(patch) =>
                    dispatch({
                      type: "configure",
                      mode: column.mode,
                      id: plot.id,
                      patch,
                    })
                  }
                />
              ))}
              <Box display='flex' justifyContent='center'>
                <Button
                  size='small'
                  variant='outlined'
                  startIcon={<AddIcon />}
                  sx={{
                    width: "50%",
                  }}
                  onClick={() => dispatch({ type: "add", mode: column.mode })}>
                  {column.addLabel}
                </Button>
              </Box>
            </Stack>
          ))}
        </Box>
      </Stack>

      <Stack spacing={2}>
        <Typography variant='h5' component='h2' fontWeight={600}>
          All Results
        </Typography>
        <Typography variant='body2' color='text.secondary'>
          One summary per plot and protocol, including every protocol in the
          queried cohort. Plot selections and the 10-protocol display limit do
          not filter this table.
        </Typography>
        <ProtocolPlotStatsTable
          plots={allPlots}
          protocolGroups={protocolGroups}
          loading={loading}
        />
      </Stack>
    </Stack>
  );
};

export default PlottingWorkspace;
