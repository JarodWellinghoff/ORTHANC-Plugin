import * as React from "react";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";

import { useDashboard } from "../context/DashboardContext";

function StatCard({ label, value }) {
  return (
    <Card
      variant='outlined'
      sx={{
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        minWidth: 250,
      }}>
      <CardContent>
        <Typography variant='h4' component='div' sx={{ fontWeight: 600 }}>
          {value ?? "-"}
        </Typography>
        <Typography variant='body2' color='text.secondary'>
          {label}
        </Typography>
      </CardContent>
    </Card>
  );
}

const StatsOverview = () => {
  const { stats } = useDashboard();
  const cards = [
    { id: "total", label: "Total Results", value: stats.total },
    {
      id: "detectability",
      label: "Detectability Results",
      value: stats.detectability,
    },
    { id: "noise", label: "Global Noise Results", value: stats.noise },
    { id: "errors", label: "No Data", value: stats.errors },
  ];

  return (
    <Grid
      container
      spacing={2}
      sx={{
        justifyContent: "center",
      }}>
      {cards.map((stat) => (
        <Grid item key={stat.id} size={{ xs: 12, md: 6, lg: 3 }}>
          <StatCard label={stat.label} value={stat.value} />
        </Grid>
      ))}
    </Grid>
  );
};

export default StatsOverview;
