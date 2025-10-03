import * as React from "react";
import Container from "@mui/material/Container";
import Stack from "@mui/material/Stack";

import StatsOverview from "./StatsOverview";
import FiltersPanel from "./FiltersPanel";
import SummarySection from "./SummarySection";
import ModalsHost from "./ModalsHost";

const DashboardContent = () => {
  return (
    <Container
      maxWidth='xl'
      sx={{
        py: 4,
      }}>
      <Stack spacing={3}>
        <StatsOverview />
        <FiltersPanel />
        <SummarySection />
      </Stack>
      <ModalsHost />
    </Container>
  );
};

export default DashboardContent;
