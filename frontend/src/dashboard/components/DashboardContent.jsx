import Stack from "@mui/material/Stack";

import StatsOverview from "./StatsOverview";
import FiltersPanel from "./FiltersPanel";
import SummarySection from "./SummarySection";

const DashboardContent = () => {
  return (
    <Stack spacing={3}>
      {/* <StatsOverview /> */}
      <FiltersPanel />
      <SummarySection />
    </Stack>
  );
};

export default DashboardContent;
