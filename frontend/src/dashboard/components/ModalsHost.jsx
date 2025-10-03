import * as React from "react";

import ChoAnalysisModal from "./modals/ChoAnalysisModal";
import DeleteConfirmationDialog from "./modals/DeleteConfirmationDialog";
import SeriesDetailsModal from "./modals/SeriesDetailsModal";

const ModalsHost = () => {
  return (
    <>
      <SeriesDetailsModal />
      <ChoAnalysisModal />
      <DeleteConfirmationDialog />
    </>
  );
};

export default ModalsHost;
