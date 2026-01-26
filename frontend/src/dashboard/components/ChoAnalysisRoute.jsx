import * as React from "react";
import { useNavigate, useParams } from "react-router-dom";

import ChoAnalysisPage from "./ChoAnalysisPage";
import { useDashboard } from "../context/DashboardContext";

const extractMatchingSeries = (items, targetId) => {
  if (!Array.isArray(items) || !targetId) {
    return null;
  }

  const normalized = String(targetId);

  return (
    items.find((item) => {
      const candidates = [
        item.series_id,
        item.series_uuid,
        item.series_instance_uid,
        item.seriesId,
        item.seriesUuid,
      ];
      return candidates.some((candidate) => {
        if (candidate === null || candidate === undefined || candidate === "") {
          return false;
        }
        return String(candidate) === normalized;
      });
    }) ?? null
  );
};

const ChoAnalysisRoute = () => {
  const { seriesId } = useParams();
  const navigate = useNavigate();
  const {
    summary,
    choModal,
    actions: { openChoModal, closeChoModal },
  } = useDashboard();

  const normalizedId = React.useMemo(() => {
    if (!seriesId || seriesId === "") {
      return null;
    }
    try {
      return decodeURIComponent(seriesId);
    } catch {
      return seriesId;
    }
  }, [seriesId]);

  React.useEffect(() => {
    if (!normalizedId) {
      navigate("/", { replace: true });
      return;
    }

    const matchesSeriesId =
      (choModal.seriesId && String(choModal.seriesId) === normalizedId) ||
      (choModal.seriesUuid && String(choModal.seriesUuid) === normalizedId);
    if (matchesSeriesId) {
      return;
    }

    const matchingSeries = extractMatchingSeries(summary.items, normalizedId);
    const fallbackSeries = {
      series_id: normalizedId,
      series_uuid: normalizedId,
      series_instance_uid: normalizedId,
    };

    openChoModal(matchingSeries ?? fallbackSeries);
  }, [
    normalizedId,
    choModal.seriesId,
    choModal.seriesUuid,
    summary.items,
    openChoModal,
    navigate,
  ]);

  React.useEffect(() => {
    return () => {
      closeChoModal();
    };
  }, [closeChoModal]);

  if (!normalizedId) {
    return null;
  }

  return <ChoAnalysisPage />;
};

export default ChoAnalysisRoute;


