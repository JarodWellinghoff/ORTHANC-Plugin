// hooks/useFilters.js
import { useCallback, useState } from "react";

export const defaultFilters = {
  patientIdSearch: [],
  patientNameSearch: [],
  instituteSearch: [],
  protocolNameSearch: [],
  scannerModelSearch: [],
  scannerStationSearch: [],
  pullScheduleSearch: [],
  studyDateStartSearch: null,
  studyDateEndSearch: null,
  ageStartSearch: 0,
  ageEndSearch: 200,
};

export const useFilters = (initial = defaultFilters) => {
  const [filters, setFilters] = useState(initial);

  const updateFilter = useCallback((name, value) => {
    setFilters((prev) => ({ ...prev, [name]: value }));
  }, []);

  const resetFilters = useCallback(() => setFilters(initial), [initial]);

  return { filters, updateFilter, resetFilters, setFilters };
};
