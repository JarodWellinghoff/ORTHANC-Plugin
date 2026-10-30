import { createContext, useContext } from "react";

export const RerunReviewContext = createContext(null);

export function useRerunReviewQueue() {
  const context = useContext(RerunReviewContext);
  if (!context) throw new Error("Rerun review requires RerunReviewHost.");
  return context;
}
