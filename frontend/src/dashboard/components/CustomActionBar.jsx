import * as React from "react";
import dayjs from "dayjs";
import Button from "@mui/material/Button";
import DialogActions from "@mui/material/DialogActions";
import {
  usePickerContext,
  usePickerTranslations,
} from "@mui/x-date-pickers/hooks";

export default function CustomActionBar({
  actions = [],
  className,
  onAdd12Hours,
  add12Label = "+12 hours",
  add12Disabled = false,
}) {
  const t = usePickerTranslations();
  const {
    clearValue,
    setValueToToday,
    acceptValueChanges,
    cancelValueChanges,
    goToNextStep,
    hasNextStep,
  } = usePickerContext();

  return (
    <DialogActions className={className}>
      {actions.includes("today") && (
        <Button onClick={setValueToToday}>{t.todayButtonLabel}</Button>
      )}
      {actions.includes("clear") && (
        <Button onClick={clearValue}>{t.clearButtonLabel}</Button>
      )}
      {actions.includes("cancel") && (
        <Button onClick={cancelValueChanges}>{t.cancelButtonLabel}</Button>
      )}
      {actions.includes("next") && hasNextStep && (
        <Button onClick={goToNextStep}>{t.nextStepButtonLabel}</Button>
      )}
      {actions.includes("nextOrAccept") && (
        <Button onClick={hasNextStep ? goToNextStep : acceptValueChanges}>
          {hasNextStep ? t.nextStepButtonLabel : t.okButtonLabel}
        </Button>
      )}
      {actions.includes("accept") && (
        <Button onClick={acceptValueChanges}>{t.okButtonLabel}</Button>
      )}

      {/* Custom: trigger parent-provided handler */}
      <Button onClick={onAdd12Hours} disabled={add12Disabled}>
        {add12Label}
      </Button>
    </DialogActions>
  );
}
