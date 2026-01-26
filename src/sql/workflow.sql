CREATE SCHEMA IF NOT EXISTS workflow AUTHORIZATION postgres;

CREATE OR REPLACE FUNCTION workflow.touch_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS workflow.dicom_pull_batches (
    id BIGSERIAL PRIMARY KEY,
    requested_by TEXT NULL,
    display_name TEXT NULL,
    remote_modality TEXT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending',
        'in_progress',
        'paused',
        'completed',
        'failed',
        'cancelled',
        'expired'
    )),
    estimated_total_seconds INTEGER NOT NULL DEFAULT 0,
    actual_total_seconds INTEGER NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    failure_reason TEXT NULL,
    notes TEXT NULL,
    timezone TEXT DEFAULT 'UTC'
);

CREATE TABLE IF NOT EXISTS workflow.dicom_pull_items (
    id BIGSERIAL PRIMARY KEY,
    batch_id BIGINT NOT NULL REFERENCES workflow.dicom_pull_batches(id) ON DELETE CASCADE,
    external_patient_id TEXT NULL,
    patient_name TEXT NULL,
    study_instance_uid TEXT NOT NULL,
    series_instance_uid TEXT NULL,
    description TEXT NULL,
    modality TEXT NULL,
    body_part TEXT NULL,
    study_date TEXT NULL,
    series_date TEXT NULL,
    number_of_instances INTEGER NULL,
    estimated_seconds INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending',
        'in_progress',
        'completed',
        'failed',
        'skipped',
        'expired',
        'cancelled'
    )),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    failure_reason TEXT NULL,
    orthanc_series_id TEXT NULL,
    metadata JSONB NULL
);

CREATE INDEX IF NOT EXISTS dicom_pull_items_batch_idx
    ON workflow.dicom_pull_items (batch_id);

CREATE INDEX IF NOT EXISTS dicom_pull_batches_status_idx
    ON workflow.dicom_pull_batches (status, start_time);

CREATE INDEX IF NOT EXISTS dicom_pull_items_status_idx
    ON workflow.dicom_pull_items (status);

DROP TRIGGER IF EXISTS dicom_pull_batches_touch_updated_at ON workflow.dicom_pull_batches;
CREATE TRIGGER dicom_pull_batches_touch_updated_at
BEFORE UPDATE ON workflow.dicom_pull_batches
FOR EACH ROW EXECUTE FUNCTION workflow.touch_updated_at();

DROP TRIGGER IF EXISTS dicom_pull_items_touch_updated_at ON workflow.dicom_pull_items;
CREATE TRIGGER dicom_pull_items_touch_updated_at
BEFORE UPDATE ON workflow.dicom_pull_items
FOR EACH ROW EXECUTE FUNCTION workflow.touch_updated_at();
