-- =============================================================================
-- LESION referential integrity triggers
-- PostgreSQL does not support FK constraints on array elements, so we enforce
-- lesion.set.lesion_ids → lesion.lesion.id integrity via two triggers:
--   1. validate_lesion_ids        : prevent inserting/updating a set with invalid ids
--   2. protect_referenced_lesions : prevent deleting a lesion still used by a set
-- =============================================================================

-- 1. Guard inserts/updates on lesion.set
CREATE OR REPLACE FUNCTION lesion.validate_lesion_ids()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM unnest(NEW.lesion_ids) AS lid
        WHERE NOT EXISTS (
            SELECT 1 FROM lesion.lesion WHERE id = lid
        )
    ) THEN
        RAISE EXCEPTION
            'lesion.set.lesion_ids contains one or more invalid lesion ids';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_validate_lesion_ids ON lesion.set;
CREATE CONSTRAINT TRIGGER trg_validate_lesion_ids
    AFTER INSERT OR UPDATE ON lesion.set
    DEFERRABLE INITIALLY IMMEDIATE
    FOR EACH ROW EXECUTE FUNCTION lesion.validate_lesion_ids();


-- 2. Guard deletes on lesion.lesion
CREATE OR REPLACE FUNCTION lesion.protect_referenced_lesions()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM lesion.set
        WHERE OLD.id = ANY(lesion_ids)
    ) THEN
        RAISE EXCEPTION
            'Cannot delete lesion id=% — it is still referenced by one or more lesion sets',
            OLD.id;
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_protect_referenced_lesions ON lesion.lesion;
CREATE CONSTRAINT TRIGGER trg_protect_referenced_lesions
    AFTER DELETE ON lesion.lesion
    DEFERRABLE INITIALLY IMMEDIATE
    FOR EACH ROW EXECUTE FUNCTION lesion.protect_referenced_lesions();


-- =============================================================================
-- DICOM orphan cleanup triggers
-- =============================================================================

-- Trigger function to clean up orphaned series when results are deleted
CREATE OR REPLACE FUNCTION cleanup_orphaned_series()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if the series has any remaining analysis results
    IF NOT EXISTS (
        SELECT 1 FROM analysis.results 
        WHERE series_id_fk = OLD.series_id_fk
    ) THEN
        -- Delete the orphaned series
        DELETE FROM dicom.series WHERE id = OLD.series_id_fk;
        RAISE NOTICE 'Deleted orphaned series with id: %', OLD.series_id_fk;
    END IF;
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Trigger function to clean up orphaned studies when series are deleted
CREATE OR REPLACE FUNCTION cleanup_orphaned_studies()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if the study has any remaining series
    IF NOT EXISTS (
        SELECT 1 FROM dicom.series 
        WHERE study_id_fk = OLD.study_id_fk
    ) THEN
        -- Delete the orphaned study
        DELETE FROM dicom.study WHERE id = OLD.study_id_fk;
        RAISE NOTICE 'Deleted orphaned study with id: %', OLD.study_id_fk;
    END IF;
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Trigger function to clean up orphaned patients when studies are deleted
CREATE OR REPLACE FUNCTION cleanup_orphaned_patients()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if the patient has any remaining studies
    IF OLD.patient_id_fk IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM dicom.study 
        WHERE patient_id_fk = OLD.patient_id_fk
    ) THEN
        -- Delete the orphaned patient
        DELETE FROM dicom.patient WHERE id = OLD.patient_id_fk;
        RAISE NOTICE 'Deleted orphaned patient with id: %', OLD.patient_id_fk;
    END IF;
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Create the triggers
DROP TRIGGER IF EXISTS trigger_cleanup_orphaned_series ON analysis.results;
CREATE TRIGGER trigger_cleanup_orphaned_series
    AFTER DELETE ON analysis.results
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_orphaned_series();

DROP TRIGGER IF EXISTS trigger_cleanup_orphaned_studies ON dicom.series;
CREATE TRIGGER trigger_cleanup_orphaned_studies
    AFTER DELETE ON dicom.series
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_orphaned_studies();

DROP TRIGGER IF EXISTS trigger_cleanup_orphaned_patients ON dicom.study;
CREATE TRIGGER trigger_cleanup_orphaned_patients
    AFTER DELETE ON dicom.study
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_orphaned_patients();

-- Optional: Create a function to manually clean up all orphaned records
CREATE OR REPLACE FUNCTION cleanup_all_orphaned_records()
RETURNS TABLE(
    deleted_patients INTEGER,
    deleted_studies INTEGER,
    deleted_series INTEGER
) AS $$
DECLARE
    patient_count INTEGER := 0;
    study_count INTEGER := 0;
    series_count INTEGER := 0;
BEGIN
    -- Clean up orphaned series (no analysis results)
    WITH deleted AS (
        DELETE FROM dicom.series s
        WHERE NOT EXISTS (
            SELECT 1 FROM analysis.results r
            WHERE r.series_id_fk = s.id
        )
        RETURNING id
    )
    SELECT COUNT(*) INTO series_count FROM deleted;
    
    -- Clean up orphaned studies (no series)
    WITH deleted AS (
        DELETE FROM dicom.study st
        WHERE NOT EXISTS (
            SELECT 1 FROM dicom.series s
            WHERE s.study_id_fk = st.id
        )
        RETURNING id
    )
    SELECT COUNT(*) INTO study_count FROM deleted;
    
    -- Clean up orphaned patients (no studies)
    WITH deleted AS (
        DELETE FROM dicom.patient p
        WHERE NOT EXISTS (
            SELECT 1 FROM dicom.study st
            WHERE st.patient_id_fk = p.id
        )
        RETURNING id
    )
    SELECT COUNT(*) INTO patient_count FROM deleted;
    
    RETURN QUERY SELECT patient_count, study_count, series_count;
END;
$$ LANGUAGE plpgsql;