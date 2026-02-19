#!/usr/bin/env python3

import numpy as np
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore
import os
from datetime import datetime
import orthanc
import traceback
from minio import Minio
from minio.error import S3Error
from PIL import Image
from psycopg2 import pool
import io


class CHOResultsStorage:
    def __init__(self):
        # Database connection parameters
        self.postgres_config = {
            "host": "postgres",
            "port": 5432,
            "database": "orthanc",
            "user": "postgres",
            "password": "pgpassword",
        }
        self.minio_config = {
            "endpoint": "minio:9000",
            "access_key": "minio",
            "secret_key": "minio12345",
            "secure": False,
        }
        self.bucket_name = "images"
        self.postgres_connection = None
        self.minio_client = None
        # SQL files paths - can be customized
        self.sql_files_path = "/src/sql"  # Default path, adjust as needed
        self.dicom_sql_file = "dicom.sql"
        self.analysis_sql_file = "analysis.sql"
        self.init_postgres_database()
        self.init_minio_client()

    def connect_postgres(self):
        """Establish database connection"""
        try:
            self.postgres_connection = psycopg2.connect(**self.postgres_config)
            print("Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {str(e)}")
            print(traceback.format_exc())
            return False

    def _check_database_exists(self):
        """Check if the database schemas and tables exist"""
        if not self.connect_postgres() or self.postgres_connection is None:
            print(f"Failed to connect to PostgreSQL database")
            return False
        try:
            with self.postgres_connection.cursor() as cursor:
                # Check for required schemas
                cursor.execute(
                    """
                    SELECT schema_name FROM information_schema.schemata 
                    WHERE schema_name IN ('dicom', 'analysis')
                """
                )
                schemas = [row[0] for row in cursor.fetchall()]

                # Check for key tables
                tables_to_check = [
                    ("dicom", "patient"),
                    ("dicom", "scanner"),
                    ("dicom", "study"),
                    ("dicom", "series"),
                    ("dicom", "ct_technique"),
                    ("analysis", "results"),
                ]

                existing_tables = []
                for schema, table in tables_to_check:
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = %s AND table_name = %s
                        )
                    """,
                        (schema, table),
                    )
                    if cursor.fetchone()[0]:
                        existing_tables.append(f"{schema}.{table}")

                # Database is considered to exist if we have both schemas and all key tables
                has_dicom_schema = "dicom" in schemas
                has_analysis_schema = "analysis" in schemas
                has_all_tables = len(existing_tables) == len(tables_to_check)

                return has_dicom_schema and has_analysis_schema and has_all_tables

        except Exception as e:
            print(f"Error checking database existence: {str(e)}")
            return False

    def _find_sql_files(self):
        """Find SQL files in various possible locations"""
        possible_paths = [
            ".",
            "sql",
            "/src/sql",
            "../sql",
            "../../sql",
            "/src",
            "/app/sql",
            "/opt/app/sql",
        ]

        sql_files = {}

        for base_path in possible_paths:
            dicom_path = os.path.join(base_path, self.dicom_sql_file)
            analysis_path = os.path.join(base_path, self.analysis_sql_file)

            if os.path.exists(dicom_path) and os.path.exists(analysis_path):
                sql_files["dicom"] = dicom_path
                sql_files["analysis"] = analysis_path
                print(f"Found SQL files in: {base_path}")
                return sql_files

        # If not found together, try individual files
        for base_path in possible_paths:
            if not sql_files.get("dicom"):
                dicom_path = os.path.join(base_path, self.dicom_sql_file)
                if os.path.exists(dicom_path):
                    sql_files["dicom"] = dicom_path

            if not sql_files.get("analysis"):
                analysis_path = os.path.join(base_path, self.analysis_sql_file)
                if os.path.exists(analysis_path):
                    sql_files["analysis"] = analysis_path

        return sql_files

    def _execute_sql_file(self, file_path, description="SQL file"):
        """Execute a SQL file"""
        if not self.connect_postgres() or self.postgres_connection is None:
            print(f"Failed to connect to PostgreSQL database")
            return False

        try:
            print(f"Executing {description}: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            statements = [
                stmt.strip() for stmt in sql_content.split(";") if stmt.strip()
            ]

            with self.postgres_connection.cursor() as cursor:
                for i, statement in enumerate(statements):
                    if statement:
                        try:
                            cursor.execute(statement)
                            print(f"\tExecuted statement {i+1}/{len(statements)}")
                        except psycopg2.Error as e:
                            if "already exists" in str(e).lower():
                                print(f"\tStatement {i+1} - object already exists (OK)")
                            else:
                                print(f"\tStatement {i+1} failed: {str(e)}")

                self.postgres_connection.commit()
                print(f"\tSuccessfully executed {description}")
                return True

        except Exception as e:
            print(f"\tError executing {description}: {str(e)}")
            print(traceback.format_exc())
            if self.postgres_connection:
                self.postgres_connection.rollback()
            return False

    def _create_database_from_sql_files(self):
        """Create database schema from SQL files"""
        print("Database schema not found. Creating from SQL files...")

        sql_files = self._find_sql_files()

        if not sql_files.get("dicom") or not sql_files.get("analysis"):
            missing = []
            if not sql_files.get("dicom"):
                missing.append(self.dicom_sql_file)
            if not sql_files.get("analysis"):
                missing.append(self.analysis_sql_file)
            print(f"Could not find required SQL files: {', '.join(missing)}")
            return False

        success = True
        if not self._execute_sql_file(sql_files["dicom"], "DICOM schema"):
            success = False
        if not self._execute_sql_file(sql_files["analysis"], "Analysis schema"):
            success = False

        return success

    def init_postgres_database(self):
        """Initialize the PostgreSQL database"""
        if not self.connect_postgres() or self.postgres_connection is None:
            print("Failed to connect to PostgreSQL for initialization")
            return False

        try:
            if self._check_database_exists():
                print("Database schema already exists, skipping creation")
            else:
                if not self._create_database_from_sql_files():
                    print("Failed to create database schema from SQL files")
                    return False

            with self.postgres_connection.cursor() as cursor:
                indices_to_create = [
                    (
                        "idx_analysis_results_series_fk",
                        "analysis.results",
                        "series_id_fk",
                    ),
                    (
                        "idx_analysis_results_created_at",
                        "analysis.results",
                        "created_at",
                    ),
                    (
                        "idx_dicom_series_instance_uid",
                        "dicom.series",
                        "series_instance_uid",
                    ),
                    (
                        "idx_dicom_study_instance_uid",
                        "dicom.study",
                        "study_instance_uid",
                    ),
                    ("idx_dicom_patient_patient_id", "dicom.patient", "patient_id"),
                ]

                for index_name, table_name, column_name in indices_to_create:
                    try:
                        cursor.execute(
                            f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})"
                        )
                        print(f"\tCreated/verified index: {index_name}")
                    except Exception as e:
                        print(f"\tIndex {index_name}: {str(e)}")

                self.postgres_connection.commit()
                print("Database initialization completed successfully")
                return True

        except Exception as e:
            print(f"Failed to initialize database: {str(e)}")
            print(traceback.format_exc())
            if self.postgres_connection:
                self.postgres_connection.rollback()
            return False

    def init_minio_client(self):
        """Initialize MinIO client"""
        try:
            self.minio_client = Minio(
                endpoint=self.minio_config["endpoint"],
                access_key=self.minio_config["access_key"],
                secret_key=self.minio_config["secret_key"],
                secure=self.minio_config["secure"],
            )
            # Ensure bucket exists
            if not self.minio_client.bucket_exists(bucket_name=self.bucket_name):
                self.minio_client.make_bucket(bucket_name=self.bucket_name)
                print(f"Created MinIO bucket: {self.bucket_name}")
            else:
                print(f"MinIO bucket already exists: {self.bucket_name}")
            return True
        except Exception as e:
            print(f"Failed to initialize MinIO client: {str(e)}")
            self.minio_client = None
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal DB helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_create_patient(self, cursor, patient_info):
        """Get or create patient record"""
        cursor.execute(
            "SELECT id FROM dicom.patient WHERE patient_id = %s",
            (patient_info.get("patient_id"),),
        )
        result = cursor.fetchone()
        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO dicom.patient (patient_id, name, birth_date, sex)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """,
            (
                patient_info.get("patient_id"),
                patient_info.get("name"),
                patient_info.get("birth_date"),
                patient_info.get("sex"),
            ),
        )
        return cursor.fetchone()[0]

    def _get_or_create_scanner(self, cursor, scanner_info):
        """Get or create scanner record"""
        cursor.execute(
            "SELECT id FROM dicom.scanner WHERE device_serial_number = %s",
            (scanner_info.get("device_serial_number"),),
        )
        result = cursor.fetchone()
        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO dicom.scanner (
                manufacturer, model_name, device_serial_number,
                software_versions, station_name, institution_name
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                scanner_info.get("manufacturer"),
                scanner_info.get("model_name"),
                scanner_info.get("device_serial_number"),
                scanner_info.get("software_versions"),
                scanner_info.get("station_name"),
                scanner_info.get("institution_name"),
            ),
        )
        return cursor.fetchone()[0]

    def _get_or_create_study(self, cursor, study_info, patient_id):
        """Get or create study record"""
        cursor.execute(
            "SELECT id FROM dicom.study WHERE study_instance_uid = %s",
            (study_info.get("study_instance_uid"),),
        )
        result = cursor.fetchone()
        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO dicom.study (
                patient_id_fk, study_instance_uid, study_id, accession_number,
                study_date, study_time, description, referring_physician,
                institution_name, institution_address
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                patient_id,
                study_info.get("study_instance_uid"),
                study_info.get("study_id"),
                study_info.get("accession_number"),
                study_info.get("study_date"),
                study_info.get("study_time"),
                study_info.get("description"),
                study_info.get("referring_physician"),
                study_info.get("institution_name"),
                study_info.get("institution_address"),
            ),
        )
        return cursor.fetchone()[0]

    def _get_or_create_series(self, cursor, series_info, study_id, scanner_id):
        """Get or create series record"""
        cursor.execute(
            "SELECT id FROM dicom.series WHERE series_instance_uid = %s",
            (series_info.get("series_instance_uid"),),
        )
        result = cursor.fetchone()
        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO dicom.series (
                uuid, study_id_fk, series_instance_uid, series_number,
                description, modality, body_part_examined, protocol_name,
                convolution_kernel, patient_position, series_date, series_time,
                frame_of_reference_uid, image_type, slice_thickness_mm,
                pixel_spacing_mm, rows, columns, scanner_id_fk,
                image_count, scan_length_cm
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                series_info.get("uuid"),
                study_id,
                series_info.get("series_instance_uid"),
                series_info.get("series_number"),
                series_info.get("description"),
                series_info.get("modality"),
                series_info.get("body_part_examined"),
                series_info.get("protocol_name"),
                series_info.get("convolution_kernel"),
                series_info.get("patient_position"),
                series_info.get("series_date"),
                series_info.get("series_time"),
                series_info.get("frame_of_reference_uid"),
                series_info.get("image_type"),
                series_info.get("slice_thickness_mm"),
                series_info.get("pixel_spacing_mm"),
                series_info.get("rows"),
                series_info.get("columns"),
                scanner_id,
                series_info.get("image_count"),
                series_info.get("scan_length_cm"),
            ),
        )
        return cursor.fetchone()[0]

    def _get_or_create_ct_technique(self, cursor, ct_info, series_id):
        """Get or create CT technique record"""
        cursor.execute(
            "SELECT series_id_fk FROM dicom.ct_technique WHERE series_id_fk = %s",
            (series_id,),
        )
        result = cursor.fetchone()
        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO dicom.ct_technique (
                series_id_fk, kvp, exposure_time_ms, generator_power_kw,
                focal_spots_mm, filter_type, data_collection_diam_mm,
                recon_diameter_mm, dist_src_detector_mm, dist_src_patient_mm,
                gantry_detector_tilt_deg, single_collimation_width_mm,
                total_collimation_width_mm, table_speed_mm_s,
                table_feed_per_rot_mm, spiral_pitch_factor,
                exposure_modulation_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s)
            RETURNING series_id_fk
        """,
            (
                series_id,
                ct_info.get("kvp"),
                ct_info.get("exposure_time_ms"),
                ct_info.get("generator_power_kW"),
                ct_info.get("focal_spots_mm"),
                ct_info.get("filter_type"),
                ct_info.get("data_collection_diam_mm"),
                ct_info.get("recon_diameter_mm"),
                ct_info.get("dist_src_detector_mm"),
                ct_info.get("dist_src_patient_mm"),
                ct_info.get("gantry_detector_tilt_deg"),
                ct_info.get("single_collimation_width_mm"),
                ct_info.get("total_collimation_width_mm"),
                ct_info.get("table_speed_mm_s"),
                ct_info.get("table_feed_per_rot_mm"),
                ct_info.get("spiral_pitch_factor"),
                ct_info.get("exposure_modulation_type"),
            ),
        )
        return cursor.fetchone()[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def save_results(self, patient, study, scanner, series, ct, results):
        """Save CHO calculation results to the database with new schema"""
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        series_id = None
        success = False

        try:
            with self.postgres_connection.cursor() as cursor:
                patient_id = self._get_or_create_patient(cursor, patient)
                scanner_id = self._get_or_create_scanner(cursor, scanner)
                study_id = self._get_or_create_study(cursor, study, patient_id)
                series_id = self._get_or_create_series(
                    cursor, series, study_id, scanner_id
                )
                self._get_or_create_ct_technique(cursor, ct, series_id)

                # Insert analysis results
                cursor.execute(
                    """
                    INSERT INTO analysis.results (
                        series_id_fk,
                        processing_time,
                        ctdivol, ctdivol_avg, dlp, dlp_ssde, dw, dw_avg,
                        location, ssde,
                        average_frequency, average_index_of_detectability,
                        average_noise_level, cho_detectability, location_sparse,
                        noise_level, nps, peak_frequency, percent_10_frequency,
                        spatial_frequency, spatial_resolution
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """,
                    (
                        series_id,
                        results.get("processing_time"),
                        results.get("ctdivol"),
                        results.get("ctdivol_avg"),
                        results.get("dlp"),
                        results.get("dlp_ssde"),
                        results.get("dw"),
                        results.get("dw_avg"),
                        results.get("location"),
                        results.get("ssde"),
                        results.get("average_frequency"),
                        results.get("average_index_of_detectability"),
                        results.get("average_noise_level"),
                        results.get("cho_detectability"),
                        results.get("location_sparse"),
                        results.get("noise_level"),
                        results.get("nps"),
                        results.get("peak_frequency"),
                        results.get("percent_10_frequency"),
                        results.get("spatial_frequency"),
                        results.get("spatial_resolution"),
                    ),
                )

                self.postgres_connection.commit()
                success = True
                print(f"Results saved successfully for series_id={series_id}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            print(traceback.format_exc())
            if self.postgres_connection:
                self.postgres_connection.rollback()

        return success

    def save_dicom_headers_only(self, patient, study, scanner, series, ct):
        """Save DICOM metadata without analysis results"""
        return self.save_results(
            patient, study, scanner, series, ct, results={"processing_time": 0.0}
        )

    def delete_results(self, series_instance_uid):
        """Delete CHO results for a specific series"""
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        try:
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM analysis.results
                    WHERE series_id_fk = (
                        SELECT id FROM dicom.series
                        WHERE series_instance_uid = %s
                    )
                """,
                    (series_instance_uid,),
                )
                deleted = cursor.rowcount > 0
            self.postgres_connection.commit()
            return deleted
        except Exception as e:
            print(f"Error deleting results: {str(e)}")
            if self.postgres_connection:
                self.postgres_connection.rollback()
            return False

    def get_result(self, series_instance_uid):
        """Get full result record for a single series"""
        if not self.connect_postgres() or self.postgres_connection is None:
            raise Exception("No database connection")

        try:
            with self.postgres_connection.cursor() as cursor:
                base_query = """
                    SELECT 
                        r.*,
                        s.series_instance_uid,
                        s.uuid as series_uuid,
                        p.patient_id,
                        p.name as patient_name,
                        p.birth_date,
                        p.sex,
                        st.study_instance_uid,
                        st.institution_name,
                        st.study_date,
                        sc.manufacturer,
                        sc.model_name as scanner_model,
                        sc.station_name,
                        s.protocol_name,
                        s.modality,
                        s.body_part_examined,
                        s.scan_length_cm,
                        st.study_id,
                        st.description as study_description,
                        st.study_date,
                        st.study_time,
                        s.description as series_description,
                        s.series_number,
                        s.convolution_kernel,
                        s.image_count,
                        s.patient_position,
                        s.pixel_spacing_mm,
                        s.rows,
                        s.columns,
                        s.series_date,
                        s.series_time,
                        s.slice_thickness_mm,
                        sc.device_serial_number,
                        ct.kvp,
                        ct.exposure_time_ms,
                        ct.generator_power_kw,
                        ct.focal_spots_mm,
                        ct.filter_type,
                        ct.data_collection_diam_mm,
                        ct.recon_diameter_mm,
                        ct.dist_src_detector_mm,
                        ct.dist_src_patient_mm,
                        ct.gantry_detector_tilt_deg,
                        ct.single_collimation_width_mm,
                        ct.total_collimation_width_mm,
                        ct.table_speed_mm_s,
                        ct.table_feed_per_rot_mm,
                        ct.spiral_pitch_factor,
                        ct.exposure_modulation_type
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    JOIN dicom.scanner sc ON s.scanner_id_fk = sc.id
                    JOIN dicom.ct_technique ct ON s.id = ct.series_id_fk
                    WHERE s.series_instance_uid = %s
                    ORDER BY r.created_at DESC
                    LIMIT 1
                """
                cursor.execute(base_query, (series_instance_uid,))
                return cursor.fetchone(), cursor.description
        except Exception as e:
            self.postgres_connection.rollback()
            raise e

    def get_results_statistics(self):
        """Get aggregate statistics for the results dashboard"""
        if not self.connect_postgres() or self.postgres_connection is None:
            raise Exception("No database connection")

        try:
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH counts AS (
                        SELECT
                        COUNT(*) AS total_results_count,
                        COUNT(*) FILTER (
                            WHERE (ctdivol, ctdivol_avg, dlp, dlp_ssde, dw, dw_avg, "location", ssde) is not null
                            AND (average_frequency, average_index_of_detectability, average_noise_level, cho_detectability,
                            location_sparse, noise_level, nps, peak_frequency, percent_10_frequency, spatial_frequency,
                            spatial_resolution) IS NULL
                        ) AS global_noise_count,
                        COUNT(*) FILTER (
                            WHERE r.* IS NOT NULL
                        ) AS detectability_count
                        FROM analysis.results r
                    )
                    SELECT
                        total_results_count,
                        global_noise_count,
                        detectability_count,
                        total_results_count
                            - (global_noise_count + detectability_count) AS error_count
                    FROM counts;
                """
                )
                return cursor.fetchone()
        except Exception as e:
            self.postgres_connection.rollback()
            raise e

    def get_results(self, params):
        """Retrieve CHO results from the database with optional filtering.

        CHANGED: Now LEFT JOINs workflow.dicom_pull_items and
        workflow.dicom_pull_batches so each row carries
        `pull_schedule_name` (the display_name of the batch that
        originally pulled this series).  A new `pull_schedule_name`
        query param lets callers filter by that name.
        """
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        try:
            with self.postgres_connection.cursor() as cursor:
                # ── query parameters ──────────────────────────────────────
                patient_id = params.get("patient_id")
                study_id = params.get("study_id")
                patient_search = params.get("patient_search")
                institute = params.get("institute")
                station_name = params.get("scanner_station")
                protocol_name = params.get("protocol_name")
                scanner_model = params.get("scanner_model")
                exam_date_from = params.get("exam_date_from")
                exam_date_to = params.get("exam_date_to")
                patient_age_min = params.get("patient_age_min")
                patient_age_max = params.get("patient_age_max")
                # ── CHANGED: new filter param ─────────────────────────────
                pull_schedule_name = params.get("pull_schedule_name")
                # ─────────────────────────────────────────────────────────

                # ── pagination ────────────────────────────────────────────
                page = int(params.get("page", "1"))
                limit = int(params.get("limit", "25"))
                page = max(1, page)
                limit = min(max(1, limit), 1000)
                offset = (page - 1) * limit

                # ── CHANGED: base_query now LEFT JOINs the pull schedule ──
                #
                # The join path is:
                #   dicom.series.series_instance_uid
                #     → workflow.dicom_pull_items.series_instance_uid  (many rows possible; use the most recent)
                #     → workflow.dicom_pull_batches.id  (carries display_name)
                #
                # We use a LATERAL subquery so we pick only the latest
                # pull-item for each series, avoiding row duplication.
                base_query = """
                    FROM dicom.series s
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    JOIN dicom.scanner sc ON s.scanner_id_fk = sc.id
                    LEFT JOIN analysis.results r ON s.id = r.series_id_fk
                    LEFT JOIN LATERAL (
                        SELECT dpb.display_name
                        FROM workflow.dicom_pull_items dpi
                        JOIN workflow.dicom_pull_batches dpb ON dpb.id = dpi.batch_id
                        WHERE dpi.series_instance_uid = s.series_instance_uid
                        ORDER BY dpi.created_at DESC
                        LIMIT 1
                    ) pull_info ON true
                """
                # ─────────────────────────────────────────────────────────

                query_params = []
                conditions = []

                if patient_id:
                    conditions.append("p.patient_id ILIKE %s")
                    query_params.append(f"%{patient_id}%")

                if study_id:
                    conditions.append("st.study_instance_uid = %s")
                    query_params.append(study_id)

                if patient_search:
                    conditions.append("(p.name ILIKE %s OR p.patient_id ILIKE %s)")
                    query_params.extend([f"%{patient_search}%", f"%{patient_search}%"])

                if institute:
                    conditions.append("st.institution_name ILIKE %s")
                    query_params.append(f"%{institute}%")

                if station_name:
                    conditions.append("sc.station_name ILIKE %s")
                    query_params.append(f"%{station_name}%")

                if protocol_name:
                    conditions.append("s.protocol_name ILIKE %s")
                    query_params.append(f"%{protocol_name}%")

                if scanner_model:
                    conditions.append("sc.model_name ILIKE %s")
                    query_params.append(f"%{scanner_model}%")

                if exam_date_from:
                    conditions.append("st.study_date >= %s")
                    query_params.append(exam_date_from)

                if exam_date_to:
                    conditions.append("st.study_date <= %s")
                    query_params.append(exam_date_to)

                if patient_age_min:
                    conditions.append(
                        "EXTRACT(YEAR FROM AGE(st.study_date::date, p.birth_date::date)) >= %s"
                    )
                    query_params.append(int(patient_age_min))

                if patient_age_max:
                    conditions.append(
                        "EXTRACT(YEAR FROM AGE(st.study_date::date, p.birth_date::date)) <= %s"
                    )
                    query_params.append(int(patient_age_max))

                # ── CHANGED: pull schedule name filter ────────────────────
                if pull_schedule_name:
                    conditions.append("pull_info.display_name ILIKE %s")
                    query_params.append(f"%{pull_schedule_name}%")
                # ─────────────────────────────────────────────────────────

                where_clause = ""
                if conditions:
                    where_clause = " WHERE " + " AND ".join(conditions)

                having_clause = " HAVING COUNT(r.id) > 0"

                # Count total results for pagination
                count_query = f"""
                    SELECT COUNT(DISTINCT s.id)
                    {base_query}
                    {where_clause}
                    GROUP BY s.id
                    {having_clause}
                """
                total_query = f"SELECT COUNT(*) FROM ({count_query}) as counted"
                cursor.execute(total_query, query_params)
                total_results = cursor.fetchone()[0]

                total_pages = (
                    (total_results + limit - 1) // limit if total_results > 0 else 1
                )

                # ── CHANGED: SELECT now includes pull_schedule_name ───────
                query = f"""
                    SELECT 
                        s.series_instance_uid AS series_id,
                        s.uuid                AS series_uuid,
                        p.patient_id,
                        p.name        AS patient_name,
                        p.birth_date,
                        st.study_instance_uid AS study_id,
                        st.institution_name,
                        st.study_date,
                        sc.manufacturer,
                        sc.model_name AS scanner_model,
                        sc.station_name,
                        s.protocol_name,
                        pull_info.display_name AS pull_schedule_name,

                        CASE
                        WHEN COUNT(r.id) FILTER (
                                WHERE 
                                (ctdivol, ctdivol_avg, dlp, dlp_ssde, dw, dw_avg, "location", ssde) IS NOT NULL
                            AND (average_frequency, average_index_of_detectability, average_noise_level,
                                    cho_detectability, location_sparse, noise_level, nps, peak_frequency,
                                    percent_10_frequency, spatial_frequency, spatial_resolution) IS NOT NULL
                            ) > 0
                            THEN 'full'

                        WHEN COUNT(r.id) FILTER (
                                WHERE 
                                (ctdivol, ctdivol_avg, dlp, dlp_ssde, dw, dw_avg, "location", ssde) IS NOT NULL
                            AND (average_frequency, average_index_of_detectability, average_noise_level,
                                    cho_detectability, location_sparse, noise_level, nps, peak_frequency,
                                    percent_10_frequency, spatial_frequency, spatial_resolution) IS NULL
                            ) > 0
                            THEN 'partial'

                        ELSE 'error'
                        END AS test_status,

                        MAX(r.created_at) AS latest_analysis_date
                        
                    {base_query}
                    {where_clause}
                    GROUP BY s.id, s.series_instance_uid, s.uuid, p.patient_id, p.name, p.birth_date,
                            st.study_instance_uid, st.institution_name, st.study_date,
                            sc.manufacturer, sc.model_name, sc.station_name, s.protocol_name,
                            pull_info.display_name
                    {having_clause}
                    ORDER BY MAX(r.created_at) DESC NULLS LAST 
                    LIMIT %s OFFSET %s
                """
                # ─────────────────────────────────────────────────────────
                query_params.extend([limit, offset])

                cursor.execute(query, query_params)
                results = cursor.fetchall()

                result_list = []
                if cursor.description is not None:
                    columns = [desc[0] for desc in cursor.description]
                    for row in results:
                        result_dict = dict(zip(columns, row))
                        for date_field in [
                            "first_analysis_date",
                            "latest_analysis_date",
                            "study_date",
                            "birth_date",
                        ]:
                            if date_field in result_dict and result_dict[date_field]:
                                if isinstance(result_dict[date_field], datetime):
                                    result_dict[date_field] = result_dict[
                                        date_field
                                    ].isoformat()
                                elif isinstance(result_dict[date_field], str):
                                    try:
                                        dt = datetime.fromisoformat(
                                            result_dict[date_field].replace(
                                                "Z", "+00:00"
                                            )
                                        )
                                        result_dict[date_field] = dt.isoformat()
                                    except Exception:
                                        pass
                        result_list.append(result_dict)

            return {
                "data": result_list,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_results,
                    "pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
                # Legacy fields for backward compatibility
                "page": page,
                "limit": limit,
                "total": total_results,
                "pages": total_pages,
            }

        except Exception as e:
            raise e

    def get_filter_options(self):
        """Get filter options for UI dropdowns.

        CHANGED: now also returns `pull_schedule_names` — the distinct
        display_name values from workflow.dicom_pull_batches.
        """
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        try:
            with self.postgres_connection.cursor() as cursor:
                filter_options = {}

                # Institutes
                cursor.execute(
                    """
                    SELECT DISTINCT st.institution_name 
                    FROM dicom.study st 
                    WHERE st.institution_name IS NOT NULL 
                    ORDER BY st.institution_name
                """
                )
                filter_options["institutes"] = [row[0] for row in cursor.fetchall()]

                # Scanner names (station names)
                cursor.execute(
                    """
                    SELECT DISTINCT sc.station_name 
                    FROM dicom.scanner sc 
                    WHERE sc.station_name IS NOT NULL 
                    ORDER BY sc.station_name
                """
                )
                filter_options["scanner_stations"] = [
                    row[0] for row in cursor.fetchall()
                ]

                # Protocol names
                cursor.execute(
                    """
                    SELECT DISTINCT s.protocol_name 
                    FROM dicom.series s 
                    WHERE s.protocol_name IS NOT NULL 
                    ORDER BY s.protocol_name
                """
                )
                filter_options["protocol_names"] = [row[0] for row in cursor.fetchall()]

                # Scanner models
                cursor.execute(
                    """
                    SELECT DISTINCT sc.model_name 
                    FROM dicom.scanner sc 
                    WHERE sc.model_name IS NOT NULL 
                    ORDER BY sc.model_name
                """
                )
                filter_options["scanner_models"] = [row[0] for row in cursor.fetchall()]

                # Date range (min and max study dates)
                cursor.execute(
                    """
                    SELECT MIN(st.study_date), MAX(st.study_date)
                    FROM dicom.study st 
                    WHERE st.study_date IS NOT NULL
                """
                )
                date_range = cursor.fetchone()
                if date_range and date_range[0] and date_range[1]:
                    filter_options["date_range"] = {
                        "min": (
                            date_range[0].isoformat()
                            if isinstance(date_range[0], datetime)
                            else str(date_range[0])
                        ),
                        "max": (
                            date_range[1].isoformat()
                            if isinstance(date_range[1], datetime)
                            else str(date_range[1])
                        ),
                    }

                # ── CHANGED: pull schedule names ──────────────────────────
                cursor.execute(
                    """
                    SELECT DISTINCT dpb.display_name
                    FROM workflow.dicom_pull_batches dpb
                    WHERE dpb.display_name IS NOT NULL
                    ORDER BY dpb.display_name
                """
                )
                filter_options["pull_schedule_names"] = [
                    row[0] for row in cursor.fetchall()
                ]
                # ─────────────────────────────────────────────────────────

                return filter_options

        except Exception as e:
            print(f"Error getting filter options: {str(e)}")
            raise e

    def export_results(self):
        """Export all CHO results as CSV"""
        if not self.connect_postgres() or self.postgres_connection is None:
            raise Exception("No database connection")

        try:
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        s.series_instance_uid,
                        p.patient_id,
                        p.name AS patient_name,
                        st.study_date,
                        st.institution_name,
                        sc.model_name AS scanner_model,
                        sc.station_name,
                        s.protocol_name,
                        r.ctdivol, r.ctdivol_avg, r.dlp, r.dlp_ssde,
                        r.dw, r.dw_avg, r.location, r.ssde,
                        r.average_frequency, r.average_index_of_detectability,
                        r.average_noise_level, r.cho_detectability,
                        r.location_sparse, r.noise_level, r.nps,
                        r.peak_frequency, r.percent_10_frequency,
                        r.spatial_frequency, r.spatial_resolution,
                        r.processing_time, r.created_at
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    JOIN dicom.scanner sc ON s.scanner_id_fk = sc.id
                    ORDER BY r.created_at DESC
                """
                )
                rows = cursor.fetchall()
                if cursor.description is None:
                    return ""

                columns = [desc[0] for desc in cursor.description]
                lines = [",".join(columns)]
                for row in rows:
                    line = ",".join(
                        "" if v is None else str(v).replace(",", ";") for v in row
                    )
                    lines.append(line)
                return "\n".join(lines)

        except Exception as e:
            self.postgres_connection.rollback()
            raise e

    # ─────────────────────────────────────────────────────────────────────────
    # MinIO / image helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _create_coronal_image(self, coronal_view_data):
        """Convert coronal view numpy array to PIL Image format"""
        coronal_view_data = np.array(coronal_view_data)
        try:
            if coronal_view_data is None or len(coronal_view_data.shape) != 2:
                print("Warning: Invalid coronal view data")
                return None

            min_val = np.min(coronal_view_data)
            max_val = np.max(coronal_view_data)

            if max_val > min_val:
                normalized = (
                    (coronal_view_data - min_val) / (max_val - min_val) * 255
                ).astype(np.uint8)
            else:
                normalized = np.zeros_like(coronal_view_data, dtype=np.uint8)

            image = Image.fromarray(normalized, mode="L")

            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            print(f"Error creating coronal image: {str(e)}")
            return None

    def save_coronal_image(self, series_instance_uid, coronal_view_data):
        """Save coronal view image to MinIO"""
        if self.minio_client is None:
            print("MinIO client not initialized")
            return None

        try:
            image = self._create_coronal_image(coronal_view_data)
            if image is None:
                return None

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            size = img_bytes.getbuffer().nbytes

            object_name = f"coronal/{series_instance_uid}.png"
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=img_bytes,
                length=size,
                content_type="image/png",
            )
            print(f"Saved coronal image: {object_name}")
            return object_name

        except S3Error as e:
            print(f"MinIO error saving coronal image: {str(e)}")
            return None
        except Exception as e:
            print(f"Error saving coronal image: {str(e)}")
            return None

    def get_coronal_image(self, series_instance_uid):
        """Retrieve coronal view image from MinIO"""
        if self.minio_client is None:
            return None

        try:
            object_name = f"coronal/{series_instance_uid}.png"
            response = self.minio_client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error:
            return None
        except Exception as e:
            print(f"Error retrieving coronal image: {str(e)}")
            return None


# Global singleton
cho_storage = CHOResultsStorage()
