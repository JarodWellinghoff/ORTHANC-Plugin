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
        self.minio_connection = None
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
            # Current directory
            ".",
            # SQL directory
            "sql",
            "/src/sql",
            # Parent directories
            "../sql",
            "../../sql",
            # Root of project
            "/src",
            # Docker volume mounts
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

        # If not found, try individual files
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

            # Split by semicolons and execute each statement
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
                            # Some statements might fail if objects already exist, that's OK
                            if "already exists" in str(e).lower():
                                print(f"\tStatement {i+1} - object already exists (OK)")
                            else:
                                print(f"\tStatement {i+1} failed: {str(e)}")
                                # Continue with other statements

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
            print("Searched in the following locations:")
            for path in [
                ".",
                "sql",
                "/src/sql",
                "../sql",
                "../../sql",
                "/src",
                "/app/sql",
                "/opt/app/sql",
            ]:
                print(f"  - {path}")
            return False

        # Execute DICOM schema first (has the base tables)
        if not self._execute_sql_file(sql_files["dicom"], "DICOM schema"):
            return False

        # Then execute analysis schema (has foreign keys to DICOM tables)
        if not self._execute_sql_file(sql_files["analysis"], "Analysis schema"):
            return False

        print("Database schema created successfully from SQL files")
        return True

    def _image_exists(self, series_instance_uid):
        """
        Check if the processed image exists in MinIO.

        Args:
            series_instance_uid (str): The unique identifier for the DICOM series.

        Returns:
            bool: True if the image exists, False otherwise.
        """
        if not self.minio_client:
            print("MinIO client not initialized")
            return False

        # Check if image already exists
        object_name = f"{series_instance_uid}_coronal_view.png"
        try:
            self.minio_client.stat_object(
                bucket_name=self.bucket_name, object_name=object_name
            )
            print(f"Image already exists in MinIO: {object_name}")
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                print(f"Image does not exist in MinIO: {object_name}")
            else:
                print(f"Error checking image existence in MinIO: {e}")
            return False

    def _delete_image_from_minio(self, series_instance_uid):
        """Delete image data from MinIO"""
        if not self.minio_client:
            print("MinIO client not initialized")
            return False

        # Generate object name
        object_name = f"{series_instance_uid}_coronal_view.png"
        try:
            self.minio_client.remove_object(
                bucket_name=self.bucket_name, object_name=object_name
            )
            print(f"Successfully deleted image from MinIO: {object_name}")
            return True
        except S3Error as e:
            raise e
        except Exception as e:
            raise e

    def _save_image_to_minio(self, image_data, series_instance_uid):
        """Save image data to MinIO with proper error handling"""
        if not self.minio_client:
            print("MinIO client not initialized")
            return False

        if self._image_exists(series_instance_uid):
            try:
                self._delete_image_from_minio(series_instance_uid)
            except S3Error as e:
                print(f"S3Error removing image from MinIO: {e}")
                print(traceback.format_exc())
                return False
            except Exception as e:
                print(f"Error removing image from MinIO: {str(e)}")
                print(traceback.format_exc())
                return False

        try:
            # Handle PIL Image objects
            if hasattr(image_data, "save"):
                # PIL Image - convert to bytes
                img_buffer = io.BytesIO()
                image_data.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()
                img_size = len(img_bytes)
                img_stream = io.BytesIO(img_bytes)
            else:
                # Assume it's already bytes or similar
                if isinstance(image_data, bytes):
                    img_bytes = image_data
                    img_size = len(img_bytes)
                    img_stream = io.BytesIO(img_bytes)
                else:
                    print(f"Unsupported image data type: {type(image_data)}")
                    return False

            # Generate object name
            object_name = f"{series_instance_uid}_coronal_view.png"

            # Upload to MinIO
            result = self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=img_stream,
                length=img_size,
                content_type="image/png",
            )

            print(f"Successfully uploaded image to MinIO: {object_name}")
            return True
        except S3Error as e:
            print(f"S3Error saving image to MinIO: {e}")
            print(traceback.format_exc())
            return False
        except Exception as e:
            print(f"Error saving image to MinIO: {str(e)}")
            print(traceback.format_exc())
            return False

    def _get_or_create_patient(self, cursor, patient_info):
        """Get or create patient record"""
        # Check if patient exists
        cursor.execute(
            """
            SELECT id FROM dicom.patient 
            WHERE patient_id = %s
        """,
            (patient_info.get("patient_id"),),
        )

        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new patient
        cursor.execute(
            """
            INSERT INTO dicom.patient (patient_id, name, birth_date, sex, weight_kg)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                patient_info.get("patient_id"),
                patient_info.get("name"),
                patient_info.get("birth_date"),
                patient_info.get("sex"),
                patient_info.get("weight_kg"),
            ),
        )

        return cursor.fetchone()[0]

    def _get_or_create_scanner(self, cursor, scanner_info):
        """Get or create scanner record"""
        # Check if scanner exists
        cursor.execute(
            """
            SELECT id FROM dicom.scanner 
            WHERE manufacturer = %s AND model_name = %s 
            AND device_serial_number = %s AND station_name = %s
        """,
            (
                scanner_info.get("manufacturer"),
                scanner_info.get("model_name"),
                scanner_info.get("device_serial_number"),
                scanner_info.get("station_name"),
            ),
        )

        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new scanner
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
        # Check if study exists
        cursor.execute(
            """
            SELECT id FROM dicom.study 
            WHERE study_instance_uid = %s
        """,
            (study_info.get("study_instance_uid"),),
        )

        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new study
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
        # Check if series exists
        cursor.execute(
            """
            SELECT id FROM dicom.series 
            WHERE series_instance_uid = %s
        """,
            (series_info.get("series_instance_uid"),),
        )

        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new series
        cursor.execute(
            """
            INSERT INTO dicom.series (
                uuid, study_id_fk, series_instance_uid, series_number,
                description, modality, body_part_examined, protocol_name,
                convolution_kernel, patient_position, series_date, series_time,
                frame_of_reference_uid, image_type, slice_thickness_mm,
                pixel_spacing_mm, rows, columns, scanner_id_fk,
                image_count, scan_length_cm
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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

    def _save_ct_technique(self, cursor, ct_info, series_id):
        """Save CT technique information"""
        cursor.execute(
            """
            INSERT INTO dicom.ct_technique (
                series_id_fk, kvp, exposure_time_ms, generator_power_kw,
                focal_spots_mm, filter_type, data_collection_diam_mm,
                recon_diameter_mm, dist_src_detector_mm, dist_src_patient_mm,
                gantry_detector_tilt_deg, single_collimation_width_mm,
                total_collimation_width_mm, table_speed_mm_s,
                table_feed_per_rot_mm, spiral_pitch_factor, exposure_modulation_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (series_id_fk) DO UPDATE SET
                kvp = EXCLUDED.kvp,
                exposure_time_ms = EXCLUDED.exposure_time_ms,
                generator_power_kw = EXCLUDED.generator_power_kw,
                focal_spots_mm = EXCLUDED.focal_spots_mm,
                filter_type = EXCLUDED.filter_type,
                data_collection_diam_mm = EXCLUDED.data_collection_diam_mm,
                recon_diameter_mm = EXCLUDED.recon_diameter_mm,
                dist_src_detector_mm = EXCLUDED.dist_src_detector_mm,
                dist_src_patient_mm = EXCLUDED.dist_src_patient_mm,
                gantry_detector_tilt_deg = EXCLUDED.gantry_detector_tilt_deg,
                single_collimation_width_mm = EXCLUDED.single_collimation_width_mm,
                total_collimation_width_mm = EXCLUDED.total_collimation_width_mm,
                table_speed_mm_s = EXCLUDED.table_speed_mm_s,
                table_feed_per_rot_mm = EXCLUDED.table_feed_per_rot_mm,
                spiral_pitch_factor = EXCLUDED.spiral_pitch_factor,
                exposure_modulation_type = EXCLUDED.exposure_modulation_type
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

    def init_minio_client(self):
        """Initialize MinIO client with better error handling"""
        try:
            self.minio_client = Minio(
                endpoint=self.minio_config["endpoint"],
                access_key=self.minio_config["access_key"],
                secret_key=self.minio_config["secret_key"],
                secure=self.minio_config["secure"],
            )

            # Test connection
            try:
                self.minio_client.list_buckets()
                print("Successfully connected to MinIO")
            except Exception as e:
                print(f"Failed to connect to MinIO: {str(e)}")
                return False

            # Create bucket if it doesn't exist
            try:
                if not self.minio_client.bucket_exists(bucket_name=self.bucket_name):
                    self.minio_client.make_bucket(bucket_name=self.bucket_name)
                    print(f"Created MinIO bucket: {self.bucket_name}")
                else:
                    print(f"MinIO bucket '{self.bucket_name}' already exists")
            except S3Error as e:
                print(f"Failed to create/check MinIO bucket: {str(e)}")
                return False

        except Exception as e:
            print(f"Failed to initialize MinIO client: {str(e)}")
            print(traceback.format_exc())
            return False

        return True

    def init_postgres_database(self):
        """Initialize the database - create schema from SQL files if it doesn't exist"""
        if not self.connect_postgres() or self.postgres_connection is None:
            print("Failed to connect to PostgreSQL")
            return False

        try:
            # Check if database already exists
            if self._check_database_exists():
                print("Database schema already exists")
            else:
                # Create database from SQL files
                if not self._create_database_from_sql_files():
                    print("Failed to create database schema from SQL files")
                    return False

            # Create additional performance indices
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

        return True

    def save_results(self, patient, study, scanner, series, ct, results):
        """Save CHO calculation results to the database with new schema"""
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        series_id = None
        success = False

        try:
            with self.postgres_connection.cursor() as cursor:
                # Create or get all the related records
                patient_id = self._get_or_create_patient(cursor, patient)
                scanner_id = self._get_or_create_scanner(cursor, scanner)
                study_id = self._get_or_create_study(cursor, study, patient_id)
                series_id = self._get_or_create_series(
                    cursor, series, study_id, scanner_id
                )

                # Save CT technique information
                if ct:
                    self._save_ct_technique(cursor, ct, series_id)

                # Save analysis results - Update results if series_id_fk already exists
                cursor.execute(
                    """
                    INSERT INTO analysis.results (
                        series_id_fk, average_frequency, average_index_of_detectability,
                        average_noise_level, cho_detectability, ctdivol, ctdivol_avg,
                        dlp, dlp_ssde, dw, dw_avg, location, location_sparse,
                        noise_level, nps, peak_frequency, percent_10_frequency,
                        processing_time, spatial_frequency, spatial_resolution, ssde, ssde_inc
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (series_id_fk) DO UPDATE SET
                        average_frequency = COALESCE(EXCLUDED.average_frequency, analysis.results.average_frequency),
                        average_index_of_detectability = COALESCE(EXCLUDED.average_index_of_detectability, analysis.results.average_index_of_detectability),
                        average_noise_level = COALESCE(EXCLUDED.average_noise_level, analysis.results.average_noise_level),
                        cho_detectability = COALESCE(EXCLUDED.cho_detectability, analysis.results.cho_detectability),
                        ctdivol = COALESCE(EXCLUDED.ctdivol, analysis.results.ctdivol),
                        ctdivol_avg = COALESCE(EXCLUDED.ctdivol_avg, analysis.results.ctdivol_avg),
                        dlp = COALESCE(EXCLUDED.dlp, analysis.results.dlp),
                        dlp_ssde = COALESCE(EXCLUDED.dlp_ssde, analysis.results.dlp_ssde),
                        dw = COALESCE(EXCLUDED.dw, analysis.results.dw),
                        dw_avg = COALESCE(EXCLUDED.dw_avg, analysis.results.dw_avg),
                        location = COALESCE(EXCLUDED.location, analysis.results.location),
                        location_sparse = COALESCE(EXCLUDED.location_sparse, analysis.results.location_sparse),
                        noise_level = COALESCE(EXCLUDED.noise_level, analysis.results.noise_level),
                        nps = COALESCE(EXCLUDED.nps, analysis.results.nps),
                        peak_frequency = COALESCE(EXCLUDED.peak_frequency, analysis.results.peak_frequency),
                        percent_10_frequency = COALESCE(EXCLUDED.percent_10_frequency, analysis.results.percent_10_frequency),
                        processing_time = EXCLUDED.processing_time,
                        spatial_frequency = COALESCE(EXCLUDED.spatial_frequency, analysis.results.spatial_frequency),
                        spatial_resolution = COALESCE(EXCLUDED.spatial_resolution, analysis.results.spatial_resolution),
                        ssde = COALESCE(EXCLUDED.ssde, analysis.results.ssde),
                        ssde_inc = COALESCE(EXCLUDED.ssde_inc, analysis.results.ssde_inc)
                    RETURNING id
                """,
                    (
                        series_id,
                        results.get("average_frequency"),
                        results.get("average_index_of_detectability"),
                        results.get("average_noise_level"),
                        results.get("cho_detectability"),
                        results.get("ctdivol"),
                        results.get("ctdivol_avg"),
                        results.get("dlp"),
                        results.get("dlp_ssde"),
                        results.get("dw"),
                        results.get("dw_avg"),
                        results.get("location"),
                        results.get("location_sparse"),
                        results.get("noise_level"),
                        results.get("nps"),
                        results.get("peak_frequency"),
                        results.get("percent_10_frequency"),
                        results.get("processing_time"),
                        results.get("spatial_frequency"),
                        results.get("spatial_resolution"),
                        results.get("ssde"),
                        results.get("ssde_inc"),
                    ),
                )

                self.postgres_connection.commit()
                print(
                    f"Successfully saved CHO results for series {series.get('series_instance_uid')}"
                )
                success = True

        except Exception as e:
            print(f"Failed to save results to PostgreSQL: {str(e)}")
            print(traceback.format_exc())
            if self.postgres_connection:
                self.postgres_connection.rollback()
            return False

        # Save coronal view image to MinIO (separate from database transaction)
        if success and results.get("coronal_view") is not None:
            # Check if image already exists

            image_success = self._save_image_to_minio(
                self.create_coronal_image(results.get("coronal_view")),
                series.get("series_instance_uid"),
            )
            if not image_success:
                print(
                    f"Warning: Failed to save coronal view image for series {series.get('series_instance_uid')}"
                )
                # Don't fail the entire operation just because image save failed

        return success

    def create_coronal_image(self, coronal_view_data):
        """Convert coronal view data to PIL Image with proper format"""
        coronal_view_data = np.array(coronal_view_data)
        try:
            # Ensure we have valid image data
            if coronal_view_data is None or len(coronal_view_data.shape) != 2:
                print("Warning: Invalid coronal view data")
                return None

            # Normalize the data to 0-255 range
            min_val = np.min(coronal_view_data)
            max_val = np.max(coronal_view_data)

            if max_val > min_val:
                normalized = (
                    (coronal_view_data - min_val) / (max_val - min_val) * 255
                ).astype(np.uint8)
            else:
                normalized = np.zeros_like(coronal_view_data, dtype=np.uint8)

            # Create PIL Image
            image = Image.fromarray(normalized, mode="L")  # 'L' for grayscale

            # Optionally resize if too large
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            print(f"Error creating coronal image: {str(e)}")
            return None

    def get_results(self, params):
        """Retrieve CHO results from the database"""
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        try:
            with self.postgres_connection.cursor() as cursor:
                # Get query parameters for enhanced filtering
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

                # Pagination parameters
                page = int(params.get("page", "1"))
                limit = int(params.get("limit", "25"))

                # Validate pagination parameters
                page = max(1, page)
                limit = min(max(1, limit), 1000)  # Limit max results per page to 1000
                offset = (page - 1) * limit

                # Base query with enhanced joins
                base_query = """
                    FROM dicom.series s
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    JOIN dicom.scanner sc ON s.scanner_id_fk = sc.id
                    LEFT JOIN analysis.results r ON s.id = r.series_id_fk
                """

                params = []
                conditions = []

                # Apply filters
                if patient_id:
                    conditions.append("p.patient_id ILIKE %s")
                    params.append(f"%{patient_id}%")

                if study_id:
                    conditions.append("st.study_instance_uid = %s")
                    params.append(study_id)

                if patient_search:
                    conditions.append("(p.name ILIKE %s OR p.patient_id ILIKE %s)")
                    params.extend([f"%{patient_search}%", f"%{patient_search}%"])

                if institute:
                    conditions.append("st.institution_name ILIKE %s")
                    params.append(f"%{institute}%")

                if station_name:
                    conditions.append("sc.station_name ILIKE %s")
                    params.append(f"%{station_name}%")

                if protocol_name:
                    conditions.append("s.protocol_name ILIKE %s")
                    params.append(f"%{protocol_name}%")

                if scanner_model:
                    conditions.append("sc.model_name ILIKE %s")
                    params.append(f"%{scanner_model}%")

                if exam_date_from:
                    conditions.append("st.study_date >= %s")
                    params.append(exam_date_from)

                if exam_date_to:
                    conditions.append("st.study_date <= %s")
                    params.append(exam_date_to)

                # Patient age calculation (requires birth_date and study_date)
                if patient_age_min or patient_age_max:
                    if patient_age_min:
                        conditions.append(
                            "EXTRACT(YEAR FROM AGE(st.study_date::date, p.birth_date::date)) >= %s"
                        )
                        params.append(int(patient_age_min))
                    if patient_age_max:
                        conditions.append(
                            "EXTRACT(YEAR FROM AGE(st.study_date::date, p.birth_date::date)) <= %s"
                        )
                        params.append(int(patient_age_max))

                where_clause = ""
                if conditions:
                    where_clause = " WHERE " + " AND ".join(conditions)

                # Add filter to only show series that have results
                having_clause = " HAVING COUNT(r.id) > 0"

                # Count total results for pagination
                count_query = f"""
                    SELECT COUNT(DISTINCT s.id)
                    {base_query}
                    {where_clause}
                    GROUP BY s.id
                    {having_clause}
                """

                # Get total count using a subquery
                total_query = f"SELECT COUNT(*) FROM ({count_query}) as counted"
                cursor.execute(total_query, params)
                total_results = cursor.fetchone()[0]

                # Calculate pagination info
                total_pages = (
                    (total_results + limit - 1) // limit if total_results > 0 else 1
                )

                # Main data query with pagination
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

                        -- test_status now uses the same filters as your global_noise_count
                        -- and detectability_count from the second query:
                    CASE
                    -- any row in this series has *all* metrics ⇒ “full”
                    WHEN COUNT(r.id) FILTER (
                            WHERE 
                            (ctdivol, ctdivol_avg, dlp, dlp_ssde, dw, dw_avg, "location", ssde) IS NOT NULL
                        AND (average_frequency, average_index_of_detectability, average_noise_level,
                                cho_detectability, location_sparse, noise_level, nps, peak_frequency,
                                percent_10_frequency, spatial_frequency, spatial_resolution) IS NOT NULL
                        ) > 0
                        THEN 'full'

                    -- else, any row has only the “global noise” metrics ⇒ “partial”
                    WHEN COUNT(r.id) FILTER (
                            WHERE 
                            (ctdivol, ctdivol_avg, dlp, dlp_ssde, dw, dw_avg, "location", ssde) IS NOT NULL
                        AND (average_frequency, average_index_of_detectability, average_noise_level,
                                cho_detectability, location_sparse, noise_level, nps, peak_frequency,
                                percent_10_frequency, spatial_frequency, spatial_resolution) IS NULL
                        ) > 0
                        THEN 'partial'

                    -- otherwise there were rows, but neither filter matched ⇒ “error”
                    ELSE 'error'
                    END AS test_status,
                        MAX(r.created_at) AS latest_analysis_date
                        
                    {base_query}
                    {where_clause}
                    GROUP BY s.id, s.series_instance_uid, s.uuid, p.patient_id, p.name, p.birth_date,
                            st.study_instance_uid, st.institution_name, st.study_date,
                            sc.manufacturer, sc.model_name, sc.station_name, s.protocol_name
                    {having_clause}
                    ORDER BY MAX(r.created_at) DESC NULLS LAST 
                    LIMIT %s OFFSET %s
                """
                params.extend([limit, offset])

                cursor.execute(query, params)
                results = cursor.fetchall()

                # Convert to list of dictionaries
                result_list = []
                if cursor.description is not None:
                    columns = [desc[0] for desc in cursor.description]
                    for row in results:
                        result_dict = dict(zip(columns, row))
                        # Convert datetime fields to ISO format
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
                                    # Handle date strings
                                    try:
                                        dt = datetime.fromisoformat(
                                            result_dict[date_field].replace(
                                                "Z", "+00:00"
                                            )
                                        )
                                        result_dict[date_field] = dt.isoformat()
                                    except:
                                        pass  # Keep original if conversion fails
                        result_list.append(result_dict)

            # Return paginated response
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
        """Get filter options"""
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        try:
            with self.postgres_connection.cursor() as cursor:
                # Get unique values for filter dropdowns - updated for new schema
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
                else:
                    filter_options["date_range"] = {"min": None, "max": None}

                # Age range (calculated from birth dates and study dates)
                cursor.execute(
                    """
                    SELECT 
                        MIN(EXTRACT(YEAR FROM AGE(st.study_date::date, p.birth_date::date))) as min_age,
                        MAX(EXTRACT(YEAR FROM AGE(st.study_date::date, p.birth_date::date))) as max_age
                    FROM dicom.patient p
                    JOIN dicom.study st ON p.id = st.patient_id_fk
                    WHERE p.birth_date IS NOT NULL AND st.study_date IS NOT NULL
                """
                )
                age_range = cursor.fetchone()
                if age_range and age_range[0] is not None and age_range[1] is not None:
                    filter_options["age_range"] = {
                        "min": int(age_range[0]),
                        "max": int(age_range[1]),
                    }
                else:
                    filter_options["age_range"] = {"min": 0, "max": 100}

            return filter_options
        except Exception as e:
            self.postgres_connection.rollback()
            raise e

    def delete_results(self, series_instance_uid):
        """Delete CHO results for a specific series"""
        if not self.connect_postgres() or self.postgres_connection is None:
            return False

        try:
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM analysis.results r
                    USING dicom.series s
                    WHERE r.series_id_fk = s.id 
                    AND s.series_instance_uid = %s
                    RETURNING r.id
                """,
                    (series_instance_uid,),
                )

                deleted_rows = cursor.fetchall()

                if not deleted_rows:
                    print(f"No results found for series {series_instance_uid}")
                    return False

                # Delete the image in the minio
                if not self._delete_image_from_minio(series_instance_uid):
                    print(f"Failed to delete image {series_instance_uid} from MinIO")
                    return False

                self.postgres_connection.commit()
                print(
                    f"Successfully deleted CHO results for series {series_instance_uid}"
                )
                return True

        except Exception as e:
            self.postgres_connection.rollback()
            raise e

    def export_results(self):
        """Export CHO results for a specific series"""
        if not self.connect_postgres() or self.postgres_connection is None:
            raise Exception("No database connection")

        try:
            with self.postgres_connection.cursor() as cursor:
                # Query for CSV export with flattened data - updated for new schema
                cursor.execute(
                    """
                    SELECT 
                        s.series_instance_uid as series_id,
                        p.patient_id,
                        p.name as patient_name,
                        st.study_instance_uid as study_id,
                        CASE 
                            WHEN r.average_index_of_detectability IS NOT NULL THEN 'Full Analysis'
                            ELSE 'Global Noise'
                        END as analysis_type,
                        r.ctdivol_avg,
                        r.ssde,
                        r.dlp,
                        r.dlp_ssde,
                        r.dw_avg,
                        r.spatial_resolution,
                        r.average_noise_level,
                        r.peak_frequency,
                        r.average_frequency,
                        r.percent_10_frequency,
                        r.average_index_of_detectability,
                        r.created_at
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    ORDER BY r.created_at DESC
                """
                )

                results = cursor.fetchall()

                # Create CSV content
                csv_header = "Series ID,Patient ID,Patient Name,Study ID,Analysis Type,CTDI Avg,SSDE,DLP,DLP SSDE,DW Avg,Spatial Resolution,Average Noise,Peak Frequency,Average Frequency,10% Frequency,Detectability Index,Created At\n"

                csv_content = csv_header
                for row in results:
                    # Convert datetime to string and handle None values
                    row_data = []
                    for item in row:
                        if item is None:
                            row_data.append("")
                        elif isinstance(item, datetime):
                            row_data.append(item.isoformat())
                        else:
                            row_data.append(str(item))
                    csv_content += ",".join(row_data) + "\n"
                return csv_content

        except Exception as e:
            self.postgres_connection.rollback()
            raise e

    def get_results_statistics(self):
        """Export CHO results for a specific series"""
        if not self.connect_postgres() or self.postgres_connection is None:
            raise Exception("No database connection")

        try:
            with self.postgres_connection.cursor() as cursor:

                # Count the number of results
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

    def get_result(self, series_instance_uid):
        if not self.connect_postgres() or self.postgres_connection is None:
            raise Exception("No database connection")

        try:
            with self.postgres_connection.cursor() as cursor:
                # Updated query for new schema
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

    def save_dicom_headers_only(self, patient, study, scanner, series, ct):
        """Save DICOM metadata without analysis results"""
        return self.save_results(
            patient, study, scanner, series, ct, results={"processing_time": 0.0}
        )
        # if not self.connect_postgres() or self.postgres_connection is None:
        #     return False

        # try:
        #     with self.postgres_connection.cursor() as cursor:
        #         # Insert/get patient
        #         patient_id = self._get_or_create_patient(cursor, patient)
        #         if patient_id is None:
        #             return False

        #         # Insert/get scanner
        #         scanner_id = self._get_or_create_scanner(cursor, scanner)
        #         if scanner_id is None:
        #             return False

        #         # Insert/get study
        #         study_id = self._get_or_create_study(cursor, study, patient_id)
        #         if study_id is None:
        #             return False

        #         # Insert/get series
        #         series_id = self._get_or_create_series(cursor, series, study_id, scanner_id)
        #         if series_id is None:
        #             return False

        #         # Insert/get CT technique
        #         ct_id = self._save_ct_technique(cursor, ct, series_id)
        #         if ct_id is None:
        #             return False

        #         # Insert headers-only record in analysis.results
        #         cursor.execute("""
        #             INSERT INTO analysis.results (
        #                 series_id_fk, average_frequency, average_index_of_detectability,
        #                 average_noise_level, cho_detectability, ctdivol, ctdivol_avg,
        #                 dlp, dlp_ssde, dw, dw_avg, location, location_sparse,
        #                 noise_level, nps, peak_frequency, percent_10_frequency,
        #                 processing_time, spatial_frequency, spatial_resolution, ssde
        #             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        #             ON CONFLICT (series_id_fk) DO UPDATE SET
        #                 average_frequency = COALESCE(EXCLUDED.average_frequency, analysis.results.average_frequency),
        #                 average_index_of_detectability = COALESCE(EXCLUDED.average_index_of_detectability, analysis.results.average_index_of_detectability),
        #                 average_noise_level = COALESCE(EXCLUDED.average_noise_level, analysis.results.average_noise_level),
        #                 cho_detectability = COALESCE(EXCLUDED.cho_detectability, analysis.results.cho_detectability),
        #                 ctdivol = COALESCE(EXCLUDED.ctdivol, analysis.results.ctdivol),
        #                 ctdivol_avg = COALESCE(EXCLUDED.ctdivol_avg, analysis.results.ctdivol_avg),
        #                 dlp = COALESCE(EXCLUDED.dlp, analysis.results.dlp),
        #                 dlp_ssde = COALESCE(EXCLUDED.dlp_ssde, analysis.results.dlp_ssde),
        #                 dw = COALESCE(EXCLUDED.dw, analysis.results.dw),
        #                 dw_avg = COALESCE(EXCLUDED.dw_avg, analysis.results.dw_avg),
        #                 location = COALESCE(EXCLUDED.location, analysis.results.location),
        #                 location_sparse = COALESCE(EXCLUDED.location_sparse, analysis.results.location_sparse),
        #                 noise_level = COALESCE(EXCLUDED.noise_level, analysis.results.noise_level),
        #                 nps = COALESCE(EXCLUDED.nps, analysis.results.nps),
        #                 peak_frequency = COALESCE(EXCLUDED.peak_frequency, analysis.results.peak_frequency),
        #                 percent_10_frequency = COALESCE(EXCLUDED.percent_10_frequency, analysis.results.percent_10_frequency),
        #                 processing_time = EXCLUDED.processing_time,
        #                 spatial_frequency = COALESCE(EXCLUDED.spatial_frequency, analysis.results.spatial_frequency),
        #                 spatial_resolution = COALESCE(EXCLUDED.spatial_resolution, analysis.results.spatial_resolution),
        #                 ssde = COALESCE(EXCLUDED.ssde, analysis.results.ssde)
        #             RETURNING id
        #         """, (
        #             series_id,
        #            None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             0.0,
        #             None,
        #             None,
        #             None
        #         ))
        #         # cursor.execute("""
        #         #     INSERT INTO analysis.results (
        #         #         series_id_fk, ct_technique_id_fk, test_status,
        #         #         analysis_timestamp, processing_time_seconds
        #         #     ) VALUES (%s, %s, %s, %s, %s)
        #         #     ON CONFLICT (series_id_fk)
        #         #     DO UPDATE SET
        #         #         test_status = EXCLUDED.test_status,
        #         #         analysis_timestamp = EXCLUDED.analysis_timestamp,
        #         #         processing_time_seconds = EXCLUDED.processing_time_seconds
        #         #     RETURNING id
        #         # """, (
        #         #     series_id, ct_id, 'headers_only',
        #         #     datetime.now(), 0.0
        #         # ))

        #         # Commit the transaction
        #         self.postgres_connection.commit()
        #         print(f"DICOM headers saved for series {series.get('series_instance_uid')}")
        #         return True

        # except Exception as e:
        #     print(f"Error saving DICOM headers: {str(e)}")
        #     if self.postgres_connection:
        #         self.postgres_connection.rollback()
        #     import traceback
        #     traceback.print_exc()
        #     return False


# Global instance
cho_storage = CHOResultsStorage()
