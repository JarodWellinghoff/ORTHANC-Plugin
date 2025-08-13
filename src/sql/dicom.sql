-- DROP SCHEMA dicom;

CREATE SCHEMA dicom AUTHORIZATION postgres;

-- DROP SEQUENCE dicom.patient_id_seq;

CREATE SEQUENCE dicom.patient_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE dicom.scanner_id_seq;

CREATE SEQUENCE dicom.scanner_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE dicom.series_id_seq;

CREATE SEQUENCE dicom.series_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE dicom.study_id_seq;

CREATE SEQUENCE dicom.study_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;-- dicom.patient definition

-- Drop table

-- DROP TABLE dicom.patient;

CREATE TABLE dicom.patient (
	id bigserial NOT NULL,
	patient_id text NULL,
	"name" text NULL,
	birth_date date NULL,
	sex text NULL,
	weight_kg float8 NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT patient_pkey PRIMARY KEY (id)
);


-- dicom.scanner definition

-- Drop table

-- DROP TABLE dicom.scanner;

CREATE TABLE dicom.scanner (
	id bigserial NOT NULL,
	manufacturer text NULL,
	model_name text NULL,
	device_serial_number text NULL,
	software_versions text NULL,
	station_name text NULL,
	institution_name text NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT scanner_manufacturer_model_name_device_serial_number_statio_key UNIQUE (manufacturer, model_name, device_serial_number, station_name),
	CONSTRAINT scanner_pkey PRIMARY KEY (id)
);


-- dicom.study definition

-- Drop table

-- DROP TABLE dicom.study;

CREATE TABLE dicom.study (
	id bigserial NOT NULL,
	patient_id_fk int8 NULL,
	study_instance_uid text NOT NULL,
	study_id text NULL,
	accession_number text NULL,
	study_date date NULL,
	study_time time NULL,
	description text NULL,
	referring_physician text NULL,
	institution_name text NULL,
	institution_address text NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT study_pkey PRIMARY KEY (id),
	CONSTRAINT study_study_instance_uid_key UNIQUE (study_instance_uid),
	CONSTRAINT study_patient_id_fk_fkey FOREIGN KEY (patient_id_fk) REFERENCES dicom.patient(id) ON DELETE CASCADE
);
CREATE INDEX study_patient_fk_idx ON dicom.study USING btree (patient_id_fk);


-- dicom.series definition

-- Drop table

-- DROP TABLE dicom.series;

CREATE TABLE dicom.series (
	id bigserial NOT NULL,
	"uuid" text NULL,
	study_id_fk int8 NULL,
	series_instance_uid text NOT NULL,
	series_number int4 NULL,
	description text NULL,
	modality text NULL,
	body_part_examined text NULL,
	protocol_name text NULL,
	convolution_kernel text NULL,
	patient_position text NULL,
	series_date date NULL,
	series_time time NULL,
	frame_of_reference_uid text NULL,
	image_type _text NULL,
	slice_thickness_mm float8 NULL,
	pixel_spacing_mm _float8 NULL,
	"rows" int4 NULL,
	"columns" int4 NULL,
	scanner_id_fk int8 NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	image_count int4 NULL,
	scan_length_cm float8 NULL,
	CONSTRAINT pixel_spacing_len2 CHECK (((pixel_spacing_mm IS NULL) OR (array_length(pixel_spacing_mm, 1) = 2))),
	CONSTRAINT series_columns_check CHECK (((columns IS NULL) OR (columns > 0))),
	CONSTRAINT series_pkey PRIMARY KEY (id),
	CONSTRAINT series_rows_check CHECK (((rows IS NULL) OR (rows > 0))),
	CONSTRAINT series_series_instance_uid_key UNIQUE (series_instance_uid),
	CONSTRAINT series_slice_thickness_mm_check CHECK (((slice_thickness_mm IS NULL) OR (slice_thickness_mm > (0)::double precision))),
	CONSTRAINT series_uuid_key UNIQUE (uuid),
	CONSTRAINT series_scanner_id_fk_fkey FOREIGN KEY (scanner_id_fk) REFERENCES dicom.scanner(id) ON DELETE CASCADE,
	CONSTRAINT series_study_id_fk_fkey FOREIGN KEY (study_id_fk) REFERENCES dicom.study(id) ON DELETE CASCADE
);
CREATE INDEX series_mod_body_proto_idx ON dicom.series USING btree (modality, body_part_examined, protocol_name);
CREATE INDEX series_scanner_fk_idx ON dicom.series USING btree (scanner_id_fk);
CREATE INDEX series_study_fk_idx ON dicom.series USING btree (study_id_fk);


-- dicom.ct_technique definition

-- Drop table

-- DROP TABLE dicom.ct_technique;

CREATE TABLE dicom.ct_technique (
	series_id_fk int8 NOT NULL,
	kvp float8 NULL,
	exposure_time_ms int4 NULL,
	generator_power_kw float8 NULL,
	focal_spots_mm _float8 NULL,
	filter_type text NULL,
	data_collection_diam_mm float8 NULL,
	recon_diameter_mm float8 NULL,
	dist_src_detector_mm float8 NULL,
	dist_src_patient_mm float8 NULL,
	gantry_detector_tilt_deg float8 NULL,
	single_collimation_width_mm float8 NULL,
	total_collimation_width_mm float8 NULL,
	table_speed_mm_s float8 NULL,
	table_feed_per_rot_mm float8 NULL,
	spiral_pitch_factor float8 NULL,
	exposure_modulation_type text NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT ct_technique_pkey PRIMARY KEY (series_id_fk),
	CONSTRAINT ct_technique_series_id_fk_fkey FOREIGN KEY (series_id_fk) REFERENCES dicom.series(id) ON DELETE CASCADE
);