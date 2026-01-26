-- DROP SCHEMA analysis;

CREATE SCHEMA analysis AUTHORIZATION postgres;

-- DROP SEQUENCE analysis.results_id_seq;

CREATE SEQUENCE analysis.results_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE analysis.results_id_seq1;

CREATE SEQUENCE analysis.results_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;-- analysis.results definition

-- Drop table

-- DROP TABLE analysis.results;

CREATE TABLE analysis.results (
	id bigserial NOT NULL,
	series_id_fk int8 NOT NULL,
	average_frequency float8 NULL,
	average_index_of_detectability float8 NULL,
	average_noise_level float8 NULL,
	cho_detectability _float8 NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	ctdivol _float8 NULL,
	ctdivol_avg float8 NULL,
	dlp float8 NULL,
	dlp_ssde float8 NULL,
	dw _float8 NULL,
	dw_avg float8 NULL,
	"location" _float8 NULL,
	location_sparse _float8 NULL,
	noise_level _float8 NULL,
	nps _float8 NULL,
	peak_frequency float8 NULL,
	percent_10_frequency float8 NULL,
	processing_time float8 NULL,
	spatial_frequency _float8 NULL,
	spatial_resolution float8 NULL,
	ssde float8 NULL,
	ssde_inc _float8 NULL,
	CONSTRAINT results_pkey PRIMARY KEY (id),
	CONSTRAINT results_series_id_fk_unique UNIQUE (series_id_fk)
);
CREATE INDEX idx_analysis_results_created_at ON analysis.results USING btree (created_at);
CREATE INDEX idx_analysis_results_series_fk ON analysis.results USING btree (series_id_fk);
CREATE INDEX results_series_fk_idx ON analysis.results USING btree (series_id_fk);


-- analysis.results foreign keys

ALTER TABLE analysis.results ADD CONSTRAINT results_series_id_fk_fkey FOREIGN KEY (series_id_fk) REFERENCES dicom.series(id) ON DELETE CASCADE;