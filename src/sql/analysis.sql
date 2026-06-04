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


-- analysis.mtf definition
-- Stores MTF curve parameters per results row and lesion set.
-- Note: the original schema had a typo on mtf_lesion_set_id_fk_fkey,
-- pointing results_id_fk at lesion.set — corrected here to lesion_set_id_fk.

-- DROP TABLE analysis.mtf;

CREATE TABLE analysis.mtf (
    id              bigserial NOT NULL,
    results_id_fk   int8      NOT NULL,
    lesion_set_id_fk int8     NOT NULL,
    f_peak          float4    NOT NULL DEFAULT 0,
    eta_peak        float4    NOT NULL DEFAULT 1,
    f_50            float4    NOT NULL,
    f_10            float4    NULL,
    f_2             float4    NULL,
    created_at      timestamptz DEFAULT now() NOT NULL,
    CONSTRAINT mtf_pkey PRIMARY KEY (id),
    CONSTRAINT mtf_results_id_fk_unique    UNIQUE (results_id_fk),
    CONSTRAINT mtf_lesion_set_id_fk_unique UNIQUE (lesion_set_id_fk),
    CONSTRAINT mtf_f_peak_check   CHECK (f_peak   >= 0),
    CONSTRAINT mtf_eta_peak_check CHECK (eta_peak  > 0),
    CONSTRAINT mtf_f_50_check     CHECK (f_50      > 0)
);

CREATE INDEX idx_analysis_mtf_created_at   ON analysis.mtf USING btree (created_at);
CREATE INDEX idx_analysis_mtf_results_fk   ON analysis.mtf USING btree (results_id_fk);
CREATE INDEX idx_analysis_mtf_lesion_set_fk ON analysis.mtf USING btree (lesion_set_id_fk);


-- analysis.mtf foreign keys

ALTER TABLE analysis.mtf
    ADD CONSTRAINT mtf_results_id_fk_fkey
    FOREIGN KEY (results_id_fk) REFERENCES analysis.results(id) ON DELETE CASCADE;

-- FK to lesion.set is declared here for co-location with other analysis FKs.
-- Array-element integrity for lesion.set.lesion_ids is enforced via triggers in triggers.sql.
ALTER TABLE analysis.mtf
    ADD CONSTRAINT mtf_lesion_set_id_fk_fkey
    FOREIGN KEY (lesion_set_id_fk) REFERENCES lesion.set(id) ON DELETE CASCADE;