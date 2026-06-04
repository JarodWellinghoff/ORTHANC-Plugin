-- DROP SCHEMA lesion;

CREATE SCHEMA lesion AUTHORIZATION postgres;


-- lesion.lesion definition

-- DROP TABLE lesion.lesion;

CREATE TABLE lesion.lesion (
    id        bigserial NOT NULL,
    size      float4    NOT NULL,
    contrast  float4    NOT NULL,
    created_at timestamptz DEFAULT now() NOT NULL,
    CONSTRAINT lesion_pkey PRIMARY KEY (id),
    CONSTRAINT lesion_size_check    CHECK (size    > 0),
    CONSTRAINT lesion_contrast_check CHECK (contrast > 0)
);

CREATE INDEX idx_lesion_lesion_created_at ON lesion.lesion USING btree (created_at);


-- lesion.set definition
-- lesion_ids holds an array of lesion.lesion.id values.
-- Referential integrity is enforced via trigger (see triggers.sql)
-- because PostgreSQL does not support FK constraints on array elements.

-- DROP TABLE lesion.set;

CREATE TABLE lesion.set (
    id         bigserial NOT NULL,
    lesion_ids int8[]    NOT NULL,
    created_at timestamptz DEFAULT now() NOT NULL,
    CONSTRAINT lesion_set_pkey     PRIMARY KEY (id),
    CONSTRAINT lesion_set_nonempty CHECK (array_length(lesion_ids, 1) > 0)
);

CREATE INDEX idx_lesion_set_created_at ON lesion.set USING btree (created_at);