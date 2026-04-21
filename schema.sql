CREATE TABLE IF NOT EXISTS company_research (
    company_number            TEXT PRIMARY KEY,
    company_name              TEXT,
    postcode                  TEXT,
    domain                    TEXT,
    domain_confidence         TEXT,
    domain_method             TEXT,
    website_raw_text_length   INTEGER,
    services_description      TEXT,
    specialisms               TEXT,
    sector                    TEXT,
    sub_sector                TEXT,
    classification_confidence TEXT,
    rationale                 TEXT,
    keywords                  TEXT,
    sampled_for_discovery     INTEGER DEFAULT 0,
    taxonomy_version_id       INTEGER,
    processed_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_log                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_sector    ON company_research(sector);
CREATE INDEX IF NOT EXISTS idx_domain    ON company_research(domain);
CREATE INDEX IF NOT EXISTS idx_conf      ON company_research(classification_confidence);
CREATE INDEX IF NOT EXISTS idx_tax_ver   ON company_research(taxonomy_version_id);

CREATE TABLE IF NOT EXISTS taxonomy_versions (
    version_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    taxonomy_json      TEXT,
    company_count_used INTEGER,
    refresh_trigger    TEXT,
    change_log         TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes              TEXT
);
