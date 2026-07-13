CREATE TABLE IF NOT EXISTS regions (
    region_id SERIAL PRIMARY KEY,
    region_name TEXT NOT NULL UNIQUE,
    revenue_2025_cny_million NUMERIC(18, 2),
    revenue_2024_cny_million NUMERIC(18, 2),
    yoy_growth_percentage NUMERIC(8, 2)
);

CREATE TABLE IF NOT EXISTS business_segments (
    segment_id SERIAL PRIMARY KEY,
    segment_name TEXT NOT NULL UNIQUE,
    revenue_2025_cny_million NUMERIC(18, 2),
    revenue_2024_cny_million NUMERIC(18, 2),
    yoy_growth_percentage NUMERIC(8, 2)
);

CREATE TABLE IF NOT EXISTS financial_highlights (
    year INT PRIMARY KEY,
    revenue_cny_million NUMERIC(18, 2),
    operating_profit_cny_million NUMERIC(18, 2),
    operating_margin_percentage NUMERIC(8, 2),
    net_profit_cny_million NUMERIC(18, 2),
    cash_flow_operating_cny_million NUMERIC(18, 2),
    total_assets_cny_million NUMERIC(18, 2),
    equity_cny_million NUMERIC(18, 2),
    liability_ratio_percentage NUMERIC(8, 2)
);

CREATE TABLE IF NOT EXISTS tech_ecosystems (
    ecosystem_id SERIAL PRIMARY KEY,
    platform_name TEXT NOT NULL UNIQUE,
    global_registered_developers INT,
    partner_count INT,
    compatible_solutions_incubated INT
);

CREATE TABLE IF NOT EXISTS operational_footprint (
    metric_id SERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL UNIQUE,
    global_count INT,
    countries_covered INT
);

CREATE TABLE IF NOT EXISTS segment_platform_mapping (
    segment_id INT NOT NULL REFERENCES business_segments(segment_id) ON DELETE CASCADE,
    ecosystem_id INT NOT NULL REFERENCES tech_ecosystems(ecosystem_id) ON DELETE CASCADE,
    PRIMARY KEY (segment_id, ecosystem_id)
);
