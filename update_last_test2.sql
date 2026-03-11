UPDATE data_source_configs 
SET config_data = jsonb_set(
    config_data,
    '{data_sources,0,last_test}',
    '"2026-03-10 12:00:00"'
)
WHERE config_key = 'data_sources_production'
AND config_data->'data_sources'->0->>'id' = 'akshare_stock_a';
