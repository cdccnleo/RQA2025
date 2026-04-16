UPDATE data_source_configs
SET config_data = (
    config_data ||
    jsonb_build_object(
        'data_sources',
        (
            SELECT jsonb_agg(
                CASE
                    WHEN (elem ->> 'id') = 'akshare_commodity_natural_gas'
                    THEN elem || '{"enabled": true}'::jsonb
                    ELSE elem
                END
            )
            FROM jsonb_array_elements(config_data -> 'data_sources') AS elem
        )
    )
),
updated_at = CURRENT_TIMESTAMP
WHERE config_key = 'data_sources_production'
AND config_data -> 'data_sources' @> '[{"id": "akshare_commodity_natural_gas"}]';
