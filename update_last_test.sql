UPDATE data_source_configs SET config_data = jsonb_set(config_data, '{akshare_stock_a,last_test}', '"2026-03-10 12:00:00"') WHERE config_key = 'data_sources_production';
