#!/usr/bin/env python3
"""Sync data source enabled status from health check results to PostgreSQL"""
import psycopg2
import json
import os
import asyncio
import sys
sys.path.insert(0, '/app')

def main():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'RQA2025_Postgres_Secure_2026'),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
        connect_timeout=10
    )
    cursor = conn.cursor()
    
    # Get current config from PostgreSQL
    config_key = 'data_sources_production'
    cursor.execute("""
        SELECT config_data FROM data_source_configs
        WHERE config_key = %s
        ORDER BY updated_at DESC LIMIT 1
    """, (config_key,))
    row = cursor.fetchone()
    if not row:
        print('No config found in PostgreSQL!')
        return
    
    # JSONB returns dict already
    config_data = row[0]
    sources = config_data if isinstance(config_data, list) else config_data.get('data_sources', [])
    print(f'Loaded {len(sources)} data sources from PostgreSQL')
    
    # Get health map
    from src.gateway.web.datasource_health_checker import get_health_checker
    hc = get_health_checker()
    
    async def get_h():
        return await hc.get_latest_health()
    
    health_list = asyncio.get_event_loop().run_until_complete(get_h())
    health_map = {h['source_id']: h for h in health_list}
    print(f'Got {len(health_map)} health statuses')
    
    # Update enabled and status based on health
    updated = 0
    for s in sources:
        sid = s.get('id', '')
        if sid in health_map:
            h = health_map[sid]
            health_status = h.get('status', '')
            # Enable if healthy
            new_enabled = (health_status == 'healthy')
            old_enabled = s.get('enabled', False)
            old_status = s.get('status', '')
            if old_enabled != new_enabled or old_status != health_status:
                s['enabled'] = new_enabled
                s['status'] = health_status
                print(f'Updated {sid}: enabled={old_enabled}->{new_enabled}, status={old_status}->{health_status}')
                updated += 1
    
    print(f'Total updated: {updated}')
    
    # Save back to PostgreSQL
    cursor.execute("""
        UPDATE data_source_configs
        SET config_data = %s, updated_at = NOW(), version = '1.1.1'
        WHERE config_key = %s
    """, (psycopg2.extras.Json(config_data), config_key))
    conn.commit()
    print(f'Saved updated config to PostgreSQL')
    
    cursor.close()
    conn.close()
    print('Done!')

if __name__ == '__main__':
    main()
