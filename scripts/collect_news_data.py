#!/usr/bin/env python3
"""
新闻数据采集脚本
从AkShare采集新闻数据并持久化到PostgreSQL
"""
import os, sys, time, logging
from datetime import datetime
sys.path.insert(0, '/app')
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_db():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432, user='rqa2025_admin',
        password=os.getenv('POSTGRES_PASSWORD', 'RQA2025_Postgres_Secure_2026'),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod')
    )


def create_tables():
    """创建新闻数据表"""
    conn = get_db()
    cur = conn.cursor()
    
    # 新闻主表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS akshare_news_data (
            id BIGSERIAL PRIMARY KEY,
            source_id VARCHAR(50) NOT NULL,
            news_source VARCHAR(50) NOT NULL,  -- 'cctv', 'eastmoney', 'shmet'
            title VARCHAR(500) NOT NULL,
            content TEXT,
            publish_date DATE,
            url VARCHAR(500),
            keywords VARCHAR(200),
            category VARCHAR(50),  -- 'macro', 'stock', 'futures', 'general'
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id, title, publish_date)
        )
    """)
    
    for idx in [
        'CREATE INDEX IF NOT EXISTS idx_news_source ON akshare_news_data(news_source, publish_date DESC)',
        'CREATE INDEX IF NOT EXISTS idx_news_date ON akshare_news_data(publish_date DESC)',
        'CREATE INDEX IF NOT EXISTS idx_news_category ON akshare_news_data(category, publish_date DESC)',
    ]:
        cur.execute(idx)
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info("News tables created OK")


def parse_date(date_val) -> str:
    if date_val is None:
        return None
    if hasattr(date_val, 'isoformat'):
        return date_val.strftime('%Y-%m-%d')
    s = str(date_val)
    # Handle formats like "20240424"
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def collect_news_cctv() -> list:
    """采集CCTV经济新闻"""
    import akshare as ak
    try:
        df = ak.news_cctv()
        records = []
        for _, row in df.iterrows():
            title = str(row.get('title', ''))[:500]
            if not title:
                continue
            records.append({
                'source_id': 'akshare_news_js',
                'news_source': 'cctv',
                'title': title,
                'content': str(row.get('content', ''))[:5000],
                'publish_date': parse_date(row.get('date')),
                'url': None,
                'keywords': None,
                'category': 'macro',
            })
        logger.info(f"news_cctv: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"news_cctv failed: {e}")
        return []


def collect_news_shmet() -> list:
    """采集上海金属期货新闻"""
    import akshare as ak
    try:
        df = ak.futures_news_shmet()
        records = []
        for _, row in df.iterrows():
            content = str(row.get('内容', row.get('content', '')))[:5000]
            # 从内容中提取标题（取前50个字符或【】内的内容）
            title = content[:80].replace('\n', ' ').strip()
            if not title:
                continue
            # 解析发布时间
            pub_date = row.get('发布时间', row.get('date'))
            pub_date_str = parse_date(pub_date)
            records.append({
                'source_id': 'akshare_news_js',
                'news_source': 'shmet',
                'title': title,
                'content': content,
                'publish_date': pub_date_str,
                'url': None,
                'keywords': None,
                'category': 'futures',
            })
        logger.info(f"futures_news_shmet: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"futures_news_shmet failed: {e}")
        return []


def collect_news_eastmoney() -> list:
    """采集东方财富个股新闻"""
    import akshare as ak
    try:
        # Get news for some key stocks
        df = ak.stock_news_em()
        records = []
        for _, row in df.iterrows():
            title = str(row.get('新闻标题', row.get('title', '')))[:500]
            if not title:
                continue
            records.append({
                'source_id': 'akshare_news_eastmoney',
                'news_source': 'eastmoney',
                'title': title,
                'content': str(row.get('新闻内容', row.get('content', '')))[:5000],
                'publish_date': parse_date(row.get('发布时间', row.get('date'))),
                'url': str(row.get('新闻链接', row.get('url', '')))[:500],
                'keywords': str(row.get('关键词', row.get('keyword', '')))[:200],
                'category': 'stock',
            })
        logger.info(f"stock_news_em: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"stock_news_em failed: {e}")
        return []


def persist_news(records: list) -> int:
    if not records:
        return 0
    conn = get_db()
    cur = conn.cursor()
    n = 0
    for r in records:
        try:
            cur.execute("""
                INSERT INTO akshare_news_data 
                (source_id, news_source, title, content, publish_date, url, keywords, category)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (source_id, title, publish_date) DO UPDATE SET
                    content = EXCLUDED.content,
                    url = COALESCE(EXCLUDED.url, akshare_news_data.url)
            """, (r['source_id'], r['news_source'], r['title'], r['content'],
                  r['publish_date'], r['url'], r['keywords'], r['category']))
            n += 1
        except Exception as e:
            logger.error(f"Insert failed: {e}")
    conn.commit()
    cur.close()
    conn.close()
    return n


def main():
    create_tables()
    logger.info("=== Starting News Collection ===")
    total = 0
    
    collectors = [
        ('CCTV', collect_news_cctv),
        ('SHMET', collect_news_shmet),
        ('EASTMONEY', collect_news_eastmoney),
    ]
    
    for name, collector in collectors:
        records = collector()
        if records:
            n = persist_news(records)
            total += n
            logger.info(f"  {name}: {len(records)} collected, {n} inserted")
        time.sleep(1)
    
    logger.info(f"=== Total: {total} records inserted ===")
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT news_source, COUNT(*) FROM akshare_news_data GROUP BY news_source ORDER BY COUNT(*) DESC")
    for r in cur.fetchall():
        logger.info(f"  {r[0]}: {r[1]}")
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
