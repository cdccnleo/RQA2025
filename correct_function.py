def get_stocks_by_industry(industry: str) -> List[Dict[str, Any]]:
    """
    根据行业获取股票列表
    
    Args:
        industry: 行业名称
        
    Returns:
        股票列表
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, name, ipo_date, industry, market,
                   total_share, float_share, pe, pb, roe
            FROM stock_basic_info
            WHERE industry = %s
            ORDER BY symbol
        """, (industry,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        stocks = []
        for row in rows:
            stocks.append({
                "symbol": row[0],
                "name": row[1],
                "ipo_date": row[2].isoformat() if row[2] else None,
                "industry": row[3],
                "market": row[4],
                "total_share": row[5],
                "float_share": row[6],
                "pe": row[7],
                "pb": row[8],
                "roe": row[9]
            })
        
        return stocks
        
    except Exception as e:
        logger.error(f"根据行业获取股票列表失败: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)
