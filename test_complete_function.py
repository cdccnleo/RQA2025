from typing import List, Dict, Any

print("开始测试完整函数定义...")

# 完整的函数定义
function_def = '''
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
        conn = None
        if not conn:
            return []
        
        cursor = None
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
        print(f"根据行业获取股票列表失败: {e}")
        return []
    finally:
        if conn:
            pass
'''

print('函数定义:')
print(function_def)
print('\n开始测试语法...')

import ast
try:
    tree = ast.parse(function_def)
    print('语法检查通过！')
except SyntaxError as e:
    print(f'语法错误: {e}')
    print(f'错误位置: 第 {e.lineno} 行, 第 {e.offset} 列')
except Exception as e:
    print(f'其他错误: {e}')
