import json
d=json.load(open('reports/coverage.json'))
print(f"{d['totals']['percent_covered']:.2f}%")
