#!/usr/bin/env python3
import csv
from collections import Counter

manifest_path = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv'

with open(manifest_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

train_rows = [r for r in rows if r.get('split') == 'train' and r.get('family') == 'green8']

# 统计省份分布
province_counts = Counter()
for row in train_rows:
    text = row.get('text', '')
    if text:
        province_counts[text[0]] += 1

# 按数量排序
sorted_provinces = sorted(province_counts.items(), key=lambda x: x[1], reverse=True)

total = len(train_rows)
print(f'Train green8 total: {total}')
print('\nProvince distribution (top 15):')
for prov, count in sorted_provinces[:15]:
    pct = count / total * 100
    print(f'{prov}: {count} ({pct:.2f}%)')

# 重点关注问题省份
print('\n=== Problem provinces (high error rate) ===')
problem_provinces = ['苏', '沪', '闽', '浙']
for prov in problem_provinces:
    count = province_counts.get(prov, 0)
    pct = count / total * 100 if total > 0 else 0
    print(f'{prov}: {count} samples ({pct:.2f}%)')

# 对比测试集分布
print('\n=== Test set distribution (for comparison) ===')
test_rows = [r for r in rows if r.get('split') == 'test' and r.get('family') == 'green8']
test_counts = Counter()
for row in test_rows:
    text = row.get('text', '')
    if text:
        test_counts[text[0]] += 1

test_total = len(test_rows)
print(f'Test green8 total: {test_total}')
for prov in problem_provinces:
    train_count = province_counts.get(prov, 0)
    test_count = test_counts.get(prov, 0)
    train_pct = train_count / total * 100 if total > 0 else 0
    test_pct = test_count / test_total * 100 if test_total > 0 else 0
    print(f'{prov}: train={train_count}({train_pct:.1f}%) test={test_count}({test_pct:.1f}%)')

# 检查synthetic vs real分布
print('\n=== Source distribution for problem provinces ===')
for prov in problem_provinces:
    prov_train = [r for r in train_rows if r.get('text', '').startswith(prov)]
    real = sum(1 for r in prov_train if 'real' in r.get('source', ''))
    synthetic = sum(1 for r in prov_train if 'synthetic' in r.get('source', ''))
    pseudo = sum(1 for r in prov_train if 'pseudo' in r.get('source', ''))
    total_prov = len(prov_train)
    print(f'{prov}: total={total_prov}, real={real}, synthetic={synthetic}, pseudo={pseudo}')
