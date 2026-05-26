#!/usr/bin/env python3
import json
from pathlib import Path

FILE = Path(__file__).parent.parent / "data" / "extracted_output.json"

with open(FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

ids = [item['id'] for item in data]
unique = set(ids)

print(f"总记录: {len(ids)}")
print(f"唯一ID: {len(unique)}")

if len(ids) == len(unique):
    print("✅ ID全部唯一")
else:
    print(f"❌ 有重复: {len(ids) - len(unique)} 个")
    from collections import Counter
    for k, v in Counter(ids).items():
        if v > 1:
            print(f"   '{k}': {v}次")