#!/usr/bin/env python3
"""
Simple checker for data/charset_ja.txt
 - Counts characters
 - Reports duplicates
 - Ensures UTF-8
"""
from pathlib import Path
from collections import Counter


def main():
    path = Path('data/charset_ja.txt')
    if not path.exists():
        print('data/charset_ja.txt not found')
        return
    text = path.read_text(encoding='utf-8')
    chars = []
    for line in text.splitlines():
        line = line.rstrip('\n')
        if not line or line.startswith('#'):
            continue
        if len(line) != 1:
            print(f'WARN: line is not a single character: {line}')
        chars.append(line)
    c = Counter(chars)
    total = len(chars)
    dups = [ch for ch, n in c.items() if n > 1]
    print(f'Characters: {total}')
    if dups:
        print(f'Duplicates ({len(dups)}): ' + ''.join(dups))
    else:
        print('No duplicates found')


if __name__ == '__main__':
    main()

