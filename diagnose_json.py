#!/usr/bin/env python3
"""
Diagnose and fix common JSON issues in REC_*.json files.

Usage:
    python diagnose_json.py REC_i.json
    python diagnose_json.py REC_i.json --fix
"""

import json
import sys
import os


def diagnose_json(filepath, fix=False):
    """Diagnose JSON file issues and optionally fix them."""

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    print(f"File: {filepath}")
    print(f"Length: {len(content)} characters")
    print(f"Starts with: {content[:50]}")
    print(f"Ends with: {content[-50:]}")
    print()

    issues_found = []

    # Check for common issues
    # Issue 1: Extra comma after opening bracket [,{
    if content.startswith('[,'):
        issues_found.append("Extra comma after opening bracket '[,'")
        if fix:
            content = '[' + content[2:]
            print("  Fixed: Removed extra comma after '['")

    # Issue 2: Missing closing bracket
    stripped = content.rstrip()
    if stripped and not stripped.endswith(']'):
        issues_found.append("Missing closing bracket ']'")
        if fix:
            content = content.rstrip() + '\n]'
            print("  Fixed: Added closing bracket ']'")

    # Issue 3: Trailing comma before closing bracket
    if ',]' in content or ', ]' in content:
        issues_found.append("Trailing comma before ']'")
        if fix:
            content = content.replace(',]', ']').replace(', ]', ']')
            print("  Fixed: Removed trailing comma before ']'")

    # Issue 4: Multiple JSON arrays concatenated (][, or ],{)
    if '][' in content or '],{' in content:
        issues_found.append("Multiple JSON arrays concatenated")
        if fix:
            # Keep only the first complete array
            # Find first ] that's followed by [ or ,
            import re
            # Find pattern }][ or }],
            match = re.search(r'\}]\s*[\[,]', content)
            if match:
                content = content[:match.start() + 2]  # Keep up to }]
                print(f"  Fixed: Kept only first JSON array (truncated at position {match.start() + 2})")

    def write_fixed_file():
        """Persist fixed content and keep one backup copy."""
        backup_path = filepath + '.bak'
        os.rename(filepath, backup_path)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"\nFixed file saved. Original backed up to: {backup_path}")

    # Try to parse
    try:
        data = json.loads(content)
        if fix and issues_found:
            write_fixed_file()

            # Verify persisted fix from disk
            with open(filepath, 'r') as f:
                data = json.load(f)
            print("Verification: Fixed file is valid JSON")

        print(f"JSON Valid: Yes")
        print(f"Number of trajectories: {len(data)}")
        return True
    except json.JSONDecodeError as e:
        print(f"JSON Valid: No")
        print(f"Error: {e.msg}")
        print(f"Error position: {e.pos}")
        print(f"Context around error:")
        start = max(0, e.pos - 50)
        end = min(len(content), e.pos + 50)
        print(f"  ...{content[start:e.pos]}<<<ERROR>>>{content[e.pos:end]}...")

        if issues_found:
            print(f"\nIssues found: {issues_found}")

        if fix and issues_found:
            write_fixed_file()

            # Verify fix
            try:
                with open(filepath, 'r') as f:
                    json.load(f)
                print("Verification: Fixed file is valid JSON")
                return True
            except json.JSONDecodeError as e2:
                print(f"Verification failed: {e2.msg}")
                return False

        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable REC files:")
        for f in os.listdir('.'):
            if f.startswith('REC_') and f.endswith('.json'):
                print(f"  {f}")
        sys.exit(1)

    filepath = sys.argv[1]
    fix = '--fix' in sys.argv

    if fix:
        print("Running in FIX mode\n")
    else:
        print("Running in DIAGNOSE mode (use --fix to auto-fix issues)\n")

    success = diagnose_json(filepath, fix=fix)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
