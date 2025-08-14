#!/usr/bin/env python3
"""
Theory Dependency Verification Script
验证BDAG理论系统中所有依赖关系的正确性
"""

import os
import re
import sys
from pathlib import Path

# Fibonacci sequence mapping
FIBONACCI = {
    1: 1, 2: 2, 3: 3, 4: 5, 5: 8, 6: 13, 7: 21, 8: 34, 9: 55, 10: 89, 11: 144, 12: 233
}

def get_zeckendorf_decomposition(n):
    """Get the Zeckendorf decomposition of n"""
    if n <= 0:
        return []
    
    fib_nums = sorted(FIBONACCI.values())
    result = []
    
    while n > 0:
        for i in range(len(fib_nums) - 1, -1, -1):
            if fib_nums[i] <= n:
                fib_index = next(k for k, v in FIBONACCI.items() if v == fib_nums[i])
                result.append(fib_index)
                n -= fib_nums[i]
                break
    
    return sorted(result)

def get_theory_dependencies_from_decomposition(theory_num):
    """Get theory dependencies based on Zeckendorf decomposition"""
    if theory_num <= 0:
        return []
    
    decomposition = get_zeckendorf_decomposition(theory_num)
    
    # If theory corresponds to a single Fibonacci number (FIBONACCI/PRIME-FIB types)
    # it should not have dependencies (fundamental theory)
    if len(decomposition) == 1 and FIBONACCI[decomposition[0]] == theory_num:
        return []  # Fundamental theories have no dependencies
    
    # For COMPOSITE theories, dependencies come from Zeckendorf decomposition
    dependencies = []
    for fib_index in decomposition:
        fib_value = FIBONACCI[fib_index]
        if fib_value != theory_num:  # Don't depend on yourself
            dependencies.append(fib_value)
    
    return sorted(dependencies)

def verify_theory_dependencies(filepath):
    """Verify dependencies in a theory file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract theory number from filename
        filename = os.path.basename(filepath)
        theory_match = re.match(r'T(\d+)__', filename)
        if not theory_match:
            return None, "Cannot extract theory number from filename"
        
        theory_num = int(theory_match.group(1))
        
        # Extract dependencies from filename (FROM section)
        from_match = re.search(r'__FROM__([^_]+)__TO__', filename)
        if not from_match:
            return None, "Cannot extract dependencies from filename"
        
        from_str = from_match.group(1)
        
        # Parse dependencies from filename
        filename_deps = []
        if from_str == "UNIVERSE":
            filename_deps = []  # T1 has no dependencies
        elif '+' in from_str:
            for part in from_str.split('+'):
                match = re.match(r'T(\d+)', part)
                if match:
                    filename_deps.append(int(match.group(1)))
        else:
            match = re.match(r'T(\d+)', from_str)
            if match:
                filename_deps.append(int(match.group(1)))
        
        # Get expected dependencies from Zeckendorf decomposition
        if theory_num == 1:
            expected_deps = []  # T1 has no dependencies
        else:
            expected_deps = get_theory_dependencies_from_decomposition(theory_num)
        
        # Check if they match
        if sorted(filename_deps) == sorted(expected_deps):
            return True, "✅ Dependencies correct"
        else:
            expected_str = "+".join([f"T{d}" for d in expected_deps]) if expected_deps else "UNIVERSE"
            filename_str = "+".join([f"T{d}" for d in filename_deps]) if filename_deps else "UNIVERSE"
            return False, f"❌ Dependency mismatch: filename={filename_str}, expected={expected_str}"
            
    except Exception as e:
        return None, f"❌ Error reading file: {e}"

def main():
    """Main verification function"""
    theories_dir = Path("/Users/cookie/mbook-binary/src/bdag/theories")
    
    if not theories_dir.exists():
        print(f"❌ Theories directory not found: {theories_dir}")
        return False
    
    print("🔗 BDAG Theory Dependency Verification")
    print("=" * 50)
    print(f"📁 Checking directory: {theories_dir}")
    print()
    
    # Get all theory files
    theory_files = sorted(theories_dir.glob("T*.md"))
    
    if not theory_files:
        print("❌ No theory files found")
        return False
    
    print(f"📊 Found {len(theory_files)} theory files")
    print()
    
    errors = []
    successes = 0
    
    print("📋 Dependency Verification Results:")
    print("-" * 40)
    
    for filepath in theory_files:
        filename = filepath.name
        result, message = verify_theory_dependencies(filepath)
        
        if result is True:
            print(f"{filename}: {message}")
            successes += 1
        elif result is False:
            print(f"{filename}: {message}")
            errors.append((filename, message))
        else:
            print(f"{filename}: {message}")
            errors.append((filename, message))
    
    print()
    print("=" * 50)
    print(f"📊 Summary: {successes} correct, {len(errors)} errors")
    
    if errors:
        print()
        print("❌ Errors found:")
        for filename, error in errors:
            print(f"  • {filename}: {error}")
        return False
    else:
        print("🎉 All dependencies are correct!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)