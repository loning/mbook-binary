#!/usr/bin/env python3
"""
Simple script to generate a single theory filename based on theory number.
Usage: python generate_single_filename.py <theory_number>
"""

import sys

def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 2]
    
    fib = [1, 2]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def zeckendorf_decomposition(n):
    """Find the Zeckendorf decomposition of n using Fibonacci numbers."""
    if n <= 0:
        return []
    
    # Generate Fibonacci sequence up to n
    fib = fibonacci_sequence(20)  # Should be enough for reasonable theory numbers
    fib = [f for f in fib if f <= n]
    
    decomp = []
    remaining = n
    
    # Greedy algorithm: always use the largest possible Fibonacci number
    for i in range(len(fib)-1, -1, -1):
        if fib[i] <= remaining:
            decomp.append(i+1)  # Fibonacci index (F1, F2, F3, ...)
            remaining -= fib[i]
            if remaining == 0:
                break
    
    return sorted(decomp)

def determine_theory_type(n):
    """Determine if n is PRIME, FIBONACCI, or COMPOSITE."""
    # Check if n is a Fibonacci number
    fib_sequence = fibonacci_sequence(20)
    if n in fib_sequence:
        return "FIBONACCI"
    
    # Check if n is prime
    if n < 2:
        return "COMPOSITE"
    if n == 2:
        return "PRIME"
    if n % 2 == 0:
        return "COMPOSITE"
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return "COMPOSITE"
    return "PRIME"

def generate_theory_filename(theory_num):
    """Generate filename for a single theory."""
    decomp = zeckendorf_decomposition(theory_num)
    theory_type = determine_theory_type(theory_num)
    
    # Build Zeckendorf string
    zeck_parts = [f"F{i}" for i in decomp]
    zeck_str = "+".join(zeck_parts)
    zeck_values = "+".join([str(fibonacci_sequence(20)[i-1]) for i in decomp])
    
    # Build FROM dependencies
    from_deps = "+".join([f"T{fibonacci_sequence(20)[i-1]}" for i in decomp])
    
    # Generate basic theory name and tensor name
    theory_name = f"Theory{theory_num}"
    tensor_name = f"Theory{theory_num}Tensor"
    
    filename = f"T{theory_num}__{theory_name}__{theory_type}__ZECK_{zeck_str}__FROM__{from_deps}__TO__{tensor_name}.md"
    
    return {
        'filename': filename,
        'theory_num': theory_num,
        'zeckendorf': f"{zeck_str} = {zeck_values} = {theory_num}",
        'theory_type': theory_type,
        'dependencies': from_deps.split('+'),
        'decomp_indices': decomp
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_single_filename.py <theory_number>")
        sys.exit(1)
    
    try:
        theory_num = int(sys.argv[1])
        if theory_num <= 0:
            print("Theory number must be positive")
            sys.exit(1)
            
        result = generate_theory_filename(theory_num)
        
        print(f"Theory Number: T{result['theory_num']}")
        print(f"Zeckendorf Decomposition: {result['zeckendorf']}")
        print(f"Theory Type: {result['theory_type']}")
        print(f"Dependencies: {{{', '.join(result['dependencies'])}}}")
        print(f"Filename: {result['filename']}")
        
    except ValueError:
        print("Error: Theory number must be an integer")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()