def summation_i_squared(n):
    # Check if n is a valid number
    if not isinstance(n, (int, float)) or n < 0:
        return None
    
    # Convert to integer
    n = int(n)
    
    # Base case: if n is 0, return 0
    if n == 0:
        return 0
    
    # Recursive case: n² + sum of squares from 1 to (n-1)
    return n * n + summation_i_squared(n - 1)
