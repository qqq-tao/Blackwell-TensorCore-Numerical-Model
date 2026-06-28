import math

def round_intermediate_to_fp16_v6(sign, extended_exponent, carry_out_bits, l_bit, fraction_bits, sticky_bit):
    """
    Rounds an intermediate floating-point result to FP16 format using Round Nearest to Even.
    Revised version to include explicit Normalization after rounding.

    Args:
        sign (int): Sign bit (0 or 1).
        extended_exponent (int): Extended exponent (integer value).
        carry_out_bits (int): 3-bit Carry out bits as an integer (0-7).
        l_bit (int): L bit (0 or 1).
        fraction_bits (int): 23-bit fraction as an integer.
        sticky_bit (int): Sticky bit (0 or 1).

    Returns:
        int: FP16 representation as an integer. Returns NaN as float('nan'), Inf as float('inf'), -Inf as float('-inf').
    """

    fp16_exponent_bias = 15
    fp16_exponent_bits = 5
    fp16_fraction_bits_count = 10
    fp16_exponent_max = (1 << fp16_exponent_bits) - 1 # 31
    fp16_exponent_inf_nan = fp16_exponent_max
    fp16_exponent_normal_max = fp16_exponent_inf_nan - 1 # 30
    fp16_exponent_min_normal = 1 # Minimum exponent for normal numbers
    fp16_exponent_min_subnormal = 0 # Exponent for subnormal numbers

    # NaN/INF Handling (Revised - Rule Supplement 3): Explicit check based on extended exponent range
    if extended_exponent >= (fp16_exponent_inf_nan + 127 + 1):
        if extended_exponent == (fp16_exponent_inf_nan + 127 + 1):
            if sign == 0:
                return float('inf')
            else:
                return float('-inf')
        else:
            return float('nan')


    # 1. Effective Significand
    significand = (carry_out_bits << 24) | (l_bit << 23) | fraction_bits
    significand_binary = bin(significand)[2:].zfill(27)

    # 2. Extract G, R, X bits
    fraction_for_rounding = significand_binary[4:]
    fraction_10bit = fraction_for_rounding[:10]
    guard_bit = int(fraction_for_rounding[10]) if len(fraction_for_rounding) > 10 else 0
    round_bit = int(fraction_for_rounding[11]) if len(fraction_for_rounding) > 11 else 0
    sticky_bit_effective = sticky_bit or (int(any(bit == '1' for bit in fraction_for_rounding[12:]))) if len(fraction_for_rounding) > 12 else sticky_bit


    # 3. Rounding Decision (Round Nearest to Even)
    fp16_fraction_int = int(fraction_10bit, 2)
    rounded_fraction_int = fp16_fraction_int

    if guard_bit == 0: # Case 1: G=0, Round down (truncate)
        pass
    elif guard_bit == 1 and round_bit == 0 and sticky_bit_effective == 0: # Case 2: Tie, round to even
        if fp16_fraction_int % 2 != 0:
            rounded_fraction_int += 1
    elif guard_bit == 1 and (round_bit == 1 or sticky_bit_effective == 1): # Case 3: G=1, R=1 or X=1, Round up
        rounded_fraction_int += 1

    # --- 4. Normalization (NEW STEP - after Rounding, before Exponent Handling) ---
    exponent_increment = 0 # Track exponent increment due to normalization
    if rounded_fraction_int > ((1 << fp16_fraction_bits_count) - 1): # Fraction overflowed due to rounding
        rounded_fraction_int >>= 1 # Right shift fraction by 1 bit (divide by 2)
        exponent_increment += 1 # Increment exponent

    # 5. Exponent Handling and Overflow/Underflow (Revised with Subnormal Rounding)
    fp16_exponent = extended_exponent - 127 + fp16_exponent_bias + exponent_increment # Add exponent increment from normalization

    if fp16_exponent > fp16_exponent_normal_max: # Exponent Overflow
        if sign == 0:
            return float('inf')
        else:
            return float('-inf')
    elif fp16_exponent < fp16_exponent_min_normal: # Exponent Underflow - Handle Subnormals
        if fp16_exponent < fp16_exponent_min_normal - fp16_fraction_bits_count: # Significant underflow, round to zero
            return 0.0 if sign == 0 else -0.0
        else: # Subnormal handling with rounding after shift
            subnormal_exponent = fp16_exponent_min_subnormal # FP16 subnormal exponent is 0
            shift_amount = fp16_exponent_min_normal - fp16_exponent
            shifted_significand = rounded_fraction_int >> shift_amount

            shifted_out_bits = rounded_fraction_int & ((1 << shift_amount) - 1)
            subnormal_guard_bit = (shifted_out_bits >> (shift_amount - 1)) & 1 if shift_amount > 0 else 0
            subnormal_round_bit = (shifted_out_bits >> (shift_amount - 2)) & 1 if shift_amount > 1 else 0
            subnormal_sticky_bit = sticky_bit_effective or (shifted_out_bits & ((1 << (shift_amount - 2)) - 1) > 0) if shift_amount > 2 else sticky_bit_effective

            subnormal_rounded_fraction_int = shifted_significand
            if subnormal_guard_bit == 0:
                pass
            elif subnormal_guard_bit == 1 and subnormal_round_bit == 0 and subnormal_sticky_bit == 0:
                if shifted_significand % 2 != 0:
                    subnormal_rounded_fraction_int += 1
            elif subnormal_guard_bit == 1 and (subnormal_round_bit == 1 or subnormal_sticky_bit == 1):
                subnormal_rounded_fraction_int += 1

            rounded_fraction_int = subnormal_rounded_fraction_int
            fp16_exponent = subnormal_exponent


    elif fp16_exponent <= 0 and rounded_fraction_int == 0: # Handle zero result explicitly
        fp16_exponent = 0


    # 6. Construct FP16 Integer Representation
    fp16_sign_bit = sign << 15
    fp16_exponent_field = (fp16_exponent & ((1 << fp16_exponent_bits) - 1)) << fp16_fraction_bits_count
    fp16_fraction_field = rounded_fraction_int & ((1 << fp16_fraction_bits_count) - 1)

    fp16_integer_representation = fp16_sign_bit | fp16_exponent_field | fp16_fraction_field
    return fp16_integer_representation


# --- Revised Test Cases (with Normalization Tests) ---
print("\n--- Revised Test Cases (with Normalization Tests) ---")

# --- 修订后的测试用例 (包含 L 位) ---
print("\n--- 修订后的测试用例 (包含 L 位) ---")

# 测试用例 1：向下舍入 (G=0)
print("\n测试用例 1 (修订版，包含 L 位)：向下舍入 (G=0)")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101010, 'sticky_bit': 0} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 2：向上舍入 (G=1, R=1, X=0)
print("\n测试用例 2 (修订版，包含 L 位)：向上舍入 (G=1, R=1, X=0)")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101011, 'sticky_bit': 0} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 3：平局，舍入到偶数 (偶数尾数)
print("\n测试用例 3 (修订版，包含 L 位)：平局，舍入到偶数 (偶数尾数)")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101000, 'sticky_bit': 0} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 4：平局，舍入到偶数 (奇数尾数)
print("\n测试用例 4 (修订版，包含 L 位)：平局，舍入到偶数 (奇数尾数)")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101011, 'sticky_bit': 0} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 5：粘滞位已设置 (G=1, R=0, X=1)
print("\n测试用例 5 (修订版，包含 L 位)：粘滞位已设置 (G=1, R=0, X=1)")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101000, 'sticky_bit': 1} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 6：溢出
print("\n测试用例 6 (修订版，包含 L 位)：溢出")
intermediate_result = {'sign': 0, 'extended_exponent': 200, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b0, 'sticky_bit': 0} # 已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果: {fp16_result}")

# 测试用例 7：负溢出
print("\n测试用例 7 (修订版，包含 L 位)：负溢出")
intermediate_result = {'sign': 1, 'extended_exponent': 200, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b0, 'sticky_bit': 0} # 已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果: {fp16_result}")

# 测试用例 8：下溢 (简化为零)
print("\n测试用例 8 (修订版，包含 L 位)：下溢 (简化为零)")
intermediate_result = {'sign': 0, 'extended_exponent': 1, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101010, 'sticky_bit': 0} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果: {fp16_result}")

# 测试用例 9：负下溢 (简化为负零)
print("\n测试用例 9 (修订版，包含 L 位)：负下溢 (简化为负零)")
intermediate_result = {'sign': 1, 'extended_exponent': 1, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101010, 'sticky_bit': 0} # 尾数已调整，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果: {fp16_result}")

# 测试用例 10：零结果
print("\n测试用例 10 (修订版，包含 L 位)：零结果")
intermediate_result = {'sign': 0, 'extended_exponent': 0, 'carry_out_bits': 0b000, 'l_bit': 0, 'fraction_bits': 0b0, 'sticky_bit': 0} # L 位对于零结果可以为 0
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 11：负零结果
print("\n测试用例 11 (修订版，包含 L 位)：负零结果")
intermediate_result = {'sign': 1, 'extended_exponent': 0, 'carry_out_bits': 0b000, 'l_bit': 0, 'fraction_bits': 0b0, 'sticky_bit': 0} # L 位对于零结果可以为 0
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 12：尾数中发生进位传播的向上舍入
print("\n测试用例 12 (修订版，包含 L 位)：尾数中发生进位传播的向上舍入")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b1111111111111111111111, 'sticky_bit': 1} # 尾数调整为最大 23 位，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果 (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}")

# 测试用例 13：指数中发生进位传播的向上舍入（如果指数为最大值，则可能溢出）
print("\n测试用例 13 (修订版，包含 L 位)：指数中发生进位传播的向上舍入（如果指数为最大值，则可能溢出）")
intermediate_result = {'sign': 0, 'extended_exponent': 30 + 127 - 15, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b1111111111111111111111, 'sticky_bit': 1} # 尾数调整为最大 23 位，已添加 L 位
fp16_result = round_intermediate_to_fp16_v3(**intermediate_result)
print(f"中间结果: {intermediate_result}, FP16 结果: {fp16_result}")

# Test case 14: Subnormal - should produce a small non-zero FP16 number
print("\nTest Case 14: Subnormal")
intermediate_result = {'sign': 0, 'extended_exponent': 110, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101010, 'sticky_bit': 0} # Exponent in subnormal range
fp16_result = round_intermediate_to_fp16_v4(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number, not zero

# Test case 15: Subnormal, rounding up
print("\nTest Case 15: Subnormal Rounding Up")
intermediate_result = {'sign': 0, 'extended_exponent': 110, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101011, 'sticky_bit': 0} # Exponent in subnormal range, needs rounding up
fp16_result = round_intermediate_to_fp16_v4(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number, rounded up

# Test case 16: Subnormal, tie to even
print("\nTest Case 16: Subnormal Tie to Even")
intermediate_result = {'sign': 0, 'extended_exponent': 110, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b01010101010101010101000, 'sticky_bit': 0} # Exponent in subnormal range, tie to even
fp16_result = round_intermediate_to_fp16_v4(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number, tie to even

# Test case 17: NaN input
print("\nTest Case 17: NaN Input")
intermediate_result = {'sign': 0, 'extended_exponent': 256, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b0, 'sticky_bit': 0} # Extended exponent signals NaN
fp16_result = round_intermediate_to_fp16_v4(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result: {fp16_result}, Is NaN: {math.isnan(fp16_result)}") # Expected: NaN

# Test case 18: Negative NaN Input
print("\nTest Case 18: Negative NaN Input (Sign bit should be ignored for NaN)")
intermediate_result = {'sign': 1, 'extended_exponent': 256, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b0, 'sticky_bit': 0} # Extended exponent signals NaN, sign bit set
fp16_result = round_intermediate_to_fp16_v4(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result: {fp16_result}, Is NaN: {math.isnan(fp16_result)}") # Expected: NaN (sign bit ignored for NaN)

# Test case 19: Zero with exponent in subnormal range
print("\nTest Case 19: Zero in Subnormal Exponent Range")
intermediate_result = {'sign': 0, 'extended_exponent': 0, 'carry_out_bits': 0b000, 'l_bit': 0, 'fraction_bits': 0b0, 'sticky_bit': 0} # Zero value, exponent in subnormal range
fp16_result = round_intermediate_to_fp16_v4(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Zero


# Test case 20: Subnormal - Rounding after shift test, rounding up subnormal fraction
print("\nTest Case 20: Subnormal Rounding After Shift - Rounding Up Fraction")
intermediate_result = {'sign': 0, 'extended_exponent': 1, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b11111111111111111111111, 'sticky_bit': 0} # Exponent in subnormal range, fraction to cause rounding after shift
fp16_result = round_intermediate_to_fp16_v5(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number, rounded up after shift

# Test case 21: Subnormal - Rounding after shift test, tie to even subnormal fraction
print("\nTest Case 21: Subnormal Rounding After Shift - Tie to Even Fraction")
intermediate_result = {'sign': 0, 'extended_exponent': 1, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b11111111111111111111110, 'sticky_bit': 0} # Exponent in subnormal range, fraction for tie to even after shift
fp16_result = round_intermediate_to_fp16_v5(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number, tie to even after shift

# Test case 22:  Exponent Overflow Boundary - Max Normal Exponent + 1
print("\nTest Case 22: Exponent Overflow Boundary - Max Normal Exponent + 1")
intermediate_result = {'sign': 0, 'extended_exponent': (fp16_exponent_normal_max + fp16_exponent_bias - 15 + 1), 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b0, 'sticky_bit': 0} # Exponent just above max normal
fp16_result = round_intermediate_to_fp16_v5(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result: {fp16_result}") # Expected: Infinity

# Test case 23: Exponent Underflow Boundary - Min Normal Exponent - 1
print("\nTest Case 23: Exponent Underflow Boundary - Min Normal Exponent - 1")
intermediate_result = {'sign': 0, 'extended_exponent': (fp16_exponent_min_normal + fp16_exponent_bias - 15 - 1), 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b1, 'sticky_bit': 0} # Exponent just below min normal
fp16_result = round_intermediate_to_fp16_v5(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number (or zero if significant underflow)

# Test case 24: L bit is 0, Subnormal case
print("\nTest Case 24: L bit is 0, Subnormal Case")
intermediate_result = {'sign': 0, 'extended_exponent': 110, 'carry_out_bits': 0b000, 'l_bit': 0, 'fraction_bits': 0b01010101010101010101010, 'sticky_bit': 0} # L bit 0, Exponent in subnormal range
fp16_result = round_intermediate_to_fp16_v5(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Subnormal number

# Test case 25: L bit is 0, Zero case
print("\nTest Case 25: L bit is 0, Zero Case")
intermediate_result = {'sign': 0, 'extended_exponent': 0, 'carry_out_bits': 0b000, 'l_bit': 0, 'fraction_bits': 0b0, 'sticky_bit': 0} # L bit 0, Exponent 0, Fraction 0
fp16_result = round_intermediate_to_fp16_v5(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Zero


# Test case 26: Normalization test - Rounding causes fraction overflow and exponent increment
print("\nTest Case 26: Normalization - Rounding causes fraction overflow")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b1111111111, 'sticky_bit': 1} # Fraction is close to max, rounding up will overflow
fp16_result = round_intermediate_to_fp16_v6(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Exponent incremented, fraction normalized

# Test case 27: Normalization test -  Fraction overflow at tie case
print("\nTest Case 27: Normalization - Fraction overflow at tie case")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b1111111110, 'sticky_bit': 1} # Fraction is tie case, rounding to even will overflow
fp16_result = round_intermediate_to_fp16_v6(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Exponent incremented, fraction normalized

# Test case 28: No Normalization - Rounding down, no overflow
print("\nTest Case 28: No Normalization - Rounding down, no overflow")
intermediate_result = {'sign': 0, 'extended_exponent': 130, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b0101010101, 'sticky_bit': 0} # Rounding down, no overflow
fp16_result = round_intermediate_to_fp16_v6(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: No normalization needed

# Test case 29: Normalization and then Overflow
print("\nTest Case 29: Normalization and then Overflow")
intermediate_result = {
    'extended_exponent': 142,  # 142 - 127 + 15 + 1 = 31 > 30
    'carry_out_bits': 0b000,
    'l_bit': 1,
    'fraction_bits': 0b1111111111,
    'sticky_bit': 1,
    'sign': 0}
fp16_result = round_intermediate_to_fp16_v6(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result: {fp16_result}") # Expected: Infinity

# Test case 30: Normalization and then No Overflow (Exponent at max normal -1 before normalization)
print("\nTest Case 30: Normalization and No Overflow (Exponent at max normal -1)")
intermediate_result = {'sign': 0, 'extended_exponent': fp16_exponent_normal_max + fp16_exponent_bias - 15 - 1, 'carry_out_bits': 0b000, 'l_bit': 1, 'fraction_bits': 0b1111111111, 'sticky_bit': 1} # Rounding and normalization will NOT cause exponent overflow
fp16_result = round_intermediate_to_fp16_v6(**intermediate_result)
print(f"Intermediate: {intermediate_result}, FP16 Result (int): {fp16_result}, (hex): {hex(fp16_result) if not isinstance(fp16_result, float) else fp16_result}") # Expected: Max normal exponent, normalized fraction