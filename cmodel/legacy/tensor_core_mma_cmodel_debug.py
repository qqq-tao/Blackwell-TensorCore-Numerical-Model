import sys
import unittest
import numpy as np
from enum import Enum
import math


class Special_tag_enum(Enum):
    Zero = 0
    INF = 1
    NEG_INF = 2
    NaN = 3
    Denormal = 4


class RoundingMode(Enum):
    RTZ = 0
    RNE = 1


bit_length = {"e4m3": 8, "e5m2": 8, "f16": 16, "bf16": 16, "tf32": 32, "f32": 32}

exponent_dict = {"e5m2": 5, "e4m3": 4, "f16": 5, "f32": 8, "tf32": 8, "bf16": 8}

mantissa_dict = {"e5m2": 2, "e4m3": 3, "f16": 10, "f32": 23, "tf32": 10, "bf16": 7}


class format:
    def __init__(self, dtype):
        self.dtype = dtype
        self.bit_length = bit_length[dtype]
        self.exp_bits = exponent_dict[dtype]
        self.mant_bits = mantissa_dict[dtype]
        self.bias = (2 ** (self.exp_bits - 1)) - 1


np_unsigned_int_dict = {8: np.uint8, 16: np.uint16, 32: np.uint32}


class TensorCore:
    def __init__(self, D_dtype, A_dtype, B_dtype, C_dtype):
        """
        Initialize the Tensor Core C Model

        """
        self.A_format = format(A_dtype)
        self.B_format = format(B_dtype)
        self.C_format = format(C_dtype)
        self.D_format = format(D_dtype)
        self.chunk_size = 128 / bit_length[A_dtype]
        if A_dtype == "f16" and C_dtype == "f16":
            self.RoundingMode = RoundingMode.RNE
        else:
            self.RoundingMode = RoundingMode.RTZ

        if A_dtype in ("e5m2", "e4m3"):
            self.accumulator_fraction_width = 13
        else:
            self.accumulator_fraction_width = 24

    def recursive_mma_accumulate(
        self, A: np.array, B: np.array, C: np.array
    ) -> np.array:
        """Perform recursive accumulative MMA, where the output of one step is the input for the next.

        Args:
            A (np.array): Input matrix A.
            B (np.array): Input matrix B.
            C (np.array): Input matrix C (initial state).

        Returns:
            np.array: Final accumulated result after recursive MMA operations.
        """
        # 递归终止条件：当矩阵尺寸小于分块阈值时直接计算
        if A.shape[0] <= self.chunk_size:
            result = self.single_step_mma(A, B, C)
            return np.array(result).reshape(C.shape)

        # 分块递归逻辑：C 的输出作为下一段的输入
        k = int(self.chunk_size)
        # 递归处理前 k 行，生成更新后的 C_updated
        C_updated = self.recursive_mma_accumulate(A[:k], B[:k], C)
        # print("=========================C_updated================================")
        # print(C_updated)
        return self.recursive_mma_accumulate(A[k:], B[k:], C_updated)

    def single_step_mma(self, A: np.array, B: np.array, C: np.array):  # -> INT:
        """
        Perform a single-step matrix multiply-accumulate (MMA) operation (simulating Tensor Core behavior).

        Args:
            A_block (np.array): Input matrix A (block).
            B_block (np.array): Input matrix B (block).
            C_in (np.array): Input matrix C (initial state).

        Returns:
            np.array: Updated matrix C after MMA operation.
        """

        # Extract components from A and B matrices
        A_sign, A_exponent, A_significand, A_special_tag = (
            self.extract_sign_exponent_mantissa(A, self.A_format.dtype)
        )
        B_sign, B_exponent, B_significand, B_special_tag = (
            self.extract_sign_exponent_mantissa(B, self.B_format.dtype)
        )

        C_sign, C_exponent, C_significand, C_special_tag = (
            self.extract_sign_exponent_mantissa(C, self.C_format.dtype)
        )

        # A * B
        # Initialize partial results
        partial_sign = np.zeros_like(A_sign, dtype=np.uint8)
        partial_exponent = np.zeros_like(A_exponent, dtype=np.int16)
        partial_significand = np.zeros_like(A_significand, dtype=np.uint32)
        partial_special = np.full_like(A_sign, None, dtype=object)

        # Process special cases first

        # Multiply normal numbers

        # 1. NaN situations: if either A or B is NaN, result is NaN
        nan_mask = (A_special_tag == Special_tag_enum.NaN) | (
            B_special_tag == Special_tag_enum.NaN
        )

        # 2. INF * 0 or 0 * INF: result is NaN
        inf_mul_zero_mask = (
            (A_special_tag == Special_tag_enum.INF)
            & (B_special_tag == Special_tag_enum.Zero)
        ) | (
            (A_special_tag == Special_tag_enum.Zero)
            & (B_special_tag == Special_tag_enum.INF)
        )
        nan_mask |= inf_mul_zero_mask

        # Set NaN in partial_special, if exists.
        partial_special[nan_mask] = Special_tag_enum.NaN

        normal_mask = ~nan_mask & (A_special_tag == None) & (B_special_tag == None)
        product_signs = A_sign ^ B_sign
        # Handle INF cases
        if np.any(~normal_mask & ~nan_mask):
            inf_mask = (
                (
                    (A_special_tag == Special_tag_enum.INF)
                    | (B_special_tag == Special_tag_enum.INF)
                )
                & (partial_sign == 0)
                & ~nan_mask
            )
            neg_inf_mask = (
                (
                    (A_special_tag == Special_tag_enum.INF)
                    | (B_special_tag == Special_tag_enum.INF)
                )
                & (partial_sign == 1)
                & ~nan_mask
            )

            partial_special[inf_mask] = Special_tag_enum.INF
            partial_special[neg_inf_mask] = Special_tag_enum.NEG_INF

        combined_exponent = (
            A_exponent[normal_mask].astype(np.int16)
            + B_exponent[normal_mask].astype(np.int16)
            - self.A_format.bias
            - self.B_format.bias
            + self.C_format.bias
        )

        combined_significand = A_significand[normal_mask].astype(
            np.uint32
        ) * B_significand[normal_mask].astype(np.uint32)
        partial_significand_fractional_part_bit_length = (
            self.A_format.mant_bits + self.B_format.mant_bits
        )

        # Point adjustment
        aligned_significand = combined_significand << (
            self.accumulator_fraction_width
            - partial_significand_fractional_part_bit_length
        )

        partial_sign = product_signs
        partial_exponent[normal_mask] = combined_exponent
        partial_significand[normal_mask] = aligned_significand

        # C + A * B
        # 1. point adjustment
        C_point_gap = self.accumulator_fraction_width - self.C_format.mant_bits
        C_significand = (
            C_significand.astype(np.uint32) << C_point_gap
            if C_point_gap > 0
            else C_significand.astype(np.uint32) >> (-C_point_gap)
        )
        partial_sign = np.concatenate(
            (partial_sign, C_sign.astype(np.uint8)), axis=0
        ).astype(np.int32)
        partial_exponent = np.concatenate(
            (partial_exponent, C_exponent.astype(np.int16)), axis=0
        ).astype(np.int32)
        partial_significand = np.concatenate(
            (partial_significand, C_significand.astype(np.uint32)), axis=0
        ).astype(np.int32)
        partial_special = np.concatenate((partial_special, C_special_tag), axis=0)
        return self.stage2_add(
            partial_sign, partial_exponent, partial_significand, partial_special
        )

    def stage2_add(
        self, partial_sign, partial_exponent, partial_significand, partial_special
    ):
        # print(partial_exponent)
        # print(partial_significand)

        sum_special_tag = np.full_like(1, None, dtype=object)

        if np.any(partial_special == Special_tag_enum.NaN):
            sum_special_tag = Special_tag_enum.NaN
        else:

            # set sum = NaN, if  exists INF - INF.
            neg_inf_mask = (partial_sign == 1) & (
                partial_special == Special_tag_enum.INF
            )
            partial_special[neg_inf_mask] = Special_tag_enum.NEG_INF

            if np.any(partial_special == Special_tag_enum.INF) and np.any(
                partial_special == Special_tag_enum.NEG_INF
            ):
                sum_special_tag = Special_tag_enum.NaN
            elif np.any(partial_special == Special_tag_enum.INF):
                sum_special_tag = Special_tag_enum.INF
            elif np.any(partial_special == Special_tag_enum.NEG_INF):
                sum_special_tag = Special_tag_enum.NEG_INF

        # print(sum_special_tag)
        if sum_special_tag != None and sum_special_tag != Special_tag_enum.Zero:
            if sum_special_tag == Special_tag_enum.Zero:
                return self.data_compact(0, 0, 0)
            if sum_special_tag == Special_tag_enum.INF:
                return self.data_compact(0, 0xFF, 0)
            if sum_special_tag == Special_tag_enum.NEG_INF:
                return self.data_compact(1, 0xFF, 0)
            if sum_special_tag == Special_tag_enum.NaN:
                return self.data_compact(0, 0xFF, 0x7FFFFF)
        else:
            # align all partials to max exponent
            max_exp = np.max(partial_exponent[partial_special == None])
            aligned_significands = []
            for i in range(len(partial_significand)):
                # if partial_special[i] != None:
                #     continue
                exp_diff = max_exp - partial_exponent[i]
                if exp_diff > 0:
                    sticky_bit_mask = np.int32(2 ** (exp_diff) - 1)
                    sticky_bit = (
                        np.int32(1)
                        if np.bitwise_and(partial_significand[i], sticky_bit_mask) > 0
                        else np.int32(0)
                    )
                    aligned = partial_significand[i] >> exp_diff
                    # aligned = aligned | sticky_bit
                else:
                    aligned = partial_significand[i]
                aligned_significands.append(aligned)
            # print(type(aligned_significands[0]))
            sum_significand = sum((-1) ** partial_sign * aligned_significands)
            sum_exponent = max_exp

            if sum_significand == 0:
                return self.stage3_normalization_rounding(
                    0, 0, 0, Special_tag_enum.Zero
                )

            sum_sign = 1 if sum_significand < 0 else 0
            sum_significand = abs(sum_significand)

            # print(f"sum_exponent: {sum_exponent}")
            # print(f"sum_significand: {sum_significand}")
            return self.stage3_normalization_rounding(
                sum_sign, sum_exponent, sum_significand, sum_special_tag
            )

    def right_shift(self, exponent, significand, shift_amount):
        right_shift_mask = np.int32(2 ** (shift_amount) - 1)
        sticky_bit = 1 if np.bitwise_and(significand, right_shift_mask) > 0 else 0
        significand >>= shift_amount
        significand = np.bitwise_or(significand, sticky_bit)
        exponent += shift_amount
        dtype = "f16"
        k = bit_length[dtype]
        w = exponent_dict[dtype]
        f = mantissa_dict[dtype]
        return exponent, significand

    def left_shift(self, exponent, significand, shift_amount):
        # print(shift_amount)
        # print(type(significand))
        # sticky_bit = np.bitwise_and(significand, np.int32(1))
        # significand = np.bitwise_and(np.uint32(significand), np.uint32(0xFFFFFFFF))
        significand <<= -shift_amount
        exponent += shift_amount

        # print()
        # dtype = "f16"
        # k = bit_length[dtype]
        # w = exponent_dict[dtype]
        # f = mantissa_dict[dtype]
        # print("right_shift: ")
        # sys.stdout.write("" + str(exponent) + " " + str(significand) + "\t")
        # print()
        return exponent, significand

    def stage3_normalization_rounding(self, sign, exponent, significand, special_tag):
        # print(
        #     f"{sign:01b}"
        #     + " "
        #     + f"{exponent:08b}"
        #     + " "
        #     + f"{significand:018b}"[0:5]
        #     + "."
        #     + f"{significand:018b}"[5:]
        # )
        # # significand = np.bitwise_and(significand, 2**30 - 2)
        if special_tag == Special_tag_enum.Zero:
            return self.data_compact(0, 0, 0)
        if special_tag == Special_tag_enum.INF:
            return self.data_compact(0, 0xFF, 0)
        if special_tag == Special_tag_enum.NEG_INF:
            return self.data_compact(1, 0xFF, 0)
        if special_tag == Special_tag_enum.NaN:
            return self.data_compact(0, 0xFF, 0x7FFFFF)

        # significand = int(significand)
        # Normalization:
        significand_width = int(significand).bit_length()
        gap = significand_width - (self.accumulator_fraction_width + 1)
        # print(f"exponent: {exponent}, significand_width: {significand_width}")

        if significand == 0:
            return self.data_compact(0, 0, 0)
        else:
            if gap + exponent > 2 ** (self.D_format.exp_bits) - 2:
                # print("case 0")
                return (
                    self.data_compact(0, 0xFF, 0)
                    if sign == 0
                    else self.data_compact(1, 0xFF, 0)
                )
            elif gap + exponent < 1 - self.accumulator_fraction_width:
                # print("case 1")
                return self.data_compact(0, 0, 0)
            elif 1 - self.accumulator_fraction_width <= (gap + exponent) < 1:
                # print("case 2")

                shift_amount = 1 - exponent

                if shift_amount > 0:
                    if self.A_format.bit_length != 8:
                        exponent, significand = self.right_shift(
                            exponent, significand, shift_amount
                        )
                    else:
                        significand >>= shift_amount
                        exponent += shift_amount
                elif shift_amount < 0:
                    exponent, significand = self.left_shift(
                        exponent, significand, shift_amount
                    )
                else:
                    pass
            else:
                # print("case 3")
                shift_amount = gap
                # print(shift_amount)
                if shift_amount > 0:
                    if self.A_format.bit_length != 8:
                        exponent, significand = self.right_shift(
                            exponent, significand, shift_amount
                        )
                    else:
                        significand >>= shift_amount
                        exponent += shift_amount
                elif shift_amount < 0:
                    exponent, significand = self.left_shift(
                        exponent, significand, shift_amount
                    )
                    # print(exponent, significand)
                else:
                    pass
        significand_width = int(significand).bit_length()
        # print(f"after nromalize significand_width: {significand_width}")

        # print(f"unnormalized significand: {significand}")

        # rounding()
        shift_amount = self.accumulator_fraction_width - self.C_format.mant_bits
        if self.RoundingMode == RoundingMode.RNE:

            sticky_mask = np.int32(2 ** (shift_amount - 2) - 1)
            X_bit = (
                np.int32(1) if np.bitwise_and(sticky_mask, significand) else np.int32(0)
            )
            # print(1 << (shift_amount))
            # print(np.bitwise_or(1 << (shift_amount), significand))
            LSB = (
                np.int32(1)
                if np.bitwise_and(1 << (shift_amount), significand)
                else np.int32(0)
            )
            G_bit = (
                np.int32(1)
                if np.bitwise_and(1 << (shift_amount - 1), significand)
                else np.int32(0)
            )
            R_bit = (
                np.int32(1)
                if np.bitwise_and(1 << (shift_amount - 2), significand)
                else np.int32(0)
            )
            significand >>= shift_amount
            if G_bit and (LSB | R_bit | X_bit):
                # print("True")
                significand += 1

            mantissa_width_after_rounding = int(significand).bit_length()
            if mantissa_width_after_rounding > self.C_format.mant_bits + 1:
                significand >> 1
                exponent += 1

        elif self.RoundingMode == RoundingMode.RTZ:
            shift_amount = self.accumulator_fraction_width - self.C_format.mant_bits
            # print()
            # print("NTZ")
            # print(shift_amount)
            # print(significand)
            # print()
            if shift_amount > 0:
                significand = significand >> (shift_amount)
            else:
                significand = significand << (-shift_amount)

        if exponent > (2**self.C_format.exp_bits - 1):
            if sign == 1:
                special_tag = Special_tag_enum.NEG_INF
            else:
                special_tag = Special_tag_enum.INF
        # print(f"normalization {special_tag}")
        # print(exponent, significand)
        if exponent <= 0 or significand == 0:
            special_tag = Special_tag_enum.Zero

        if (
            exponent == 1
            and int(significand).bit_length() < self.C_format.mant_bits + 1
        ):
            special_tag = Special_tag_enum.Denormal
        # print(special_tag)
        if special_tag == Special_tag_enum.Denormal:
            return self.data_compact(sign, 0, significand)
        if special_tag == Special_tag_enum.Zero:
            return self.data_compact(0, 0, 0)
        if special_tag == Special_tag_enum.INF:
            return self.data_compact(0, 0xFF, 0)
        if special_tag == Special_tag_enum.NEG_INF:
            return self.data_compact(1, 0xFF, 0)
        if special_tag == Special_tag_enum.NaN:
            return self.data_compact(0, 0xFF, 0x7FFFFF)

        return self.data_compact(sign, exponent, significand)

    def extract_sign_exponent_mantissa(self, array, dtype):
        k = bit_length[dtype]
        w = exponent_dict[dtype]
        f = mantissa_dict[dtype]
        sign = np.right_shift(array, k - 1)
        sign_removal = np.left_shift(array, 1)

        exponent = np.right_shift(sign_removal, k - w)
        sign_exponent_removal = np.left_shift(array, 1 + w)

        mantissa = np.right_shift(sign_exponent_removal, 1 + w + (k - 1 - w - f))
        special_tags = np.empty_like(exponent, dtype=object)
        # Handle denormal numbers and special values
        zero_mask = (exponent == 0) & (mantissa == 0)
        special_tags[zero_mask] = Special_tag_enum.Zero
        inf_nan_mask = exponent == (2**w - 1)
        nan_mask = (
            inf_nan_mask & (mantissa != 0)
            if dtype != "e4m3"
            else inf_nan_mask & (mantissa == 2**f - 1)
        )
        inf_mask = inf_nan_mask & (mantissa == 0) if dtype != "e4m3" else False

        special_tags[nan_mask] = Special_tag_enum.NaN
        special_tags[inf_mask] = Special_tag_enum.INF

        denormal_flag = (exponent == 0) & (mantissa != 0)

        implicit_bit = (
            ~(nan_mask | inf_mask | inf_mask | zero_mask | denormal_flag)
        ).astype(np_unsigned_int_dict[bit_length[dtype]])

        exponent += denormal_flag.astype(np_unsigned_int_dict[bit_length[dtype]])

        significand = mantissa + np.left_shift(implicit_bit, f)
        # print(type(array))
        binary_array = [f"{x:0{str(k)}b}" for x in array]
        binary_sign = [f"{x:01b}" for x in sign]
        binary_exponent = [f"{x:0{str(w)}b}" for x in exponent]
        binary_significand = [f"{x:0{str(f+1)}b}" for x in significand]

        # print()

        # for i in range(len(binary_array)):
        #     sys.stdout.write(
        #         str(binary_array[i][0:1])
        #         + " "
        #         + str(binary_array[i][1 : w + 1])
        #         + " "
        #         + str(binary_array[i][k - f : k])
        #         + "\t"
        #         + str(binary_sign[i])
        #         + " "
        #         + str(binary_exponent[i])
        #         + " "
        #         + str(binary_significand[i][0:1])
        #         + "."
        #         + str(binary_significand[i][1:])
        #         + "\t"
        #         + str(special_tags[i])
        #     )
        #     print()

        return sign, exponent, significand, special_tags

    def data_compact(self, sign, exponent, mantissa):
        if self.D_format.dtype == "f16":
            return np.array(
                (sign << 15) | ((exponent & 0x1F) << 10) | (mantissa & 0x3FF)
            ).astype(np.uint16)
        elif self.D_format.dtype == "f32":
            # print(f"D_data: {sign} {exponent} {mantissa}")
            return np.array(
                (np.uint32(sign) << 31)
                | ((exponent & 0xFF) << 23)
                | (np.uint32(mantissa) & 0x7FFFFF)
            ).astype(np.uint32)
        else:
            raise ValueError(f"Unsupported dtype {self.D_dtype} for compacting")


def symbolic_to_binary(value, dtype="f16"):
    if dtype not in ["f16", "f32"]:
        raise ValueError("Unsupported dtype for symbolic conversion")

    if dtype == "f16":
        uint_type = np.uint16
        exp_bits, mant_bits = 5, 10
        bias = 15
    else:  # f32
        uint_type = np.uint32
        exp_bits, mant_bits = 8, 23
        bias = 127

    if value == "zero":
        return uint_type(0)
    elif value == "(-0.0)":
        return uint_type(1 << (exp_bits + mant_bits))
    elif value == "one":
        return uint_type((bias << mant_bits) | 0)
    elif value == "minustwo":
        return uint_type((1 << (exp_bits + mant_bits)) | (bias + 1) << mant_bits)
    elif value == "two":
        return uint_type((bias + 1) << mant_bits)
    elif value == "half":
        return uint_type((bias - 1) << mant_bits)
    elif value == "minusone":
        return uint_type((1 << (exp_bits + mant_bits)) | (bias << mant_bits))
    elif value == "aboveone":
        return uint_type((bias << mant_bits) | (1 << (mant_bits - 1)))
    elif value == "minsubnormal":
        return uint_type(1)
    elif value == "inf" or value == "INF":
        return uint_type(((2**exp_bits - 1) << mant_bits))
    elif value == "qnan":
        return uint_type(((2**exp_bits - 1) << mant_bits) | (1 << (mant_bits - 1)))
    elif value == "snan":
        return uint_type(((2**exp_bits - 1) << mant_bits) | 1)
    elif value.startswith("math.ldexp"):
        exp = int(value.split(",")[1].strip(" )"))
        mant = 1 << mant_bits
        adjusted_exp = bias + exp
        if adjusted_exp <= 0:
            mant = 1 << (mant_bits + adjusted_exp - 1)
            adjusted_exp = 0
        elif adjusted_exp >= (2**exp_bits - 1):
            return uint_type(((2**exp_bits - 1) << mant_bits))
        return uint_type(adjusted_exp << mant_bits)
    else:
        raise ValueError(f"Unsupported symbolic value: {value}")


def expected_to_f32(expected):
    if expected == "zero":
        return 0.0
    elif expected == "minustwo":
        return -2.0
    elif expected == "two":
        return 2.0
    elif expected == "one":
        return 1.0
    elif expected == "minsubnormal":
        return np.float32(2**-24)
    elif expected.startswith("math.ldexp"):
        exp = int(expected.split(",")[1].strip(" )"))
        return np.float32(math.ldexp(1.0, exp))
    elif "+" in expected or "-" in expected:
        parts = expected.replace("-", "+-").split("+")
        result = 0.0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("math.ldexp"):
                exp = int(part.split(",")[1].strip(" )"))
                result += math.ldexp(1.0, exp)
            else:
                result += float(part)
        return np.float32(result)
    elif expected == "(np.nan)":
        return np.nan
    else:
        raise ValueError(f"Unsupported expected value: {expected}")
