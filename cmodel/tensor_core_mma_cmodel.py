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
        self.chunk_size = 256 / bit_length[A_dtype]
        if C_dtype == "f16":
            self.RoundingMode = RoundingMode.RNE
        else:
            self.RoundingMode = RoundingMode.RTZ

        if A_dtype in ("e5m2", "e4m3"):
            self.accumulator_fraction_width = 25
            # self.accumulator_fraction_width = 13
        else:
            self.accumulator_fraction_width = 25

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
        # print(A, B, C)
        # 递归终止条件：当矩阵尺寸小于分块阈值时直接计算
        # if A.shape[0] <= self.chunk_size:
        #  print(f"C: {C}")
        C_compact_list = None
        k = int(self.chunk_size)
        if A.shape[0] <= self.chunk_size:
            result = self.single_step_mma(
                A[:k], B[:k], C, C_compact_list, need_format=True
            )

        else:

            C_compact_list = self.single_step_mma(
                A[:k], B[:k], C, C_compact_list, need_format=False
            )
            # print(C_compact_list)
            result = self.single_step_mma(
                A[k:], B[k:], C, C_compact_list, need_format=True
            )
        return np.array(result).reshape(C.shape)

        # return np.array(result).reshape(C.shape)

        # 分块递归逻辑：C 的输出作为下一段的输入
        # 递归处理前 k 行，生成更新后的 C_updated
        # C_updated = self.recursive_mma_accumulate(A[:k], B[:k], C, need_format=False)
        # print("=========================C_updated================================")
        # print(C_updated)

        # return

    def single_step_mma(
        self, A: np.array, B: np.array, C: np.array, C_compact_list, need_format=False
    ):  # -> INT:
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
        #  print(f"need_format: {need_format}")
        #  print(C)
        #  print(C_compact_list)
        #  print(C.shape)
        if C_compact_list != None:
            # print(np.shape(C_compact_list))
            C_sign = np.array(C_compact_list[0]).reshape(C.shape)
            # print(f"C_sign:{C_sign}")
            C_exponent = np.array(C_compact_list[1]).reshape(C.shape)
            C_significand = np.array(C_compact_list[2]).reshape(C.shape)
            C_special_tag = np.array(C_compact_list[3]).reshape(C.shape)
            # print(C_special_tag)
        else:
            C_sign, C_exponent, C_significand, C_special_tag = (
                self.extract_sign_exponent_mantissa(C, self.C_format.dtype)
            )
            C_point_gap = self.accumulator_fraction_width - self.C_format.mant_bits
            C_significand = (
                C_significand.astype(np.uint32) << C_point_gap
                if C_point_gap > 0
                else C_significand.astype(np.uint32) >> (-C_point_gap)
            )

        #  print(f"C_significand: {C_significand}")

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
        partial_sign = product_signs

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

        partial_exponent[normal_mask] = combined_exponent
        partial_significand[normal_mask] = aligned_significand

        # C + A * B
        # 1. point adjustment

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
        # print(f"C_special_tag: {C_special_tag}")
        #  print(f"partial_significand: {partial_significand}")
        return self.stage2_add(
            partial_sign,
            partial_exponent,
            partial_significand,
            partial_special,
            need_format,
        )

    def stage2_add(
        self,
        partial_sign,
        partial_exponent,
        partial_significand,
        partial_special,
        need_format,
    ):
        # print(partial_exponent)
        # print(partial_significand)
        # print(partial_special)
        sum_special_tag = np.full_like(1, None, dtype=object)

        # print(partial_special)
        if np.any(partial_special == Special_tag_enum.NaN):
            sum_special_tag = Special_tag_enum.NaN
        else:
            # print(partial_special)
            # print(partial_sign)

            # set sum = NaN, if  exists INF - INF.
            neg_inf_mask = (partial_sign == 1) & (
                partial_special == Special_tag_enum.INF
            )
            partial_special[neg_inf_mask] = Special_tag_enum.NEG_INF
            # print(partial_special)
            if np.any(partial_special == Special_tag_enum.INF) and np.any(
                partial_special == Special_tag_enum.NEG_INF
            ):
                sum_special_tag = Special_tag_enum.NaN
                # pass
            elif np.any(partial_special == Special_tag_enum.INF):
                sum_special_tag = Special_tag_enum.INF
            elif np.any(partial_special == Special_tag_enum.NEG_INF):
                sum_special_tag = Special_tag_enum.NEG_INF
        # print(sum_special_tag)
        # print(sum_special_tag)
        if sum_special_tag != None:
            if need_format:
                # print(sum_special_tag)
                if sum_special_tag == Special_tag_enum.INF:
                    return self.data_compact(0, 0xFF, 0)
                if sum_special_tag == Special_tag_enum.NEG_INF:
                    return self.data_compact(1, 0xFF, 0)
                if sum_special_tag == Special_tag_enum.NaN:
                    return self.data_compact(0, 0xFF, 0x7FFFFF)
            else:
                if sum_special_tag == Special_tag_enum.INF:
                    return [
                        np.array(0).astype(np.int32),
                        partial_exponent[0],
                        partial_significand[0],
                        sum_special_tag,
                    ]
                if sum_special_tag == Special_tag_enum.NEG_INF:
                    return [
                        np.array(1).astype(np.int32),
                        partial_exponent[0],
                        partial_significand[0],
                        sum_special_tag,
                    ]
                if sum_special_tag == Special_tag_enum.NaN:
                    return [
                        np.array(0).astype(np.int32),
                        partial_exponent[0],
                        partial_significand[0],
                        sum_special_tag,
                    ]
        else:
            # align all partials to max exponent
            max_exp = np.max(partial_exponent[partial_special == None])
            # print(f"max_exp:{max_exp}")
            # print(f"partial_sign: {partial_sign}")
            # print(f"partial_significands: {partial_significand}")
            aligned_significands = []
            for i in range(len(partial_significand)):
                # if partial_special[i] != None:
                #     continue
                exp_diff = max_exp - partial_exponent[i]
                if exp_diff > 0:
                    # sticky_bit_mask = np.int32(2 ** (exp_diff) - 1)
                    # sticky_bit = (
                    #     np.int32(1)
                    #     if np.bitwise_and(partial_significand[i], sticky_bit_mask) > 0
                    #     else np.int32(0)
                    # )
                    aligned = partial_significand[i] >> exp_diff
                    # aligned = aligned | sticky_bit
                else:
                    aligned = partial_significand[i]
                aligned_significands.append(aligned)
            # i = len(partial_significand) - 1
            # exp_diff = max_exp - partial_exponent[i]
            # if exp_diff > 0:
            #     if need_format == True:
            #         sticky_bit_mask = np.int32(2 ** (exp_diff) - 1)
            #         sticky_bit = (
            #             np.int32(1)
            #             if np.bitwise_and(partial_significand[i], sticky_bit_mask) > 0
            #             else np.int32(0)
            #         )
            #         aligned = partial_significand[i] >> exp_diff
            #         aligned = aligned | sticky_bit
            #     else:
            #         aligned = partial_significand[i] >> exp_diff
            # else:
            #     aligned = partial_significand[i]
            # aligned_significands.append(aligned)
            # print(f"aligned_significands: {aligned_significands}")
            # print(f"aligned_significands: {aligned_significands[3]: 08x}")
            # print(type(aligned_significands[0]))
            sum_significand = sum((-1) ** partial_sign * aligned_significands)
            sum_exponent = max_exp

            # if sum_significand == 0:
            #     return self.stage3_normalization_rounding(
            #         0, 0, 0, Special_tag_enum.Zero
            #     )

            sum_sign = 1 if sum_significand < 0 else 0
            sum_significand = abs(sum_significand)
            # print(sum_special_tag)
            # print(f"sum_exponent: {sum_exponent}")
            # print(f"sum_significand: {sum_significand}")
            if need_format:
                return self.stage3_normalization_rounding(
                    sum_sign, sum_exponent, sum_significand, sum_special_tag
                )
            else:
                significand_width = int(sum_significand).bit_length()
                gap = significand_width - (self.accumulator_fraction_width + 1)
                # print(gap)
                # if gap > 0:
                #     sum_exponent += gap
                #     sum_significand = sum_significand >> gap

                # print(sum_sign, sum_exponent, f"{sum_significand:08x}", sum_special_tag)
                C_compact_list_result = [
                    sum_sign,
                    sum_exponent,
                    sum_significand,
                    sum_special_tag,
                ]
                #  print(f"sum_significand:{sum_significand:030b}")
                # print("C_compact_list_result")
                # print(C_compact_list_result)
                return C_compact_list_result

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
        # significand >>= shift_amount
        # exponent += shift_amount
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

        # if special_tag == Special_tag_enum.Zero:
        #     return self.data_compact(0, 0, 0)
        # if special_tag == Special_tag_enum.INF:
        #     return self.data_compact(0, 0xFF, 0)
        # if special_tag == Special_tag_enum.NEG_INF:
        #     return self.data_compact(1, 0xFF, 0)
        # if special_tag == Special_tag_enum.NaN:
        #     return self.data_compact(0, 0xFF, 0x7FFFFF)

        # significand = int(significand)
        # Normalization:
        significand_width = int(significand).bit_length()
        gap = significand_width - (self.accumulator_fraction_width + 1)
        # print(f"exponent: {exponent}, significand_width: {significand_width}")
        #  print(f"shift_amount:{gap}")
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
                    # if self.A_format.bit_length != 8:
                    # if True:
                    exponent, significand = self.right_shift(
                        exponent, significand, shift_amount
                    )
                    # # # NOTE: 对于 8-bit，不需要调整指数和尾数，都可被精确表示
                    # else:
                    #     significand >>= shift_amount
                    #     exponent += shift_amount

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
                    # if self.A_format.bit_length != 8:
                    # if True:
                    exponent, significand = self.right_shift(
                        exponent, significand, shift_amount
                    )
                    # else:
                    #     significand >>= shift_amount
                    #     exponent += shift_amount
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
            # print(f"rounding significand: {significand}")
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

        # 仅当RN进位溢出时会走入，但按照有限数处理也正确
        if exponent > (2 ** (self.D_format.exp_bits) - 2):
            if sign == 1:
                special_tag = Special_tag_enum.NEG_INF
                return self.data_compact(1, 0xFF, 0)
            else:
                special_tag = Special_tag_enum.INF
                return self.data_compact(0, 0xFF, 0)

        # print(f"normalization {special_tag}")
        # print(exponent, significand)
        if exponent <= 0 or significand == 0:
            special_tag = Special_tag_enum.Zero
            return self.data_compact(0, 0, 0)

        if (
            exponent == 1
            and int(significand).bit_length() < self.C_format.mant_bits + 1
        ):
            special_tag = Special_tag_enum.Denormal
            return self.data_compact(sign, 0, significand)

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

        # inf_mask = (
        #     inf_nan_mask & (sign == 0) & (mantissa == 0) if dtype != "e4m3" else False
        # )
        # neg_inf_mask = (
        #     inf_nan_mask & (sign == 1) & (mantissa == 0) if dtype != "e4m3" else False
        # )

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
        #     # print()

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
