import numpy as np
import math
from fp_cvt import fpemu


def print_32bit_hex(float_val):
    integer_repr = np.frombuffer(float_val.tobytes(), dtype=np.uint32)[0]
    # print("inner_0x{:08X}".format(integer_repr))
    return "0x{:08X}".format(integer_repr)


def print_16bit_hex(half_val):
    print(half_val)
    two_byte = np.frombuffer(half_val.tobytes(), dtype=np.uint16)[0]
    # print("0x{:04X}".format(two_byte))
    return "0x{:04X}".format(two_byte)


def print_8bit_hex(char_val):
    one_byte = np.frombuffer(char_val.tobytes(), dtype=np.uint8)[0]
    # print("0x{:02X}".format(one_byte))
    return "0x{:02X}".format(one_byte)


# 定义位宽到无符号整数类型的映射
uint_dtype_map = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
}

# 已有的 bit_length 字典
bit_length = {
    "f64": 64,
    "tf32": 32,
    "f32": 32,
    "s32": 32,
    "bf16": 16,
    "f16": 16,
    "e4m3": 8,
    "e5m2": 8,
    "e2m1": 4,
    "u8": 8,
    "s8": 8,
    "u4": 4,
    "s4": 4,
    "b1": 1,
}

# 已有的 exponent_dict, mantissa_dict, np.dtype 等保持不变
exponent_dict = {"e5m2": 5, "e4m3": 4, "e2m1": 2, "f16": 5, "f32": 8, "tf32": 8, "bf16": 8}

mantissa_dict = {"e5m2": 2, "e4m3": 3, "e2m1": 1, "f16": 10, "f32": 23, "tf32": 10, "bf16": 7}

np_dtype_dict = {
    "e5m2": np.float32,
    "e4m3": np.float32,
    "e2m1": np.float32,
    "tf32": np.float32,
    "bf16": np.float32,
    "f16": np.float16,
    "f32": np.float32,
}

uint_map = {
    "tf32": np.uint32,
    "f32": np.uint32,
    "bf16": np.uint32,
    "f16": np.uint16,
    "e4m3": np.uint32,
    "e5m2": np.uint32,
    "e2m1": np.uint32,
}


class TestDataGenerator:
    def __init__(self, d_type):
        self.dtype = d_type
        self.e = exponent_dict[self.dtype]
        self.m = mantissa_dict[self.dtype]
        self.np_dtype = np_dtype_dict[self.dtype]
        self.bit_length = bit_length[self.dtype]
        # 设置目标无符号整数类型
        self.uint_dtype = uint_dtype_map[self.bit_length]

    def generate_value(self, value):
        # 生成浮点数或特殊值
        if self.dtype == "bf16":
            result = np.array((1,), dtype=np.uint16)
            value = value.reshape(
                1,
            )
            fpemu.cvt_float_to_bf16(value, result)
            float_val = result[0]
        elif self.dtype == "e4m3":
            result = np.array((1,), dtype=np.uint8)
            value = value.reshape(
                1,
            )
            fpemu.cvt_float_to_fp8(value, result, True, False)
            float_val = result[0]
        elif self.dtype == "e5m2":
            result = np.array((1,), dtype=np.uint8)
            value = value.reshape(
                1,
            )
            fpemu.cvt_float_to_fp8(value, result, False, False)
            float_val = result[0]
        elif self.dtype == "e2m1":
            result = np.array((1,), dtype=np.uint8)
            value = value.reshape(
                1,
            )
            fpemu.cvt_float_to_e2m1(value, result)
            float_val = result[0]
        else:
            float_val = value

        # 将浮点数的二进制内容转换为无符号整数
        return np.frombuffer(float_val.tobytes(), dtype=self.uint_dtype)[0]

    # 其他方法保持不变，只需确保返回值通过 generate_value 处理
    def minsubnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(1.0, -(2 ** (self.e - 1) - 2 + self.m))],
                dtype=self.np_dtype,
            )[0]
        )

    def neg_minsubnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(-1.0, -(2 ** (self.e - 1) - 2 + self.m))],
                dtype=self.np_dtype,
            )[0]
        )

    def maxsubnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(1.0 - math.ldexp(1.0, -self.e), -(2 ** (self.e - 1) - 2))],
                dtype=self.np_dtype,
            )[0]
        )

    def neg_maxsubnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(-1.0 + math.ldexp(1.0, -self.e), -(2 ** (self.e - 1) - 2))],
                dtype=self.np_dtype,
            )[0]
        )

    def half(self):
        return self.generate_value(np.array([0.5], dtype=self.np_dtype)[0])

    def neg_half(self):
        return self.generate_value(np.array([-0.5], dtype=self.np_dtype)[0])

    def minusone(self):
        return self.generate_value(np.array([-1.0], dtype=self.np_dtype)[0])

    def one(self):
        return self.generate_value(np.array([1.0], dtype=self.np_dtype)[0])

    def minustwo(self):
        return self.generate_value(np.array([-2.0], dtype=self.np_dtype)[0])

    def belowone(self):
        return self.generate_value(
            np.array([1.0 - math.ldexp(1.0, -(self.m + 1))], dtype=self.np_dtype)[0]
        )

    def above_neg_one(self):
        return self.generate_value(
            np.array([-1.0 + math.ldexp(1.0, -(self.m + 1))], dtype=self.np_dtype)[0]
        )

    def aboveone(self):
        return self.generate_value(
            np.array([1.0 + math.ldexp(1.0, -self.m)], dtype=self.np_dtype)[0]
        )

    def below_neg_one(self):
        return self.generate_value(
            np.array([-1.0 - math.ldexp(1.0, -self.m)], dtype=self.np_dtype)[0]
        )

    def zero(self):
        return self.generate_value(np.array([0.0], dtype=self.np_dtype)[0])

    def neg_zero(self):
        return self.generate_value(np.array([-0.0], dtype=self.np_dtype)[0])

    def two(self):
        return self.generate_value(np.array([2.0], dtype=self.np_dtype)[0])

    def four(self):
        return self.generate_value(np.array([4.0], dtype=self.np_dtype)[0])

    def qnan(self):
        if self.dtype == "e2m1":
            # e2m1 does not have NaN, return max normal value
            return self.maxnormal()
        return self.generate_value(np.array([np.nan], dtype=self.np_dtype)[0])

    def snan(self):
        if self.dtype == "e2m1":
            # e2m1 does not have NaN, return max normal value
            return self.maxnormal()
        shift_value = 1 << (self.e + 1)
        exponent_mask = shift_value - 1
        if self.dtype == "tf32":
            float_value = exponent_mask << (self.m + 12)
        else:
            float_value = exponent_mask << self.m
        sNaN = np.array([float_value], dtype=uint_map[self.dtype])
        # print(self.np_dtype)
        # if self.dtype == "bf16":
        #     return np.frombuffer(sNaN.tobytes(), dtype=uint_dtype_map[self.dtype])[0]
        # elif self.dtype == "e5m2" or self.dtype == "e4m3":
        #     return np.frombuffer(sNaN.tobytes(), dtype=np.uint8)[0]
        # else:
        #     return np.frombuffer(sNaN.tobytes(), dtype=self.np_dtype)[0]
        return self.generate_value(np.array([sNaN], dtype=self.np_dtype)[0])

    def inf(self):
        if self.dtype == "e2m1":
            # e2m1 does not have infinity, return max normal value
            return self.maxnormal()
        return self.generate_value(np.array([np.inf], dtype=self.np_dtype)[0])

    def neg_inf(self):
        if self.dtype == "e2m1":
            # e2m1 does not have infinity, return negative max normal value
            return self.neg_maxnormal()
        return self.generate_value(np.array([-np.inf], dtype=self.np_dtype)[0])

    def maxnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(1.0 - math.ldexp(1.0, -self.m), 2 ** (self.e - 1) - 1)],
                dtype=self.np_dtype,
            )[0]
        )

    def neg_maxnormal(self):
        return self.generate_value(
            np.array(
                [-math.ldexp(1.0 - math.ldexp(1.0, -self.m), 2 ** (self.e - 1) - 1)],
                dtype=self.np_dtype,
            )[0]
        )

    def minnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(1.0, -(2 ** (self.e - 1) - 2))],
                dtype=self.np_dtype,
            )[0]
        )

    def neg_minnormal(self):
        return self.generate_value(
            np.array(
                [math.ldexp(-1.0, -(2 ** (self.e - 1) - 2))],
                dtype=self.np_dtype,
            )[0]
        )

    def numpify(self, value):
        if self.np_dtype == np.float16:
            return np.float16(value)
        elif self.np_dtype == np.float32:
            return np.float32(value)

    def generate(self, value):
        if "(" in value:
            value = self.generate_value(self.numpify(eval(value)))
        elif isinstance(value, str):
            value = getattr(self, value)()
        elif callable(value):
            value = value()
        else:
            raise ValueError("Invalid value type. Expected a string or a callable.")
        return value


# 测试代码
# generator = TestDataGenerator("e5m2")
# e5m2_inf = generator.inf()
# print(f"e5m2 inf: {e5m2_inf}, type: {type(e5m2_inf)}")  # 应为 uint8

# generator = TestDataGenerator("e4m3")
# e4m3_inf = generator.inf()
# print(f"e4m3 inf: {e4m3_inf}, type: {type(e4m3_inf)}")  # 应为 uint8

# generator = TestDataGenerator("bf16")
# bf16_one = generator.one()
# print(f"bf16 one: {bf16_one}, type: {type(bf16_one)}")  # 应为 uint16
