import numpy as np

# 定义浮点类型的位宽
float_types = {
    "f32": {"sign": 1, "exp": 8, "mant": 23, "total": 32},
    "f16": {"sign": 1, "exp": 5, "mant": 10, "total": 16},
    "bf16": {"sign": 1, "exp": 8, "mant": 7, "total": 16},
    "tf32": {"sign": 1, "exp": 8, "mant": 10, "total": 32},
    "e5m2": {"sign": 1, "exp": 5, "mant": 2, "total": 8},
    "e4m3": {"sign": 1, "exp": 4, "mant": 3, "total": 8},
}

# 定义存储类型
input_type = {
    "f32": np.uint32,
    "f16": np.uint16,
    "bf16": np.uint16,
    "tf32": np.uint32,
    "e5m2": np.uint8,
    "e4m3": np.uint8,
}


class FloatRandomGenerator:
    def __init__(self, dtype):
        """初始化生成器，指定浮点类型"""
        self.dtype = dtype
        self.config = float_types[dtype]
        self.np_type = input_type[dtype]

    def _generate_random_bits(self, bits):
        """生成指定位数的随机二进制数"""
        return np.random.randint(0, 2**bits, dtype=np.uint32)

    def generate_random_float(self):
        """生成随机浮点数的二进制表示"""
        sign = self._generate_random_bits(self.config["sign"])
        exp = self._generate_random_bits(self.config["exp"])
        mant = self._generate_random_bits(self.config["mant"])
        # 组合二进制表示
        binary = (
            (sign << (self.config["exp"] + self.config["mant"]))
            | (exp << self.config["mant"])
            | mant
        )
        # 转换为对应类型
        return self.np_type(binary)

    def generate_special_values(self):
        """生成扩展的特殊值，包括零、亚正常数、正常数边界、无穷大和NaN"""
        special_values = []

        # Zero and negative zero
        special_values.append(self.create_float_value(0, 0, 0))  # zero
        special_values.append(self.create_float_value(1, 0, 0))  # negative zero

        # Subnormal numbers: min and max for both positive and negative
        mant_min_subnormal = 1  # smallest non-zero mantissa
        mant_max_subnormal = (1 << self.config["mant"]) - 1  # largest mantissa for subnormal numbers
        for sign in [0, 1]:
            # Min subnormal number
            value_min_subnormal = self.create_float_value(sign, 0, mant_min_subnormal)
            special_values.append(value_min_subnormal)
            # Max subnormal number
            value_max_subnormal = self.create_float_value(sign, 0, mant_max_subnormal)
            special_values.append(value_max_subnormal)

        # Normal numbers: min and max for both positive and negative
        exponent_min_normal = 1  # assuming exponent field value of 1 for min normal
        exponent_max_normal = (1 << self.config["exp"]) - 2  # maximum exponent for normal numbers
        mant_min_normal = 0  # smallest mantissa for normal numbers
        mant_max_normal = (1 << self.config["mant"]) - 1  # largest mantissa for normal numbers
        for sign in [0, 1]:
            # Min normal number
            value_min_normal = self.create_float_value(sign, exponent_min_normal, mant_min_normal)
            special_values.append(value_min_normal)
            # Max normal number
            value_max_normal = self.create_float_value(sign, exponent_max_normal, mant_max_normal)
            special_values.append(value_max_normal)

        # Infinity and negative infinity
        exponent_inf = (1 << self.config["exp"]) - 1  # all ones in exponent field
        for sign in [0, 1]:
            value_inf = self.create_float_value(sign, exponent_inf, 0)
            special_values.append(value_inf)

        # NaN values: one positive and one negative with minimal and maximal mantissa
        for sign in [0, 1]:
            # Minimal NaN: mantissa=1
            value_nans_min = self.create_float_value(sign, exponent_inf, 1)
            special_values.append(value_nans_min)
            # Maximal NaN: mantissa=(2^mant_bits - 1)
            value_nans_max = self.create_float_value(sign, exponent_inf, mant_max_normal)
            special_values.append(value_nans_max)

        return special_values
    
    def generate_batch(self, size):
        """生成一批随机浮点数"""
        batch = []
        for _ in range(size):
            if np.random.rand() < 0.8:  # 20% 概率生成特殊值
                special_values = self.generate_special_values()
                batch.append(np.random.choice(special_values))
            else:
                batch.append(self.generate_random_float())
        return np.array(batch, dtype=self.np_type)


# 示例使用
# def generate_test_case(dtype, size=16):
#     generator = FloatRandomGenerator(dtype)
#     batch = generator.generate_batch(size)
#     print(f"Generated {dtype} test case ({size} values):")
#     print(batch)


# # 测试各种类型
# for dtype in ["fp32", "fp16", "bf16", "tf32", "e5m2", "e4m3"]:
#     generate_test_case(dtype)
#     print("-" * 50)
