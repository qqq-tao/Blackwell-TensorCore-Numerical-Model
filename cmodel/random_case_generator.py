import numpy as np
import benchmark_values

# 定义浮点类型的位宽
float_types = {
    "f32": {"sign": 1, "exp": 8, "mant": 23, "total": 32},
    "f16": {"sign": 1, "exp": 5, "mant": 10, "total": 16},
    "bf16": {"sign": 1, "exp": 8, "mant": 7, "total": 16},
    "tf32": {"sign": 1, "exp": 8, "mant": 10, "total": 32},
    "e5m2": {"sign": 1, "exp": 5, "mant": 2, "total": 8},
    "e4m3": {"sign": 1, "exp": 4, "mant": 3, "total": 8},
    "e2m1": {"sign": 1, "exp": 2, "mant": 1, "total": 4},
}

# 定义存储类型
input_type = {
    "f32": np.uint32,
    "f16": np.uint16,
    "bf16": np.uint16,
    "tf32": np.uint32,
    "e5m2": np.uint8,
    "e4m3": np.uint8,
    "e3m2": np.uint8,
    "e2m3": np.uint8,
    "e2m1": np.uint8,
}


class FloatRandomGenerator:
    def __init__(self, dtype_A, dtype_B, dtype_C, stype):
        """初始化生成器，指定浮点类型"""
        self.dtype_A = dtype_A
        self.dtype_B = dtype_B
        self.dtype_C = dtype_C

        self.config = float_types
        self.np_type_A = input_type[dtype_A]
        self.np_type_B = input_type[dtype_B]
        self.np_type_C = input_type[dtype_C]

    def _generate_random_bits(self, bits, mode="uniform"):
        """
        生成指定位数的随机整数值 (用于填充符号、指数、尾数位)。

        参数:
            bits (int): 需要生成的随机位数。
            mode (str): 生成模式:
                "uniform": 在 [0, 2^bits - 1] 范围内均匀分布。
                "near_upper_bound": 生成靠近上限 (2^bits - 1) 的值。
                "near_lower_bound": 生成靠近下限 (0) 的值。

        返回:
            np.uint32 or np.uint64: 生成的随机整数。如果位数 <= 32 返回 uint32，否则 uint64。
                                      注意：浮点数各部分位数通常远小于64。
        """
        if bits <= 0:
            return np.uint32(0)  # 位数为0或负数，返回0

        # 计算该位数能表示的最大值 (2^bits - 1)
        # 使用 uint64 避免计算过程中溢出 (虽然对标准浮点格式不太可能)
        max_val = (np.uint64(1) << bits) - 1

        if mode == "uniform":
            # 生成 [0, max_val] 范围内的均匀随机整数
            # np.random.randint 上界不包含，所以用 max_val + 1
            if max_val < np.iinfo(np.uint32).max:
                # 如果最大值在 uint32 范围内，直接生成 uint32
                return np.random.randint(0, max_val + 1, dtype=np.uint32)
            else:
                # 否则生成 uint64 (理论上，对浮点数分量用不到)
                val = np.random.randint(0, max_val + 1, dtype=np.uint64)
                # 如果位数<=32，转回uint32，否则返回uint64
                return np.uint32(val) if bits <= 32 else val

        # --- 非均匀分布 ---
        u = np.random.rand()  # 生成 [0, 1) 范围的浮点数

        if mode == "near_upper_bound":
            k = 0.5  # k < 1 使 u^k 偏向 1
            powered = u**k
            # 将接近 1 的 powered 映射到接近 max_val 的整数
            value = powered * float(max_val)
        elif mode == "near_lower_bound":
            k = 2.0  # k > 1 使 u^k 偏向 0
            powered = u**k
            # 将接近 0 的 powered 映射到接近 0 的整数
            value = powered * float(max_val)
        else:
            # 未知模式，回退到均匀分布
            print(f"警告: 未知的随机位生成模式 '{mode}'，使用 'uniform'")
            return self._generate_random_bits(bits, mode="uniform")

        # 四舍五入并确保结果在 [0, max_val] 范围内
        value_int = round(value)
        value_int = max(0, min(value_int, max_val))

        # 根据位数返回合适的整数类型
        return np.uint32(value_int) if bits <= 32 else np.uint64(value_int)

    def generate_random_float(self, dtype):
        """生成随机浮点数的二进制表示"""
        sign = self._generate_random_bits(self.config[dtype]["sign"])
        # exp = self.exp_global
        exp = self._generate_random_bits(self.config[dtype]["exp"])
        mant = self._generate_random_bits(self.config[dtype]["mant"])
        # 组合二进制表示
        binary = (
            (sign << (self.config[dtype]["exp"] + self.config[dtype]["mant"]))
            | (exp << self.config[dtype]["mant"])
            | mant
        )
        # 转换为对应类型
        return input_type[dtype](binary)

    def generate_special_values(self, dtype):
        """生成特殊值列表，与TestDataGenerator方法保持一致"""
        generator = benchmark_values.TestDataGenerator(dtype)

        special_values = [
            # 零值
            generator.zero(),  # 正零
            generator.neg_zero(),  # 负零
            # 特殊值
            generator.inf(),  # 正无穷
            generator.neg_inf(),  # 负无穷
            generator.qnan(),  # 静默NaN
            generator.snan(),  # 信号NaN
            # 单位值
            generator.one(),  # 1.0
            generator.minusone(),  # -1.0
            generator.two(),  # 2.0
            generator.minustwo(),  # -2.0
            generator.half(),  # 0.5
            generator.neg_half(),  # -0.5
            generator.four(),  # 4.0
            # 边界值
            generator.maxnormal(),  # 最大正规数
            generator.neg_maxnormal(),  # 最小正规数(负)
            generator.minnormal(),  # 最小正规数
            generator.neg_minnormal(),  # 最大正规数(负)
            # 亚正规数
            generator.maxsubnormal(),  # 最大亚正规数
            generator.neg_maxsubnormal(),  # 最小亚正规数(负)
            generator.minsubnormal(),  # 最小亚正规数
            generator.neg_minsubnormal(),  # 最大亚正规数(负)
            # 接近1的边界值
            generator.belowone(),  # 略小于1.0
            generator.aboveone(),  # 略大于1.0
            generator.above_neg_one(),  # 略大于-1.0
            generator.below_neg_one(),  # 略小于-1.0
        ]

        return special_values

    def generate_finite_special_values(self, dtype):
        """生成特殊值列表，与TestDataGenerator方法保持一致"""
        generator = benchmark_values.TestDataGenerator(dtype)

        special_values = [
            # 零值
            generator.zero(),  # 正零
            generator.neg_zero(),  # 负零
            # 单位值
            generator.one(),  # 1.0
            generator.minusone(),  # -1.0
            generator.two(),  # 2.0
            generator.minustwo(),  # -2.0
            generator.half(),  # 0.5
            generator.neg_half(),  # -0.5
            generator.four(),  # 4.0
            # 边界值
            generator.maxnormal(),  # 最大正规数
            generator.neg_maxnormal(),  # 最小正规数(负)
            generator.minnormal(),  # 最小正规数
            generator.neg_minnormal(),  # 最大正规数(负)
            # 亚正规数
            generator.maxsubnormal(),  # 最大亚正规数
            generator.neg_maxsubnormal(),  # 最小亚正规数(负)
            generator.minsubnormal(),  # 最小亚正规数
            generator.neg_minsubnormal(),  # 最大亚正规数(负)
            # 接近1的边界值
            generator.belowone(),  # 略小于1.0
            generator.aboveone(),  # 略大于1.0
            generator.above_neg_one(),  # 略大于-1.0
            generator.below_neg_one(),  # 略小于-1.0
        ]

        return special_values

    def create_float_value(self, dtype, sign, exponent_field_value, mantissa_value):
        """创建浮点数的值，根据指定的符号、指数字段值和尾数值"""
        return input_type[dtype](
            (sign << (self.config[dtype]["exp"] + self.config[dtype]["mant"]))
            | (exponent_field_value << self.config[dtype]["mant"])
            | mantissa_value
        )

    def generate_batch(self, k, mode=None):
        """生成测试用例，随机选择 mode"""
        # 随机选择 mode：70% 概率选择 random，30% 概率选择 special_multiplication
        A_data_generator = benchmark_values.TestDataGenerator(self.dtype_A)
        B_data_generator = benchmark_values.TestDataGenerator(self.dtype_B)
        C_data_generator = benchmark_values.TestDataGenerator(self.dtype_C)

        available_modes = [
            "random",
            "special_value_multiplication",
            "dot_product_boundary",
            "denormal_round_up_to_normal_value",
        ]
        mode_probabilities = [0.6, 0.05, 0.3, 0.05]  # 概率可调
        # mode_probabilities = [0, 0, 0, 1]  # 概率可调
        if not mode:
            mode = np.random.choice(available_modes, p=mode_probabilities)
        elif mode not in available_modes and mode not in [
            "normalization_case_0",
            "normalization_case_1",
            "normalization_case_2",
            "normalization_case_3",
        ]:
            print(f"警告: 指定模式 '{mode}' 未完全实现或无法识别。使用 'random' 代替。")
            mode = "random"

        # --- 初始化输出数组 ---
        A = np.zeros(k, dtype=self.np_type_A)
        B = np.zeros(k, dtype=self.np_type_B)
        C = np.zeros(1, dtype=self.np_type_C)  # C 始终是 1 个元素

        # --- 根据模式生成数据 ---
        if mode == "dot_product_boundary":
            # 调用新的点积边界生成函数，它返回 A[k], B[k], C[1]
            A, B, C = self.generate_dot_product_boundary_batch(k)  # 直接赋值

        elif mode == "special_value_multiplication":
            # 从特殊值中随机选择
            special_values_A = self.generate_special_values(self.dtype_A)
            special_values_B = self.generate_special_values(self.dtype_B)
            special_values_C = self.generate_special_values(self.dtype_C)
            A = np.random.choice(special_values_A, size=k)
            B = np.random.choice(special_values_B, size=k)
            C = np.random.choice(special_values_C, size=k)

        elif mode == "denormal_round_up_to_normal_value":
            if self.dtype_A == "f16" and self.dtype_C == "f16":
                A[0] = A_data_generator.generate("2.0 - math.ldexp(1.0, -10 ) ")
                B[0] = B_data_generator.generate("math.ldexp(1.0, -15)")
            else:
                A = np.array(
                    [self.generate_random_float(self.dtype_A) for _ in range(k)],
                    dtype=self.np_type_A,
                )
                B = np.array(
                    [self.generate_random_float(self.dtype_B) for _ in range(k)],
                    dtype=self.np_type_B,
                )
                C = np.array(
                    [self.generate_random_float(self.dtype_C)],
                    dtype=self.np_type_C,
                )

        elif mode == "random":
            # 预生成特殊值列表（避免重复生成）
            special_values_A = self.generate_finite_special_values(self.dtype_A)
            special_values_B = self.generate_finite_special_values(self.dtype_B)
            special_values_C = self.generate_finite_special_values(self.dtype_C)

            A = []
            B = []
            for _ in range(k):
                # 生成 A: 5% 概率用特殊值，否则用随机值
                if np.random.rand() < 0.1:
                    A.append(np.random.choice(special_values_A))
                else:
                    A.append(self.generate_random_float(self.dtype_A))

                # 生成 B: 5% 概率用特殊值，否则用随机值
                if np.random.rand() < 0.1:
                    B.append(np.random.choice(special_values_B))
                else:
                    B.append(self.generate_random_float(self.dtype_B))

            A = np.array(A, dtype=self.np_type_A)
            B = np.array(B, dtype=self.np_type_B)
            if np.random.rand() < 0.1:
                C[0] = np.random.choice(special_values_C)
            else:
                C[0] = self.generate_random_float(self.dtype_C)

        if not isinstance(C, np.ndarray) or C.dtype != self.np_type_C or C.size != 1:
            # print(
            #     f"警告: 校正 C 的形状/类型。原始: {type(C)}, 大小: {getattr(C, 'size', 'N/A')}"
            # )
            if isinstance(C, np.ndarray) and C.size > 0:  # 如果是数组但大小不对
                C = np.array([C[0]], dtype=self.np_type_C)  # 取第一个元素
            elif not isinstance(C, np.ndarray) and C is not None:  # 如果是标量
                C = np.array([C], dtype=self.np_type_C)  # 包装成数组
            else:  # 其他异常情况，生成默认随机值
                C = np.array(
                    [self.generate_random_float(self.dtype_C)], dtype=self.np_type_C
                )

        return A, B, C

    def _get_exp_pair_for_sum(
        self,
        target_dot_product_exp_val,
        bias_A,
        bias_B,
        max_norm_exp_A,
        max_norm_exp_B,
        allow_subnormal=False,
    ):
        """
        (辅助函数) 根据乘积的目标指数值 (`target_dot_product_exp_val`)，
        尝试找到一对 A 和 B 的指数字段值 (`exp_a`, `exp_b`)，
        使得它们的实际指数值之和接近目标值。
        (val_a + val_b ≈ target_dot_product_exp_val)

        参数:
            target_dot_product_exp_val (float): A*B 的目标指数值 (已去除偏移)。
            bias_A, bias_B (int): A 和 B 的指数偏移量。
            max_norm_exp_A, max_norm_exp_B (int): A 和 B 的最大正规数指数字段值。
            allow_subnormal (bool): 是否允许生成的指数为 0 (次正规数)。

        返回:
            tuple[int, int]: 计算出的 (exp_a_field, exp_b_field)。结果会被钳位到有效范围。
        """
        min_norm_exp = 1  # 正规数的最小指数字段值通常为 1

        # 策略：先随机生成一个 A 的指数 (通常是正规数)，然后计算 B 需要的指数，最后钳位 B 的指数。

        # 确定 A 的指数范围 (优先正规数)
        exp_a_min_limit = min_norm_exp if (max_norm_exp_A >= min_norm_exp) else 0
        exp_a_max_limit = max_norm_exp_A + 1  # randint 上界不包含

        if exp_a_min_limit >= exp_a_max_limit:  # A 不支持正规数
            exp_a = 0
        else:
            exp_a = np.random.randint(
                exp_a_min_limit, exp_a_max_limit
            )  # 随机选一个 A 的指数

        val_a = exp_a - bias_A  # A 的实际指数值

        # 计算 B 需要的实际指数值 val_b
        # target_dot_product_exp_val ≈ val_a + val_b => val_b ≈ target - val_a
        req_val_b = target_dot_product_exp_val - val_a

        # 将 B 需要的实际指数值转换回指数字段值
        # req_val_b = req_exp_b_field - bias_B => req_exp_b_field = req_val_b + bias_B
        req_exp_b_field = int(round(req_val_b + bias_B))  # 四舍五入取整

        # --- 钳位 B 的指数字段值 ---
        lower_bound_b = 0 if allow_subnormal else min_norm_exp
        # 确保下界不超过 B 可能的最大正规指数；如果 B 不支持正规数，下界只能是0
        if max_norm_exp_B < min_norm_exp:
            lower_bound_b = 0
        upper_bound_b = max_norm_exp_B

        exp_b = max(lower_bound_b, min(req_exp_b_field, upper_bound_b))
        # --- 钳位结束 ---

        # --- 修正 A 的指数 (如果需要) ---
        # 如果不允许次正规数，且计算出的 B 被迫钳位到最小正规数以下，
        # 或者 B 本身就不支持正规数，那么 B 只能是0。
        # 这种情况下，如果 A 支持正规数，也应避免 A 是次正规数 (除非明确测试 A(次)*B(次))
        if (
            not allow_subnormal
            and exp_b < min_norm_exp
            and max_norm_exp_B >= min_norm_exp
        ):
            exp_b = min_norm_exp  # 强制 B 为最小正规数 (如果可能)

        # 如果 A 本身就不支持正规数，强制为0
        if max_norm_exp_A < min_norm_exp:
            exp_a = 0
        # 如果 B 本身就不支持正规数，强制为0
        if max_norm_exp_B < min_norm_exp:
            exp_b = 0

        return exp_a, exp_b

    def _calculate_bias(self, dtype):
        """计算指数偏移量 (bias)。"""
        exp_bits = self.config[dtype]["exp"]
        return (1 << (exp_bits - 1)) - 1

    def generate_dot_product_boundary_batch(self, size, scenario=None):
        """
        生成向量 A[size], B[size] 和标量 C[0]，用于测试点积运算：
        D = Normalize&Rounding(dot(A, B) + C)
        (使用优化后的 C 值选择逻辑)
        """
        batch_A = []
        batch_B = []

        # ... (选择 dot_product_scenarios 和 chosen_scenario 的逻辑不变) ...
        dot_product_scenarios = [
            "SUM_large_positive",
            "SUM_large_negative",
            "SUM_cancel_near_zero",
            "SUM_small_terms_positive",
            "SUM_small_terms_mixed",
            "SUM_mixed_magnitudes",
        ]
        if scenario is None or scenario not in dot_product_scenarios:
            chosen_scenario = np.random.choice(dot_product_scenarios)
        else:
            chosen_scenario = scenario

        # --- 1. 生成标量输入 C (优化后) ---
        c_val = None
        c_special_choices = self.generate_special_values(self.dtype_C)

        # 如果 generate_special_values 失败或返回空列表
        if not c_special_choices:
            print(f"警告: 无法为 {self.dtype_C} 生成特殊值列表。将使用随机 C 值。")
            c_val = self.generate_random_float(self.dtype_C)
        else:
            # *** 优化点：直接选择有限值的索引 ***
            # 假设上面注释中标注的索引是正确的且固定的
            finite_indices = [
                0,
                1,  # Zeros
                6,
                7,
                8,
                9,
                10,
                11,
                12,  # Units etc.
                13,
                14,
                15,
                16,  # Normals
                17,
                18,
                19,
                20,  # Subnormals
                21,
                22,
                23,
                24,  # Near ones
            ]
            # 确保索引在实际列表长度范围内
            valid_finite_indices = [
                idx for idx in finite_indices if idx < len(c_special_choices)
            ]

            if not valid_finite_indices:
                print(
                    f"警告: 在 {self.dtype_C} 的特殊值列表中找不到有效的有限值索引。将使用随机 C 值。"
                )
                c_val = self.generate_random_float(self.dtype_C)
            else:
                # 策略：70% 概率从有限特殊值中选，30% 概率用随机正规数
                if np.random.rand() < 0.7:
                    chosen_index = np.random.choice(valid_finite_indices)
                    c_val = c_special_choices[chosen_index]
                else:
                    c_val = self.generate_random_float(self.dtype_C)  # 生成随机值

        # 包装成 1 元素的 Numpy 数组返回
        c_to_return = np.array([c_val], dtype=self.np_type_C)
        # --- C 生成结束 ---

        # --- 2. 根据选定场景生成向量 A 和 B ---
        # ... (这部分逻辑保持不变，根据 chosen_scenario 生成 A[i], B[i]) ...
        cfg_A = self.config[self.dtype_A]
        exp_A_bits, mant_A_bits = cfg_A["exp"], cfg_A["mant"]
        bias_A = self._calculate_bias(self.dtype_A)
        max_exp_A_field = (1 << exp_A_bits) - 1
        max_normal_exp_A_field = max(0, max_exp_A_field - 1)
        min_normal_exp_A_field = 1 if max_normal_exp_A_field >= 1 else 0
        cfg_B = self.config[self.dtype_B]
        exp_B_bits, mant_B_bits = cfg_B["exp"], cfg_B["mant"]
        bias_B = self._calculate_bias(self.dtype_B)
        max_exp_B_field = (1 << exp_B_bits) - 1
        max_normal_exp_B_field = max(0, max_exp_B_field - 1)
        min_normal_exp_B_field = 1 if max_normal_exp_B_field >= 1 else 0

        max_prod_exp_val = float("-inf")
        min_prod_exp_val = float("inf")
        if (
            max_normal_exp_A_field >= min_normal_exp_A_field
            and max_normal_exp_B_field >= min_normal_exp_B_field
        ):  # 确保正规数存在
            max_prod_exp_val = (max_normal_exp_A_field - bias_A) + (
                max_normal_exp_B_field - bias_B
            )
            min_prod_exp_val = (min_normal_exp_A_field - bias_A) + (
                min_normal_exp_B_field - bias_B
            )

        for i in range(size):
            sign_a, exp_a_field, mant_a = 0, 0, 0
            sign_b, exp_b_field, mant_b = 0, 0, 0

            if chosen_scenario == "SUM_large_positive":
                sign_a, sign_b = 0, 0
                target_exp_sum = (
                    max_prod_exp_val
                    - np.random.randint(0, max(1, exp_A_bits // 2 + exp_B_bits // 2))
                    if max_prod_exp_val > float("-inf")
                    else -bias_A - bias_B
                )
                exp_a_field, exp_b_field = self._get_exp_pair_for_sum(
                    target_exp_sum,
                    bias_A,
                    bias_B,
                    max_normal_exp_A_field,
                    max_normal_exp_B_field,
                )
                mant_a = self._generate_random_bits(mant_A_bits, mode="uniform")
                mant_b = self._generate_random_bits(mant_B_bits, mode="uniform")
            elif chosen_scenario == "SUM_large_negative":
                sign_a, sign_b = 0, 1
                target_exp_sum = (
                    max_prod_exp_val
                    - np.random.randint(0, max(1, exp_A_bits // 2 + exp_B_bits // 2))
                    if max_prod_exp_val > float("-inf")
                    else -bias_A - bias_B
                )
                exp_a_field, exp_b_field = self._get_exp_pair_for_sum(
                    target_exp_sum,
                    bias_A,
                    bias_B,
                    max_normal_exp_A_field,
                    max_normal_exp_B_field,
                )
                mant_a = self._generate_random_bits(mant_A_bits, mode="uniform")
                mant_b = self._generate_random_bits(mant_B_bits, mode="uniform")
            elif chosen_scenario == "SUM_cancel_near_zero":
                sign_a = i % 2
                sign_b = 0
                target_exp_sum = (
                    (max_prod_exp_val + min_prod_exp_val) / 2 + np.random.randint(-2, 3)
                    if max_prod_exp_val > float("-inf")
                    and min_prod_exp_val < float("inf")
                    else -bias_A - bias_B
                )
                exp_a_field, exp_b_field = self._get_exp_pair_for_sum(
                    target_exp_sum,
                    bias_A,
                    bias_B,
                    max_normal_exp_A_field,
                    max_normal_exp_B_field,
                )
                mant_a = self._generate_random_bits(mant_A_bits, mode="uniform")
                mant_b = self._generate_random_bits(mant_B_bits, mode="uniform")
            elif chosen_scenario == "SUM_small_terms_positive":
                sign_a, sign_b = 0, 0
                target_exp_sum = (
                    min_prod_exp_val + np.random.randint(0, 5)
                    if min_prod_exp_val < float("inf")
                    else -bias_A - bias_B
                )
                exp_a_field, exp_b_field = self._get_exp_pair_for_sum(
                    target_exp_sum,
                    bias_A,
                    bias_B,
                    max_normal_exp_A_field,
                    max_normal_exp_B_field,
                    allow_subnormal=True,
                )
                mant_a = self._generate_random_bits(mant_A_bits, mode="uniform")
                if exp_a_field == 0 and mant_A_bits > 0:
                    mant_a |= 1
                mant_b = self._generate_random_bits(mant_B_bits, mode="uniform")
                if exp_b_field == 0 and mant_B_bits > 0:
                    mant_b |= 1
            elif chosen_scenario == "SUM_small_terms_mixed":
                sign_a = np.random.choice([0, 1])
                sign_b = np.random.choice([0, 1])
                target_exp_sum = (
                    min_prod_exp_val + np.random.randint(0, 5)
                    if min_prod_exp_val < float("inf")
                    else -bias_A - bias_B
                )
                exp_a_field, exp_b_field = self._get_exp_pair_for_sum(
                    target_exp_sum,
                    bias_A,
                    bias_B,
                    max_normal_exp_A_field,
                    max_normal_exp_B_field,
                    allow_subnormal=True,
                )
                mant_a = self._generate_random_bits(mant_A_bits, mode="uniform")
                if exp_a_field == 0 and mant_A_bits > 0:
                    mant_a |= 1
                mant_b = self._generate_random_bits(mant_B_bits, mode="uniform")
                if exp_b_field == 0 and mant_B_bits > 0:
                    mant_b |= 1
            elif chosen_scenario == "SUM_mixed_magnitudes":
                sign_a = np.random.choice([0, 1])
                sign_b = np.random.choice([0, 1])
                if np.random.rand() < 0.5 and max_prod_exp_val > float("-inf"):
                    target_exp_sum = max_prod_exp_val - np.random.randint(
                        0, max(1, exp_A_bits // 2 + exp_B_bits // 2)
                    )
                    allow_sub = False
                elif min_prod_exp_val < float("inf"):
                    target_exp_sum = min_prod_exp_val + np.random.randint(0, 5)
                    allow_sub = True
                else:
                    target_exp_sum = -bias_A - bias_B
                    allow_sub = True
                exp_a_field, exp_b_field = self._get_exp_pair_for_sum(
                    target_exp_sum,
                    bias_A,
                    bias_B,
                    max_normal_exp_A_field,
                    max_normal_exp_B_field,
                    allow_subnormal=allow_sub,
                )
                mant_a = self._generate_random_bits(mant_A_bits, mode="uniform")
                if exp_a_field == 0 and mant_A_bits > 0:
                    mant_a |= 1
                mant_b = self._generate_random_bits(mant_B_bits, mode="uniform")
                if exp_b_field == 0 and mant_B_bits > 0:
                    mant_b |= 1

            a_i = self.create_float_value(self.dtype_A, sign_a, exp_a_field, mant_a)
            b_i = self.create_float_value(self.dtype_B, sign_b, exp_b_field, mant_b)

            batch_A.append(a_i)
            batch_B.append(b_i)

        # --- 返回结果 ---
        return (
            np.array(batch_A, dtype=self.np_type_A),
            np.array(batch_B, dtype=self.np_type_B),
            c_to_return,
        )  # 返回包含单个 C 值的数组
