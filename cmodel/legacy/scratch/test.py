class TestTensorCore(unittest.TestCase):
    def setUp(self):
        self.tc = TensorCore(
            "f16", "f16", "f16", "f16"
        )  # Initialize with f16 for testing

    # def test_extract_f16_normal(self):
    #     f16_representation = np.array([0x0020, 0x0000], dtype=np.uint16)  # 1.0 in f16
    #     sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
    #         f16_representation, "f16"
    #     )
    #     # self.assertEqual(sign[0], 0)
    #     # self.assertEqual(exponent[0], 15)
    #     # self.assertEqual(mantissa[0], 0x400)

    #     self.assertEqual(
    #         special_tag[1], Special_tag_enum.Zero
    #     )  # Normal numbers have no special tag

    # Normal numbers have no special tag
    def test_extract_f16_array_normal(self):
        f16_representation = np.array(
            [0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00],
            dtype=np.uint16,
        )  # 1.0 in f16 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 15))
        self.assertTrue(np.all(mantissa == 0x400))
        self.assertTrue(
            np.all(special_tag == None)
        )  # Normal numbers have no special tag

    def test_extract_f16_array_denormal(self):
        f16_representation = np.array(
            [0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080],
            dtype=np.uint16,
        )  # Denormal numbers array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 1))  # Denormals have exponent 0
        self.assertTrue(np.array_equal(mantissa, [1, 2, 4, 8, 16, 32, 64, 128]))
        self.assertTrue(np.all(special_tag == None))  # Denormals have no special tag

    def test_extract_f16_array_zero(self):
        f16_representation = np.array(
            [0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000],
            dtype=np.uint16,
        )
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 0))
        self.assertTrue(np.all(mantissa == 0))
        self.assertTrue(np.all(special_tag == Special_tag_enum.Zero))

    def test_extract_f16_array_inf(self):
        f16_representation = np.array(
            [0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00],
            dtype=np.uint16,
        )
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 31))  # all 1's
        self.assertTrue(np.all(mantissa == 0))
        self.assertTrue(np.all(special_tag == Special_tag_enum.INF))

    def test_extract_f16_array_nan(self):
        f16_representation = np.array(
            [0x7C01, 0x7C01, 0x7C02, 0x7C04, 0x7C08, 0x7C10, 0x7C20, 0x7C40],
            dtype=np.uint16,
        )  # NaN array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 31))  # all 1's
        self.assertTrue(
            np.array_equal(mantissa, [1, 1, 2, 4, 8, 16, 32, 64])
        )  # not zero
        self.assertTrue(np.all(special_tag == Special_tag_enum.NaN))

    def test_extract_f16_array_negative(self):
        f16_representation = np.array(
            [0xBC00, 0xBC00, 0xBC00, 0xBC00, 0xBC00, 0xBC00, 0xBC00, 0xBC00],
            dtype=np.uint16,
        )  # -1.0 in f16 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 1))
        self.assertTrue(np.all(exponent == 15))
        self.assertTrue(np.all(mantissa == 0x400))
        self.assertTrue(np.all(special_tag == None))  # Negative normal number

    def test_extract_f16_array_normal_positive_boundary(self):
        f16_representation = np.array(
            [0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF],
            dtype=np.uint16,
        )  # f16 最大正常数 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 30)
        )  # 实际指数 30 + 1 (隐含位) - 15 (bias) = 16
        self.assertTrue(np.all(mantissa == 0x7FF))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_f16_array_normal_negative_boundary(self):
        f16_representation = np.array(
            [0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF],
            dtype=np.uint16,
        )  # f16 最大负数 (更正) array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 1))
        self.assertTrue(
            np.all(exponent == 30)
        )  # 实际指数 30 + 1 (隐含位) - 15 (bias) = 16
        self.assertTrue(np.all(mantissa == 0x7FF))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_f16_array_denormal_positive_boundary(self):
        f16_representation = np.array(
            [0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001],
            dtype=np.uint16,
        )  # f16 最小非规格化数 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 1))
        self.assertTrue(np.all(mantissa == 0x001))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_f16_array_denormal_negative_boundary(self):
        f16_representation = np.array(
            [0x8001, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001],
            dtype=np.uint16,
        )  # f16 最大非规格化数 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(sign == 1))
        self.assertTrue(np.all(exponent == 1))
        self.assertTrue(np.all(mantissa == 0x001))
        self.assertTrue(np.all(special_tag == None))

    # --------------------- 特殊值 Array 测试 ---------------------
    def test_check_special_value_f16_array_zero(self):
        f16_representation = np.array(
            [0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000],
            dtype=np.uint16,
        )
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(special_tag == Special_tag_enum.Zero))

    def test_check_special_value_f16_array_inf(self):
        print("*************")
        f16_representation = np.array(
            [0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00],
            dtype=np.uint16,
        )
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(special_tag == Special_tag_enum.INF))

    def test_check_special_value_f16_array_nan(self):
        print("****=====****")

        f16_representation = np.array(
            [
                0x7E01,
                0x7E02,
                0x7E04,
                0x7E08,
                0x7E10,
                0x7E20,
                0x7E40,
                0x7E01,
            ],  # 混合 NaN
            dtype=np.uint16,
        )  # NaN array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            f16_representation, "f16"
        )
        self.assertTrue(np.all(special_tag == Special_tag_enum.NaN))


class TestTensorCore_FP8(unittest.TestCase):
    def setUp(self):
        self.tc = TensorCore(
            "f32", "e5m2", "e4m3", "f32"
        )  # Initialize with f32, e5m2, e4m3, f32

    def test_extract_e5m2_array_normal(self):
        e5m2_representation = np.array(
            [0b0_01111_01, 0b0_01111_01, 0b0_01111_01, 0b0_01111_01], dtype=np.uint8
        )  # normal number in e5m2 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e5m2_representation, "e5m2"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 15))
        self.assertTrue(np.all(mantissa == 0b101))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_e4m3_array_normal(self):
        e4m3_representation = np.array(
            [0b0_0111_010, 0b0_0111_010, 0b0_0111_010, 0b0_0111_010], dtype=np.uint8
        )  # normal number in e4m3 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e4m3_representation, "e4m3"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 7))
        self.assertTrue(np.all(mantissa == 0b1010))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_e5m2_array_zero(self):
        e5m2_representation = np.array(
            [0b0_00000_00, 0b0_00000_00, 0b0_00000_00, 0b0_00000_00], dtype=np.uint8
        )  # Zero in e5m2 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e5m2_representation, "e5m2"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 0))
        self.assertTrue(np.all(mantissa == 0))
        self.assertTrue(np.all(special_tag == Special_tag_enum.Zero))

    def test_extract_e4m3_array_zero(self):
        e4m3_representation = np.array(
            [0b0_0000_000, 0b0_0000_000, 0b0_0000_000, 0b0_0000_000], dtype=np.uint8
        )  # Zero in e4m3 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e4m3_representation, "e4m3"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(np.all(exponent == 0))
        self.assertTrue(np.all(mantissa == 0))
        self.assertTrue(np.all(special_tag == Special_tag_enum.Zero))

    def test_extract_e5m2_array_inf(self):
        e5m2_representation = np.array(
            [0b0_11111_00, 0b0_11111_00, 0b0_11111_00, 0b0_11111_00], dtype=np.uint8
        )  # Inf in e5m2 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e5m2_representation, "e5m2"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 31)
        )  # Exponent is all ones (11111 in binary is 31)
        self.assertTrue(np.all(mantissa == 0))
        self.assertTrue(np.all(special_tag == Special_tag_enum.INF))

    def test_extract_e4m3_array_inf_nan(self):
        e4m3_representation = np.array(
            [0b0_1111_111, 0b0_1111_111, 0b0_1111_111, 0b0_1111_111], dtype=np.uint8
        )  # Inf/NaN in e4m3 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e4m3_representation, "e4m3"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 15)
        )  # Exponent is all ones (1111 in binary is 15)
        self.assertTrue(np.all(mantissa == 0b111))  # Mantissa is all ones
        self.assertTrue(
            np.all(special_tag == Special_tag_enum.NaN)
        )  # e4m3 treats 1111_111 as NaN

    def test_extract_e4m3_array_no_inf(self):
        e4m3_representation = np.array(
            [0b0_1111_000, 0b0_1111_000, 0b0_1111_000, 0b0_1111_000], dtype=np.uint8
        )  # Inf/NaN in e4m3 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e4m3_representation, "e4m3"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 15)
        )  # Exponent is all ones (1111 in binary is 15)
        self.assertTrue(np.all(mantissa == 0b1000))  # Mantissa is all ones
        self.assertTrue(np.all(special_tag == None))

    def test_extract_e5m2_array_nan(self):
        e5m2_representation = np.array(
            [0b0_11111_01, 0b0_11111_01, 0b0_11111_01, 0b0_11111_01], dtype=np.uint8
        )  # NaN in e5m2 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e5m2_representation, "e5m2"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 31)
        )  # Exponent is all ones (11111 in binary is 31)
        self.assertTrue(np.all(mantissa == 0b01))  # Mantissa is not zero
        self.assertTrue(np.all(special_tag == Special_tag_enum.NaN))

    def test_extract_e4m3_array_negative_normal(self):
        e4m3_representation = np.array(
            [0b1_0111_010, 0b1_0111_010, 0b1_0111_010, 0b1_0111_010], dtype=np.uint8
        )  # Negative normal number in e4m3 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e4m3_representation, "e4m3"
        )
        self.assertTrue(np.all(sign == 1))
        self.assertTrue(np.all(exponent == 7))
        self.assertTrue(np.all(mantissa == 0b1010))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_e5m2_array_negative_normal(self):
        e5m2_representation = np.array(
            [0b1_01111_01, 0b1_01111_01, 0b1_01111_01, 0b1_01111_01], dtype=np.uint8
        )  # Negative normal number in e5m2 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e5m2_representation, "e5m2"
        )
        self.assertTrue(np.all(sign == 1))
        self.assertTrue(np.all(exponent == 15))
        self.assertTrue(np.all(mantissa == 0b101))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_e4m3_array_denormal(self):
        e4m3_representation = np.array(
            [0b0_0000_001, 0b0_0000_001, 0b0_0000_001, 0b0_0000_001], dtype=np.uint8
        )  # Denormal number in e4m3 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e4m3_representation, "e4m3"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 1)
        )  # Denormals exponent should be 1 after adding denormal_flag
        self.assertTrue(np.all(mantissa == 0b001))
        self.assertTrue(np.all(special_tag == None))

    def test_extract_e5m2_array_denormal(self):
        e5m2_representation = np.array(
            [0b0_00000_01, 0b0_00000_01, 0b0_00000_01, 0b0_00000_01], dtype=np.uint8
        )  # Denormal number in e5m2 array
        sign, exponent, mantissa, special_tag = self.tc.extract_sign_exponent_mantissa(
            e5m2_representation, "e5m2"
        )
        self.assertTrue(np.all(sign == 0))
        self.assertTrue(
            np.all(exponent == 1)
        )  # Denormals exponent should be 1 after adding denormal_flag
        self.assertTrue(np.all(mantissa == 0b01))
        self.assertTrue(np.all(special_tag == None))
