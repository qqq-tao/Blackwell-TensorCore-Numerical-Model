import os
import time
import json
import sys
import yaml
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from unittest import TestCase

sys.path.append("benchmark_values")
sys.path.append("nvidia_dense_mma")
import dense_mma_sync
import benchmark_values
import tensor_core_mma_cmodel
import random_case_generator

input_type = {
    "f64": np.uint64,
    "f32": np.uint32,
    "tf32": np.uint32,
    "f16": np.uint16,
    "bf16": np.uint16,
    "e4m3": np.uint8,
    "e5m2": np.uint8,
    "s32": np.int32,
    "s8": np.int8,
    "u8": np.uint8,
    "s4": np.uint8,
    "u4": np.uint8,
    "b1": np.int8,
}


def printpass(flag):
    return "PASS" if flag else "FAIL"


class NumericalStrategy(TestCase):
    def __init__(self, mma_shape, minimum_mma, dtype_dict):
        super().__init__()
        self.mma_shape = mma_shape
        self.minimum_mma = minimum_mma
        self.dtype_dict = dtype_dict
        self.mma_kernel_init()
        # self.A_value_generator = benchmark_values.TestDataGenerator(
        #     self.dtype_dict["A"]
        # )
        # self.B_value_generator = benchmark_values.TestDataGenerator(
        #     self.dtype_dict["B"]
        # )
        # self.C_value_generator = benchmark_values.TestDataGenerator(
        #     self.dtype_dict["C"]
        # )
        # print(self.dtype_dict)
        self.A_value_generator = random_case_generator.FloatRandomGenerator(
            self.dtype_dict["A"]
        )
        self.B_value_generator = random_case_generator.FloatRandomGenerator(
            self.dtype_dict["B"]
        )
        self.C_value_generator = random_case_generator.FloatRandomGenerator(
            self.dtype_dict["C"]
        )

    def host_reset(self):
        M = self.mma_shape[0]
        N = self.mma_shape[1]
        K = self.mma_shape[2]
        h_a = np.zeros((M, K), dtype=input_type[self.dtype_dict["A"]])
        h_b = np.zeros((K, N), dtype=input_type[self.dtype_dict["B"]], order="F")
        h_c = np.zeros((M, N), dtype=input_type[self.dtype_dict["C"]])

        return h_a, h_b, h_c

    def mma_kernel_init(self):
        A_type = self.dtype_dict["A"]
        B_type = self.dtype_dict["B"]
        C_type = self.dtype_dict["C"]
        M = self.mma_shape[0]
        N = self.mma_shape[1]
        K = self.mma_shape[2]

        cuda_code = dense_mma_sync.fill_in_params(
            M,
            N,
            K,
            self.minimum_mma,
            A_type,
            B_type,
            C_type,
            dense_mma_sync.bit_length[A_type],
            dense_mma_sync.bit_length[C_type],
        )
        print(cuda_code)
        self.mod = SourceModule(
            cuda_code, options=["-arch=sm_89", "--cubin"], keep=True
        )
        self.func_name = dense_mma_sync.cuda_func_pattern.format(
            M=M, N=N, K=K, output_type=C_type, A_type=A_type, B_type=B_type
        )

    def mma_kernel_run(self, h_a, h_b, h_c):
        d_output = cuda.mem_alloc(h_c.nbytes)
        d_inputA = cuda.mem_alloc(h_a.nbytes)
        d_inputB = cuda.mem_alloc(h_b.nbytes)
        d_inputC = cuda.mem_alloc(h_c.nbytes)
        cuda.memcpy_htod(d_inputA, h_a)
        cuda.memcpy_htod(d_inputB, h_b)
        cuda.memcpy_htod(d_inputC, h_c)
        func = self.mod.get_function(self.func_name)
        block_size = (32, 1, 1)
        grid_size = (1, 1, 1)
        func(d_inputA, d_inputB, d_inputC, d_output, block=block_size, grid=grid_size)
        h_output = np.empty_like(h_c)
        cuda.memcpy_dtoh(h_output, d_output)
        return h_output

    def host_reset(self):
        M = self.mma_shape[0]
        N = self.mma_shape[1]
        K = self.mma_shape[2]
        h_a = np.zeros((M, K), dtype=input_type[self.dtype_dict["A"]])
        h_b = np.zeros((K, N), dtype=input_type[self.dtype_dict["B"]], order="F")
        # h_b = np.zeros((K, N), dtype=input_type[self.dtype_dict["B"]])
        h_c = np.zeros((M, N), dtype=input_type[self.dtype_dict["C"]])
        return h_a, h_b, h_c

    def run_test_case(self):
        flag = True
        h_a, h_b, h_c = self.host_reset()
        a_data = self.A_value_generator.generate_batch(self.mma_shape[2])
        b_data = self.B_value_generator.generate_batch(self.mma_shape[2])
        h_c[0:1, 0] = self.C_value_generator.generate_batch(1)
        for i in range(self.mma_shape[2]):
            h_a[0][i] = a_data[i]
            h_b[i][0] = b_data[i]
        # print()
        # print("=" * 50)
        # print(h_a)
        # print(h_b)
        # print(h_c)
        # h_c[0][0] = self.C_value_generator.generate(c)
        output = self.mma_kernel_run(h_a, h_b, h_c)
        # print(output)
        # reference_c = self.C_value_generator.generate(expected_c)
        tc = tensor_core_mma_cmodel.TensorCore(
            self.dtype_dict["C"],
            self.dtype_dict["A"],
            self.dtype_dict["B"],
            self.dtype_dict["C"],
        )
        cmodel_result = tc.recursive_mma_accumulate(h_a[0, :], h_b[:, 0], h_c[0:1, 0])
        flag = flag and (output[0][0] == cmodel_result)
        outfile = ""
        # outfile = f"{description}:\n"
        # outfile += f"h_a: {h_a[0][0]}, h_b: {h_b[0][0]}, h_c: {h_c[0][0]}\n"
        # outfile += f"h_a: {benchmark_values.print_16bit_hex(h_a[0][0])}, h_b: {benchmark_values.print_16bit_hex(h_b[0][0])}, h_c: {benchmark_values.print_16bit_hex(h_c[0][0])}\n"
        # outfile += f"h_c: {benchmark_values.print_32bit_hex(h_c[0][0])}\n"
        # outfile += f"d_c: {benchmark_values.print_16bit_hex(output[0][0])}, cmodel_result: {benchmark_values.print_16bit_hex(cmodel_result)}, expected_c: {benchmark_values.print_16bit_hex(reference_c)}\n"
        # outfile += f"d_c: {benchmark_values.print_32bit_hex(output[0][0])}, cmodel_result: {benchmark_values.print_32bit_hex(cmodel_result)}\n"
        # outfile += f"d_c: {output[0][0]} expected_c: {expected_c}\n"
        # return outfile + "result: " + printpass(flag) + "\n"
        if output[0][0] != cmodel_result:
            flag = False
            self.save_failed_case(h_a, h_b, h_c, output, cmodel_result)
        return flag

    def save_failed_case(self, h_a, h_b, h_c, output, cmodel_result):
        """保存失败的测试用例到 JSON 文件"""
        # 确保保存目录存在
        if not os.path.exists("failed_cases"):
            os.makedirs("failed_cases")

        failed_case = {
            "h_a": h_a.tolist(),  # 转换为列表以便 JSON 序列化
            "h_b": h_b.tolist(),
            "h_c": h_c.tolist(),
            "output": output.tolist(),
            "cmodel_result": cmodel_result.tolist(),
            "timestamp": time.time_ns(),  # 添加时间戳作为唯一标识
        }

        # 保存到文件
        file_path = f"failed_cases/failed_case_{time.time_ns()}.json"
        with open(file_path, "w") as f:
            json.dump(failed_case, f, indent=4)
        print(f"Failed case saved to {file_path}")

    # def save_failed_case(self, h_a, h_b, h_c, output, cmodel_result):
    #     """保存失败的测试用例到 JSON 文件"""

    #     # 确保保存目录存在
    #     if not os.path.exists("failed_cases"):
    #         os.makedirs("failed_cases")

    #     h_a_list = [list(row) for row in h_a]
    #     h_b_list = [list(row) for row in h_b]
    #     h_c_list = [list(row) for row in h_c]
    #     output_list = [list(row) for row in output]
    #     cmodel_result_list = (
    #         list(cmodel_result)
    #         if isinstance(cmodel_result, np.ndarray)
    #         else [cmodel_result]
    #     )

    #     failed_case = {
    #         "h_a": h_a_list,
    #         "h_b": h_b_list,
    #         "h_c": h_c_list,
    #         "output": output_list,
    #         "cmodel_result": cmodel_result_list,
    #         "timestamp": time.time_ns(),
    #     }
    #     print(failed_case)
    #     # 保存到文件，使用 indent 参数进行缩进
    #     file_path = f"failed_cases/failed_case_{time.time_ns()}.json"
    #     with open(file_path, "w") as f:
    #         json.dump(failed_case, f, indent=4)  # indent=4 使 JSON 可读性更好

    #     print(f"Failed case saved to {file_path}")

    # # 示例调用（使用您提供的数据）
    # h_a = [[31428, 35497, 33645, 13373, 50347, 34003, 60780, 17416, 27785, 31208, 36461, 4882, 59751, 61879, 61546, 49642], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...]  # 省略部分数据
    # h_b = [[34972, 0, 0, 0, 0, 0, 0, 0], [14716, 0, 0, 0, 0, 0, 0, 0], ...]
    # h_c = [[789, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], ...]
    # output = [[64512, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], ...]
    # cmodel_result = [29858]

    # save_failed_case(h_a, h_b, h_c, output, cmodel_result)

    def load_and_run_failed_case(self, file_path):
        """读取并重新运行失败的测试用例"""
        # 从 JSON 文件加载数据
        with open(file_path, "r") as f:
            failed_case = json.load(f)

        # 还原 NumPy 数组并指定正确的数据类型
        h_a = np.array(failed_case["h_a"], dtype=input_type[self.dtype_dict["A"]])
        # h_b = np.array(
        #     (self.mma_shape[2], self.mma_shape[1]),
        #     dtype=input_type[self.dtype_dict["B"]],
        #     order="F",
        # )
        h_b = np.zeros(
            (self.mma_shape[2], self.mma_shape[1]),
            dtype=input_type[self.dtype_dict["B"]],
            order="F",
        )
        print(h_b.shape)
        print(np.array(failed_case["h_b"])[:, 0])
        h_b[:, 0] = np.array(failed_case["h_b"])[:, 0]
        # h_b = np.array(
        #     failed_case["h_b"], dtype=input_type[self.dtype_dict["B"]], order="F"
        # )
        h_c = np.array(failed_case["h_c"], dtype=input_type[self.dtype_dict["C"]])

        # 重新运行测试
        output = self.mma_kernel_run(h_a, h_b, h_c)
        tc = tensor_core_mma_cmodel.TensorCore(
            self.dtype_dict["C"],
            self.dtype_dict["A"],
            self.dtype_dict["B"],
            self.dtype_dict["C"],
        )
        cmodel_result = tc.recursive_mma_accumulate(
            h_a[0, :], np.array(failed_case["h_b"], dtype=np.uint16)[:, 0], h_c[0:1, 0]
        )
        cmodel_result = int(cmodel_result)
        # 输出结果以便调试
        print(f"Debugging failed case from {file_path}:")
        print("Input h_a:", end=" ")
        for val in h_a[0, :]:
            print(f"{val:04x}", end="    ")
        print()
        print("Input h_b:", end=" ")
        for val in h_b[:, 0]:
            print(f"{val:04x}", end="    ")
        print()
        print(f"Input h_c: {h_c[0][0]:4x}")
        print(f"Output: {output[0][0]:4x}")
        # print(f"Output: {output}")
        print(f"CModel Result: {cmodel_result:4x}")
        # print(f"CModel Result: {cmodel_result}")

    # get A, B, C configuration from yaml, turn into array, callable benchmark_values included.
    # only one configuration needed. replicated by different dtypes.
    def test_cases_from_yaml(self):
        # with open(
        #     f"test_cases/{self.dtype_dict["A"]}.{self.dtype_dict["B"]}.{self.dtype_dict["C"]}_test_cases.yaml",
        #     "r",
        # ) as file:
        #     # with open("case_check_list.yaml", "r") as file:
        #     test_cases = yaml.safe_load(file)

        # defaults = test_cases["defaults"]
        # cases = test_cases["test_cases"]
        error_count = 0
        results = []
        test_case_num = 1 << 22
        start = time.time_ns()
        for k in range(test_case_num):
            # description = case_data["description"]
            # override = case_data.get("override", {})
            # a = override.get("a", defaults["a"])
            # b = override.get("b", defaults["b"])
            # c = override.get("c", defaults["c"])
            # expected_c = case_data.get("expected_c", defaults["expected_c"])

            result = self.run_test_case()
            # results.append(case_name)
            # for _ in case_data["override"]:
            #     results.append(_ + ": " + str(case_data["override"][_]))
            if result == False:
                error_count += 1
            results.append(f"test_case {k}: " + str(result))
        results.append(
            f"Error rate: {error_count / test_case_num * 100:.2f}% ({error_count} errors)"
        )

        end = time.time_ns()
        results.append(f"elapsed time(s): {(end-start)/1e9}")
        return results


def test_run_func(failed_cases):
    json_file_path = "configs/mma_shape_configuration.json"
    with open(json_file_path, "r") as file:
        test_configurations = json.load(file)

    dtype_dict = {"A": "tf32", "B": "tf32", "C": "f32"}
    mma_shape = (
        test_configurations[dtype_dict["A"]]["dense"]["combinations"][1]
        if len(test_configurations[dtype_dict["A"]]["dense"]["combinations"]) > 1
        else test_configurations[dtype_dict["A"]]["dense"]["combinations"][0]
    )
    print(mma_shape)
    minimum_mma = test_configurations[dtype_dict["A"]]["dense"]["subcore_shapes"][0]
    print("mma_shap: ", mma_shape)
    start = time.time_ns()
    strategy = NumericalStrategy(mma_shape, minimum_mma, dtype_dict)

    results = ""
    if len(failed_cases) == 0:
        results = strategy.test_cases_from_yaml()
    else:
        for case in failed_cases:
            strategy.load_and_run_failed_case(case)
    # results_str = json.dumps(results, indent=4)
    end = time.time_ns()
    print((end - start) / 1e9)
    for result in results:
        print(result)


# Example usage
if __name__ == "__main__":
    failed_cases = []
    test_run_func(failed_cases)
