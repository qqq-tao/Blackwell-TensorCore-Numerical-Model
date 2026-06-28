from operator import is_
import os
from pickletools import long1
import time
import json
import sys
import yaml
import numpy as np
import glob
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
import os
import pycuda.compiler

print(pycuda.compiler.__file__)

os.environ["PYCUDA_DEBUG"] = "1"

input_type = {
    "f64": np.uint64,
    "f32": np.uint32,
    "tf32": np.uint32,
    "f16": np.uint16,
    "bf16": np.uint16,
    "e4m3": np.uint8,
    "e5m2": np.uint8,
    "e2m3": np.uint8,
    "e3m2": np.uint8,
    "e3m2": np.uint8,
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
    def __init__(self, mma_shape, minimum_mma, dtype_dict, stype):
        super().__init__()
        self.mma_shape = mma_shape
        self.minimum_mma = minimum_mma
        self.dtype_dict = dtype_dict
        if stype is not None:
            self.stype = stype
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
        self.value_generator = random_case_generator.FloatRandomGenerator(
            self.dtype_dict["A"], self.dtype_dict["B"], self.dtype_dict["C"], stype
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

        stype = self.stype if hasattr(self, "stype") else None
        is_block_scale = True if stype else False

        scale_index = (
            {
                "tidA": 0,
                "bidA": 0,
                "tidB": 0,
                "bidB": 0,
            }
            if is_block_scale
            else None
        )

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
            is_block_scale,
            scale_index,
            stype,
        )
        print(cuda_code)
        self.mod = SourceModule(
            cuda_code,
            nvcc=os.environ.get("NVCC", "nvcc"),
            arch="sm_120a",
            keep=True,
        )

        self.func_name = (
            dense_mma_sync.cuda_func_pattern.format(
                M=M, N=N, K=K, output_type=C_type, A_type=A_type, B_type=B_type
            )
            if stype is None
            else dense_mma_sync.cuda_block_scale_func_pattern.format(
                M=M,
                N=N,
                K=K,
                output_type=C_type,
                A_type=A_type,
                B_type=B_type,
                stype=stype,
            )
        )

    def mma_kernel_run(self, h_a, h_b, h_c, h_scale_A=None, h_scale_B=None):
        d_output = cuda.mem_alloc(h_c.nbytes)
        d_inputA = cuda.mem_alloc(h_a.nbytes)
        d_inputB = cuda.mem_alloc(h_b.nbytes)
        d_inputC = cuda.mem_alloc(h_c.nbytes)
        h_output = np.zeros_like(h_c)
        cuda.memcpy_htod(d_inputA, h_a)
        cuda.memcpy_htod(d_inputB, h_b)
        cuda.memcpy_htod(d_inputC, h_c)
        cuda.memcpy_htod(d_output, h_output)
        func = self.mod.get_function(self.func_name)
        block_size = (32, 1, 1)
        grid_size = (1, 1, 1)
        if hasattr(self, "stype"):
            d_scale_A = cuda.mem_alloc(h_scale_A.nbytes)
            d_scale_B = cuda.mem_alloc(h_scale_B.nbytes)
            cuda.memcpy_htod(d_scale_A, h_scale_A)
            cuda.memcpy_htod(d_scale_B, h_scale_B)
            func(
                d_inputA,
                d_inputB,
                d_inputC,
                d_output,
                d_scale_A,
                d_scale_B,
                block=block_size,
                grid=grid_size,
            )
        else:
            func(
                d_inputA, d_inputB, d_inputC, d_output, block=block_size, grid=grid_size
            )
        # cuda.memcpy_dtoh(h_a, d_inputA)
        # cuda.memcpy_dtoh(h_b, d_inputB)
        # cuda.memcpy_dtoh(h_c, d_inputC)

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
        """
        运行单个随机测试用例，比较 GPU 和 CModel 结果。
        如果失败，调用 save_failed_case 保存详细信息。

        返回:
            bool: True 表示通过，False 表示失败。
        """
        passed = True  # 假设测试通过
        h_a, h_b, h_c = self.host_reset()

        # 生成随机 A, B, C 数据
        # 注意: generate_batch 现在返回 A[K], B[K], C[1]
        # K = self.mma_shape[2]
        a_data, b_data, c_data_array = self.value_generator.generate_batch(
            self.mma_shape[2]
        )

        # 将生成的数据填充到 host buffer
        # 假设 MMA 计算只使用 M=1, N=1 slice (根据下面的 CModel 调用推断)
        # 如果 MMA 形状 M 或 N > 1，这里的填充逻辑需要调整
        # if self.mma_shape[0] > 1 or self.mma_shape[1] > 1:
        #     print(
        #         f"警告: run_test_case 当前假设 M=1, N=1。实际 MMA shape 为 {self.mma_shape}。填充和比较逻辑可能需要调整。"
        #     )

        h_a[0, :] = a_data  # 填充第一行
        h_b[:, 0] = b_data  # 填充第一列
        h_c[0, 0] = c_data_array[0]  # 填充 C[0,0]

        # 运行 CUDA Kernel
        output = self.mma_kernel_run(h_a, h_b, h_c)

        # 运行 CModel (注意参数对应关系)
        tc = tensor_core_mma_cmodel.TensorCore(
            self.dtype_dict["C"],
            self.dtype_dict["A"],
            self.dtype_dict["B"],
            self.dtype_dict["C"],
        )
        # 使用与 host buffer 填充对应的 slice
        cmodel_result = tc.recursive_mma_accumulate(
            h_a[0, :], h_b[:, 0], h_c[0:1, 0]
        )  # 使用 h_c[0:1, 0] 保持维度

        # --- 比较结果 ---
        # 只比较输出矩阵的第一个元素 (假设 M=1, N=1)
        gpu_result_val = output[0, 0]
        cmodel_result_val = cmodel_result[0]  # CModel 返回的是数值

        if gpu_result_val != cmodel_result_val:
            passed = False
            # 如果失败，立即保存详细信息
            print(
                f"检测到不匹配! GPU: {gpu_result_val:032b}, CModel: {cmodel_result_val:032b}\ngap: {gpu_result_val.astype(np.int64) - cmodel_result_val.astype(np.int64)}"
            )  # 打印十六进制
            # 注意 cmodel_result 可能需要转换类型才能保存
            # 假设 cmodel_result 是单个值，需要包装成与 output 兼容的形状进行保存
            cmodel_result_array = np.zeros_like(output)
            cmodel_result_array[0, 0] = cmodel_result_val
            self.save_failed_case(
                h_a, h_b, h_c, output, cmodel_result_array
            )  # 传递一致形状的 cmodel 结果

        return passed  # 只返回布尔状态

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

    def load_and_run_failed_case(self, file_path):
        """读取并重新运行失败的测试用例"""
        with open(file_path, "r") as f:
            failed_case = json.load(f)

        h_a = np.array(failed_case["h_a"], dtype=input_type[self.dtype_dict["A"]])
        h_b = np.array(
            failed_case["h_b"], dtype=input_type[self.dtype_dict["B"]], order="F"
        )
        h_c = np.array(failed_case["h_c"], dtype=input_type[self.dtype_dict["C"]])

        output = self.mma_kernel_run(h_a, h_b, h_c)
        tc = tensor_core_mma_cmodel.TensorCore(
            self.dtype_dict["C"],
            self.dtype_dict["A"],
            self.dtype_dict["B"],
            self.dtype_dict["C"],
        )
        cmodel_result = tc.recursive_mma_accumulate(h_a[0, :], h_b[:, 0], h_c[0:1, 0])
        cmodel_result = int(cmodel_result)
        # 输出结果以便调试
        print(f"Debugging failed case from {file_path}:")
        print("Input h_a:", end=" ")
        for val in h_a[0, :]:
            print(f"0x{val:08x}", end="    ")
            # print(f"0x{val}", end="    ")
        print()
        print("Input h_b:", end=" ")
        for val in h_b[:, 0]:
            print(f"0x{val:08x}", end="    ")
            # print(f"0x{val}", end="    ")
        print()
        print(f"Input h_c:     0b{h_c[0][0]:032b} 0x {h_c[0][0]:08x}")
        print(f"Output:        0b{output[0][0]:032b} 0x {output[0][0]:08x}")
        # print(f"Output: {output}")
        print(f"CModel Result: 0b{cmodel_result:032b} 0x {cmodel_result:08x}")
        output_int32 = output.astype(np.int64)
        # cmodel_result_int32 = long1(cmodel_result)
        print(f"gap: {output_int32[0][0]-cmodel_result}")
        # print(f"CModel Result: {cmodel_result}")

    # get A, B, C configuration from yaml, turn into array, callable benchmark_values included.
    # only one configuration needed. replicated by different dtypes.
    # Inside class NumericalStrategy:
    def test_cases_from_yaml(self):
        """
        运行大量随机测试用例，并将结果流式写入 .jsonl 文件，
        最后附加摘要信息。避免将所有结果存储在内存中。

        返回:
            dict: 包含测试摘要信息的字典。
        """
        # test_case_num = 1 << 17 # 131072 个测试用例
        # 默认保持原始 1024 个用例；可用 MMA_SIM_TEST_CASES 缩短 smoke。
        test_case_num = int(os.environ.get("MMA_SIM_TEST_CASES", 1 << 10))

        error_count = 0
        # 输出文件名
        # 使用 .jsonl (JSON Lines) 扩展名表示每行一个 JSON 对象
        output_filename = f"result/{self.dtype_dict['A']}.{self.dtype_dict['B']}.{self.dtype_dict['C']}_results.jsonl"

        # 确保结果目录存在
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        print(f"开始运行 {test_case_num} 个测试用例，结果将写入 {output_filename}...")

        start_ns = time.time_ns()  # 使用纳秒获得更高精度计时

        # 使用 'with open' 确保文件总是被关闭
        with open(output_filename, "w", encoding="utf-8") as f:
            # 写入元数据/头部信息 (可选, 作为 JSON 对象)
            meta_data = {
                "type": "metadata",
                "dtype_A": self.dtype_dict["A"],
                "dtype_B": self.dtype_dict["B"],
                "dtype_C": self.dtype_dict["C"],
                "mma_shape": self.mma_shape,
                "minimum_mma": self.minimum_mma,
                "total_cases_planned": test_case_num,
                "start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(start_ns / 1e9)
                ),
            }
            f.write(json.dumps(meta_data, ensure_ascii=False) + "\n")

            # 运行测试用例循环
            for k in range(test_case_num):
                # 运行单个测试用例，只获取通过/失败状态
                passed = self.run_test_case()

                if not passed:
                    error_count += 1

                # 创建一个日志条目 (可以根据需要增减信息)
                log_entry = {
                    "type": "test_result",
                    "case_index": k,
                    "status": "PASS" if passed else "FAIL",
                    # 可以选择性添加时间戳等: "timestamp_ns": time.time_ns()
                }
                # 将日志条目转换为 JSON 字符串并写入文件，每行一个
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # (可选) 打印进度
                if (k + 1) % 1000 == 0:  # 每 1000 次打印一次进度
                    elapsed_sec = (time.time_ns() - start_ns) / 1e9
                    print(
                        f"  已完成 {k+1}/{test_case_num} ({elapsed_sec:.2f} 秒)... Errors: {error_count}"
                    )

        end_ns = time.time_ns()  # 记录结束时间
        duration_sec = (end_ns - start_ns) / 1e9  # 计算总耗时（秒）
        error_rate = (error_count / test_case_num * 100) if test_case_num > 0 else 0

        print(f"测试完成。总共 {test_case_num} 个用例，发现 {error_count} 个错误。")
        print(f"总耗时: {duration_sec:.3f} 秒。")
        print(f"结果日志已保存到: {output_filename}")

        # 创建摘要信息
        summary = {
            "type": "summary",
            "total_cases_run": test_case_num,
            "errors": error_count,
            "error_rate_percent": f"{error_rate:.4f}",  # 保留更多小数位
            "elapsed_time_sec": f"{duration_sec:.3f}",
            "end_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(end_ns / 1e9)
            ),
        }

        # 将摘要信息追加到文件末尾 (作为最后一个 JSON 对象)
        try:
            with open(output_filename, "a", encoding="utf-8") as f:  # 使用 'a' 模式追加
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except IOError as e:
            print(f"错误：无法追加摘要信息到 {output_filename}: {e}")

        # 返回摘要信息字典
        return summary


def test_run_func(failed_cases, dtype_dict, stype=None):
    json_file_path = "configs/mma_shape_configuration.json"
    try:
        with open(json_file_path, "r") as file:
            test_configurations = json.load(file)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 {json_file_path}")
        return  # 或者抛出异常
    except json.JSONDecodeError:
        print(f"错误: 无法解析配置文件 {json_file_path}")
        return

    # --- 获取 MMA 配置 ---
    # (与之前类似，增加一些健壮性检查)
    dtype_A = dtype_dict["A"]
    if dtype_A not in test_configurations:
        print(f"错误: 在配置中找不到 {dtype_A} 的配置")
        return
    # ... (可以为 B, C 也添加类似检查) ...

    # 简化 mma_shape 和 minimum_mma 的获取，增加默认值或错误处理
    dense_config = test_configurations[dtype_A].get("dense", {})
    combinations = dense_config.get("combinations", [])
    subcore_shapes = dense_config.get("subcore_shapes", [])

    if not combinations:
        print(f"错误: {dtype_A} 的配置中缺少 'dense/combinations'")
        return
    # 选择一个组合，例如第一个或最后一个
    mma_shape = combinations[-1] if combinations else (0, 0, 0)  # 使用最后一个或默认值

    if not subcore_shapes:
        print(f"错误: {dtype_A} 的配置中缺少 'dense/subcore_shapes'")
        return
    minimum_mma = (
        subcore_shapes[0] if subcore_shapes else (0, 0, 0)
    )  # 使用第一个或默认值

    print(f"使用 MMA Shape: {mma_shape}, Minimum MMA: {minimum_mma}")
    print(f"数据类型: A={dtype_dict['A']}, B={dtype_dict['B']}, C={dtype_dict['C']}")

    # --- 运行测试 ---
    start_ns = time.time_ns()  # 对 test_run_func 本身也计时 (可选)
    strategy = NumericalStrategy(mma_shape, minimum_mma, dtype_dict, stype)

    summary_results = {}  # 存储来自 strategy 的摘要

    if len(failed_cases) == 0:
        # 调用修改后的函数，它现在写入文件并返回摘要
        summary_results = strategy.test_cases_from_yaml()
        # 注意：这里不再需要保存 results 到 JSON 文件，因为已经写入 .jsonl
        print("随机测试摘要:", json.dumps(summary_results, indent=4))
    else:
        print(f"重新运行 {len(failed_cases)} 个失败的用例...")
        # 重新运行失败用例的逻辑保持不变
        for case_path in failed_cases:
            print(f"\n--- 重新运行: {case_path} ---")
            try:
                strategy.load_and_run_failed_case(case_path)
            except FileNotFoundError:
                print(f"  错误: 找不到文件 {case_path}")
            except Exception as e:
                print(f"  错误: 重新运行时发生错误 {e}")
        # 对于重跑失败用例，可能没有统一的摘要，可以标记一下
        summary_results = {"status": "rerun_failed_cases", "count": len(failed_cases)}

    end_ns = time.time_ns()
    run_func_duration_sec = (end_ns - start_ns) / 1e9
    print(f"test_run_func 总耗时: {run_func_duration_sec:.3f} 秒")

    # test_run_func 本身不再需要写入主结果文件
    # 它的主要作用是协调测试流程和打印最终摘要（如果适用）
    return summary_results  # 可以返回摘要供外部使用


def get_failed_case_json_paths(folder_name="failed_cases"):
    """
    Opens a subfolder, finds all JSON files, and returns their full paths.

    Args:
        folder_name (str): The name of the subfolder to search.

    Returns:
        list: A list of full paths to JSON files in the specified folder.
              Returns an empty list if the folder doesn't exist or contains no JSON files.
    """
    failed_cases = []
    # Construct the full path to the subfolder
    subfolder_path = os.path.join(os.getcwd(), folder_name)

    # Check if the subfolder exists
    if not os.path.isdir(subfolder_path):
        print(f"Error: Subfolder '{subfolder_path}' not found.")
        return failed_cases

    # Use glob to find all files ending with .json in the subfolder
    # os.path.join is used to correctly construct the path for glob
    json_files = glob.glob(os.path.join(subfolder_path, "*.json"))

    # Add each found JSON file's full path to the list
    for file_path in json_files:
        failed_cases.append(file_path)

    return failed_cases


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(
            "使用方法: python nv_result_cmp_cmodel.py <type_A> <type_B> <type_C> [<stype>]"
        )
        sys.exit(1)
    dtype_A = sys.argv[1]
    dtype_B = sys.argv[2]
    dtype_C = sys.argv[3]
    stype = sys.argv[4] if len(sys.argv) > 4 else None
    dtype_dict = {"A": dtype_A, "B": dtype_B, "C": dtype_C}
    print(dtype_dict)
    failed_cases = []
    # failed_cases = get_failed_case_json_paths("failed_cases")
    # if failed_cases:
    #     print("Found JSON files in 'failed_cases':")
    #     for case_path in failed_cases:
    #         print(case_path)
    # else:
    #     print("No JSON files found in 'failed_cases' or the folder doesn't exist.")

    test_run_func(failed_cases, dtype_dict, stype)
