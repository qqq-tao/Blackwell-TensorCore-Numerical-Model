#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>


namespace py=pybind11;
// enum __nv_fp8_interpretation_t
// Enumerates the possible interpretations of the 8-bit values when referring to them as fp8 types.

// Values:

// enumerator __NV_E4M3
// Stands for fp8 numbers of e4m3 kind.

// enumerator __NV_E5M2
// Stands for fp8 numbers of e5m2 kind.

// enum __nv_saturation_t
// Enumerates the modes applicable when performing a narrowing conversion to fp8 destination types.

// Values:

// enumerator __NV_NOSAT
// Means no saturation to finite is performed when conversion results in rounding values outside the range of destination type.

// NOTE: for fp8 type of e4m3 kind, the results that are larger than the maximum representable finite number of the target format become NaN.

// enumerator __NV_SATFINITE
// Means input larger than the maximum representable finite number MAXNORM of the target format round to the MAXNORM of the same sign as input.

/***********************************************
* __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
* __nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
*                      const __nv_fp8_interpretation_t fp8_interpretation);
*/

void check_buf_dim(const py::buffer_info& srcbuf, const py::buffer_info& dstbuf)
{
    if (srcbuf.ndim !=1 || dstbuf.ndim !=1)
    {
        throw std::runtime_error("Number of dimensions must be one");
    }

    if (srcbuf.size != dstbuf.size)
    {
        throw std::runtime_error("Input shape must match");
    }
}

void cvt_float_to_fp8(py::array_t<float>& src, py::array_t<unsigned char>& dst, bool is_e4m3, bool do_sat)
{
    py::buffer_info srcbuf = src.request();
    py::buffer_info dstbuf = dst.request();
    check_buf_dim(srcbuf, dstbuf);

    //Obtain numpy.ndarray data pointer
    const float* psrc = (float*)srcbuf.ptr;
    __nv_fp8_storage_t* pdst = (__nv_fp8_storage_t*)dstbuf.ptr;

    __nv_saturation_t sat = do_sat ? __nv_saturation_t::__NV_SATFINITE : __nv_saturation_t::__NV_NOSAT;
    __nv_fp8_interpretation_t t = is_e4m3 ? __nv_fp8_interpretation_t::__NV_E4M3 : __nv_fp8_interpretation_t::__NV_E5M2;

    for(size_t i=0; i<srcbuf.size; i++)
    {
        pdst[i] = __nv_cvt_float_to_fp8(psrc[i], sat, t);
    }
}

// __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
// __nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate,
//                         const __nv_fp8_interpretation_t fp8_interpretation);
void cvt_half_to_fp8(py::array_t<unsigned short>& src, py::array_t<unsigned char>& dst, bool is_e4m3, bool do_sat)
{
    py::buffer_info srcbuf = src.request();
    py::buffer_info dstbuf = dst.request();
    check_buf_dim(srcbuf, dstbuf);

    //Obtain numpy.ndarray data pointer
    const __half_raw* psrc = (__half_raw*)srcbuf.ptr;
    __nv_fp8_storage_t* pdst = (__nv_fp8_storage_t*)dstbuf.ptr;

    __nv_saturation_t sat = do_sat ? __nv_saturation_t::__NV_SATFINITE : __nv_saturation_t::__NV_NOSAT;
    __nv_fp8_interpretation_t t = is_e4m3 ? __nv_fp8_interpretation_t::__NV_E4M3 : __nv_fp8_interpretation_t::__NV_E5M2;

    for(size_t i=0; i<srcbuf.size; i++)
    {
        pdst[i] = __nv_cvt_halfraw_to_fp8(psrc[i], sat, t);

    }
}

// __CUDA_HOSTDEVICE_FP8_DECL__ __half_raw
// __nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
//                         const __nv_fp8_interpretation_t fp8_interpretation);

void cvt_fp8_to_half(py::array_t<unsigned char>& src, py::array_t<unsigned short>& dst, bool is_e4m3)
{
    py::buffer_info srcbuf = src.request();
    py::buffer_info dstbuf = dst.request();
    check_buf_dim(srcbuf, dstbuf);

    //Obtain numpy.ndarray data pointer
    const __nv_fp8_storage_t* psrc = (const __nv_fp8_storage_t*)srcbuf.ptr;
    __half_raw* pdst = (__half_raw*)dstbuf.ptr;

    __nv_fp8_interpretation_t t = is_e4m3 ? __nv_fp8_interpretation_t::__NV_E4M3 : __nv_fp8_interpretation_t::__NV_E5M2;

    for(size_t i=0; i<srcbuf.size; i++)
    {
        pdst[i] = __nv_cvt_fp8_to_halfraw(psrc[i], t);
    }
}

// cvt_float_to_bf16
void cvt_float_to_bf16(py::array_t<float>& src, py::array_t<unsigned short>& dst) {
    py::buffer_info srcbuf = src.request();
    py::buffer_info dstbuf = dst.request();
    check_buf_dim(srcbuf, dstbuf);

    const float* psrc = (const float*)srcbuf.ptr;
    __nv_bfloat16* pdst = (__nv_bfloat16*)dstbuf.ptr;
    for(size_t i=0; i<srcbuf.size; i++) {
        pdst[i] = __float2bfloat16(psrc[i]);
    }
}
// Custom conversion functions for e2m1 (1 sign bit, 2 exponent bits, 1 mantissa bit)
void cvt_float_to_e2m1(py::array_t<float>& src, py::array_t<unsigned char>& dst) {
    py::buffer_info srcbuf = src.request();
    py::buffer_info dstbuf = dst.request();
    check_buf_dim(srcbuf, dstbuf);

    const float* psrc = (const float*)srcbuf.ptr;
    unsigned char* pdst = (unsigned char*)dstbuf.ptr;

    for(size_t i=0; i<srcbuf.size; i++) {
        pdst[i] = __nv_cvt_float_to_fp4(psrc[i], __NV_E2M1, cudaRoundNearest);
    }
}

PYBIND11_MODULE(fpemu, m) {
    m.doc() = "fp8 converting module"; // optional module docstring
    m.def("cvt_float_to_fp8", &cvt_float_to_fp8, "convert float to fp8");  
    m.def("cvt_half_to_fp8", &cvt_half_to_fp8, "convert half to fp8");
    m.def("cvt_fp8_to_half", &cvt_fp8_to_half, "convert fp8 to half");
    // cvt_float_to_bf16
    m.def("cvt_float_to_bf16", &cvt_float_to_bf16, "convert float to bf16"); // Add this line to register the function
    // Add e2m1 conversion function
    m.def("cvt_float_to_e2m1", &cvt_float_to_e2m1, "convert float to e2m1");
}
