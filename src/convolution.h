#pragma once

#include "matrix.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/example_utils.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

enum paddingOptions {
    noPadding  = 0,
    zeroPadding = 1,
    replicatePadding = 2,
    mirrorPadding = 3
};

template <typename T>
class Convolution
{
    private:
        Matrix<T> GetFragmentCoveredByKernel(Matrix<T>& paddingMatrix, Matrix<T>& matrixCoveredByKernel, unsigned omi, unsigned omj);
        T SumNeighborhood(Matrix<T>& matrixCoveredByKernel, Matrix<T>& kernel, T bias);
    public:
        Matrix<T> AddPadding(Matrix<T>& matrix, paddingOptions padding, unsigned kernelWidth, unsigned kernelHeight);
        static Matrix<T> Convolve(Matrix<T>& matrix, Matrix<T>& kernel, paddingOptions padding, unsigned stride = 1);
};

template <typename T>
Matrix<T> Convolve(Matrix<T>& matrix, Matrix<T>& kernel, paddingOptions padding, unsigned stride = 1, T bias = 0.0)
{
    Matrix<T> paddingMatrix = AddPadding(matrix, padding, kernel.getColsCount(),kernel.getRowsCount());
    Matrix<T> featureMap(matrix.getRowsCount() / stride, matrix.getColsCount() / stride);
    Matrix<T> matrixCoveredByKernel(kernel.getRowsCount(), kernel.getColsCount());

    for (unsigned i = 0; i < matrix.getRowsCount(); i += stride)
    {
        for (unsigned j = 0; j < matrix.getColsCount(); j += stride)
        {
            matrixCoveredByKernel = GetFragmentCoveredByKernel(paddingMatrix, matrixCoveredByKernel, i, j);
            featureMap(i,j) = SumNeighborhood(matrixCoveredByKernel, kernel, bias);
        }
    }

    return featureMap;
}

template <typename T>
Matrix<T> AddPadding(Matrix<T>& matrix, paddingOptions padding, unsigned kernelWidth, unsigned kernelHeight)
{
    if(padding == noPadding)
    {
        return matrix;
    }

    unsigned paddingWidth = (kernelWidth - 1) / 2;
    unsigned paddingHeight = (kernelHeight - 1) / 2;
    Matrix<T> paddingMatrix = Matrix<T>(matrix.getRowsCount() + 2 * paddingHeight, matrix.getColsCount() + 2 * paddingWidth);

    for (unsigned i = 0; i < matrix.getRowsCount();  i++)
    {
        for (unsigned j = 0; j < matrix.getColsCount();  j++)
        {
            paddingMatrix(i + paddingWidth, j + paddingHeight) = matrix(i,j);
        }
    }

    switch(padding)
    {
        case replicatePadding:
        {
            // padding left and right
            for (unsigned i = 0; i < matrix.getRowsCount(); i++)
            {
                T currentLeftValue = matrix(i, 0);
                T currentRightValue = matrix(i, matrix.getColsCount()-1);
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i + paddingHeight, j) = currentLeftValue;
                    paddingMatrix(i + paddingHeight, j + matrix.getColsCount() + paddingWidth) = currentRightValue;
                }
            }

            // padding top and bottom
            for (unsigned j = 0; j < matrix.getColsCount(); j++)
            {
                T currentTopValue = matrix(matrix.getRowsCount()-1, j);
                T currentBottomValue = matrix(0, j);

                for (unsigned i = 0; i < paddingHeight; i++)
                {
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight, j + paddingWidth)  = currentTopValue;
                    paddingMatrix(i, j + paddingWidth) = currentBottomValue;
                }
            }

            // padding corners
            T leftTopValue = matrix(0, 0);
            T rightTopValue = matrix(matrix.getRowsCount() - 1, 0);
            T leftBottomValue = matrix(0, matrix.getColsCount() - 1);
            T rightBottomValue = matrix(matrix.getRowsCount() - 1, matrix.getColsCount() - 1);

            for (unsigned i = 0; i < paddingHeight; i++)
            {
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i, j) = leftTopValue;
                    paddingMatrix(i, j + matrix.getColsCount() + paddingWidth) = rightTopValue;
                }
            }

            for (unsigned i = paddingMatrix.getRowsCount() - 1; i > paddingMatrix.getRowsCount() - 1 - paddingHeight; i--)
            {
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i, j) = leftBottomValue;
                    paddingMatrix(i, j + matrix.getColsCount() + paddingWidth) = rightBottomValue;
                }
            }

        }
        break;

        case mirrorPadding:
        {
            // padding left and right
            for (unsigned i = 0; i < matrix.getRowsCount(); i++)
            {
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i + paddingHeight, j) = matrix(i, paddingWidth - j - 1);
                    paddingMatrix(i + paddingHeight, j + matrix.getColsCount() + paddingWidth) = matrix(i, matrix.getColsCount() - 1- j);
                }
            }

            // padding top and bottom
            for (unsigned j = 0; j < matrix.getColsCount(); j++)
            {

                for (unsigned i = 0; i < paddingHeight; i++)
                {
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight, j + paddingWidth)  = matrix(matrix.getRowsCount() - 1 - i, j);;
                    paddingMatrix(i, j + paddingWidth) = matrix(paddingHeight - i - 1, j);;
                }
            }

            // padding corners

            for (unsigned i = 0; i < paddingHeight; i++)
            {
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i, j) = matrix(paddingHeight - i - 1, paddingWidth - j - 1);
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight , j + paddingWidth + matrix.getColsCount()) = matrix(matrix.getRowsCount() - 1 - i , matrix.getColsCount() - 1 - j);
                    paddingMatrix(i, j + matrix.getColsCount() + paddingWidth) = matrix(paddingHeight - i - 1, matrix.getColsCount() - j - 1);
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight, j) = matrix(matrix.getRowsCount() - i - 1, paddingWidth - j - 1);
                }
            }
        }
        break;
    }
    return paddingMatrix;
}

template <typename T>
Matrix<T> GetFragmentCoveredByKernel(Matrix<T>& paddingMatrix, Matrix<T>& matrixCoveredByKernel, unsigned omi, unsigned omj)
{
    for (unsigned i = 0; i < matrixCoveredByKernel.getRowsCount(); i++)
    {
        for (unsigned j = 0; j < matrixCoveredByKernel.getColsCount(); j++)
        {
             matrixCoveredByKernel(i, j) = paddingMatrix(omi + i, omj + j);
        }
    }

    return matrixCoveredByKernel;
}

template <typename T>
T SumNeighborhood(Matrix<T>& matrixCoveredByKernel, Matrix<T>& kernel, T bias = 0.0)
{
    T sum = bias;
    for (unsigned i = 0; i < matrixCoveredByKernel.getRowsCount(); i++)
    {
        for (unsigned j = 0; j < matrixCoveredByKernel.getColsCount(); j++)
        {
            sum += matrixCoveredByKernel(i,j) * kernel(i,j);
        }
    }
    return sum;
}

Matrix<float> ConvolutionOneDNN(Matrix<float> userData, Matrix<float> kernel, float bias = 0, unsigned strides = 1)
{
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    const dnnl::memory::dim
        N = 1,                                  // batch size
        IC = 1,                                 // input channels
        IH = userData.getRowsCount(),           // input height
        IW = userData.getColsCount(),           // input width
        OC = 1,                                 // output channels
        KH = kernel.getRowsCount(),             // weights height
        KW = kernel.getColsCount(),             // weights width
        PH_L = 1,                               // height padding: left
        PH_R = 1,                               // height padding: right
        PW_L = 1,                               // width padding: left
        PW_R = 1,                               // width padding: right
        SH = 1,                                 // height-wise stride
        SW = 1,                                 // width-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    std::vector<float> src_data = Flatten(userData);
    std::vector<float> weights_data = Flatten(kernel);
    std::vector<float> bias_data{bias};
    std::vector<float> dst_data(product(dst_dims));

    auto user_src_mem = memory({src_dims, dt::f32, tag::nchw}, engine);
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::oihw}, engine);
    auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, engine);

    auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);

    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);

    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
            padding_dims_r);

    auto conv_pd = convolution_forward::primitive_desc(conv_desc, engine);

    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
            .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    auto conv_prim = convolution_forward(conv_pd);

    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    conv_prim.execute(engine_stream, conv_args);

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;

    engine_stream.wait();
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
    return Matrix<float>(dst_data, OW);
}

Matrix<float> QuantizedConvolutionOneDNN(Matrix<float> userData, Matrix<float> kernel, float bias = 0, unsigned strides = 1)
{
    
} 