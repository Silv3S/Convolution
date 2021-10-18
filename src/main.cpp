#include "convolution.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/example_utils.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;


void ConsoleWriteLine(std::string text)
{
    std::cout << text << std::endl; 
}

Matrix<float> ConvolutionOneDNN(Matrix<float> userData, Matrix<float> kernel, float bias = 0.0, int strides = 1)
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
   
   
    auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
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
    Matrix<float> result(dst_data, OW);
    return result;

}

int main(int argc, char **argv)
{

    std::vector<std::vector<float>> matrixValues =
    {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
    };

    std::vector<std::vector<float>> kernelValues =
    {
        { 1.0,  2.0,  1.0},
        { 0.0,  0.0,  0.0},
        {-1.0, -2.0, -1.0}
    };
        
    Matrix<float> userMatrix(matrixValues.size(), matrixValues[0].size(), matrixValues);
    Matrix<float> userKernel(kernelValues.size(), kernelValues[0].size(), kernelValues);
    
    ConsoleWriteLine("User-defined data"); 
    userMatrix.print();

    ConsoleWriteLine("User-defined kernel"); 
    userKernel.print();

    ConsoleWriteLine("Convolution.h - reference convolution result"); 
    Matrix<float> refResult = Convolve(userMatrix, userKernel, zeroPadding);
    refResult.print();
    
    ConsoleWriteLine("oneDNN - float convolution result"); 
    Matrix<float> oneResult = ConvolutionOneDNN(userMatrix, userKernel);
    oneResult.print();

    
    return 0;
}