#include "convolution.h"

std::vector<std::pair<Matrix<float>, Matrix<float>>> GetExampleData()
{
    std::vector<std::pair<Matrix<float>, Matrix<float>>> exampleData;

    exampleData.push_back(
        std::make_pair(
            Matrix<float>(3, 3, 
            {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
            }),
            Matrix<float>(3, 3,
            {
                { 1.0,  2.0,  1.0},
                { 0.0,  0.0,  0.0},
                {-1.0, -2.0, -1.0}
            })
        )
    );

    exampleData.push_back(
        std::make_pair(
            Matrix<float>(4, 4, 
            {
                {0.0, 2.0, 1.0, 3.0},
                {1.0, 1.0, 0.0, 2.0},
                {1.0, 0.0, 4.0, 1.0},
                {4.0, 2.0, 0.0, 1.0}
            }),
            Matrix<float>(3, 3,
            {
                { 0.0,  1.0,  0.0},
                { 1.0, -1.0,  1.0},
                { 0.0,  1.0,  0.0}
            })
        )
    );    

    exampleData.push_back(
        std::make_pair(
            Matrix<float>(5, 5, 
            {
                {2.0, 4.0, 9.0, 1.0, 4.0},
                {2.0, 1.0, 4.0, 4.0, 6.0},
                {1.0, 1.0, 2.0, 9.0, 2.0},
                {7.0, 3.0, 5.0, 1.0, 3.0},
                {2.0, 3.0, 4.0, 8.0, 5.0}
            }),
            Matrix<float>(3, 3,
            {
                { 1.0,  2.0,  3.0},
                {-4.0,  7.0,  4.0},
                { 2.0, -5.0,  1.0}
            })
        )
    );

    return exampleData;
}

void QuantizeAndDequantize(Matrix<float> src_matrix)
{
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    const dnnl::memory::dim 
        N = 1,
        IC = 1,
        IH = src_matrix.getRowsCount(),
        IW = src_matrix.getColsCount();

    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    
    float scaleIn = 127 / src_matrix.AbsoluteMaximumValue();
    
    std::vector<float> src_data = src_matrix.Flatten();
    std::vector<int8_t> dst_data(product(src_dims));
    std::vector<float> src_data_scaled(product(src_dims));

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(src_dims, memory::data_type::s8, memory::format_tag::nhwc);
    auto src_scaled_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);
    auto src_scaled_mem = memory(src_scaled_md, engine);

    write_to_dnnl_memory(src_data.data(), src_mem);

    std::vector<float> scalesIn{scaleIn};
    std::vector<float> scalesOut{1 / scaleIn};

    primitive_attr reorder_attr;    
    reorder_attr.set_output_scales(0 | (1 << IC), scalesIn);
    primitive_attr reorder_attr_scaled;    
    reorder_attr_scaled.set_output_scales(0 | (1 << IC), scalesOut);

    auto reorder_pd = reorder::primitive_desc(engine, src_md, engine, dst_md, reorder_attr);
    auto reorder_pd_scaled = reorder::primitive_desc(engine, dst_md, engine, src_scaled_md, reorder_attr_scaled);

    auto reorder_prim = reorder(reorder_pd);
    auto reorder_prim_scal = reorder(reorder_pd_scaled);

    std::unordered_map<int, memory> reorder_args;
        reorder_args.insert({DNNL_ARG_SRC, src_mem});
        reorder_args.insert({DNNL_ARG_DST, dst_mem});
    std::unordered_map<int, memory> reorder_args_scaled;
        reorder_args_scaled.insert({DNNL_ARG_SRC, dst_mem});
        reorder_args_scaled.insert({DNNL_ARG_DST, src_scaled_mem});

    reorder_prim.execute(engine_stream, reorder_args);
    reorder_prim_scal.execute(engine_stream, reorder_args_scaled);

    engine_stream.wait();

    read_from_dnnl_memory(dst_data.data(), dst_mem);
    read_from_dnnl_memory(src_data_scaled.data(), src_scaled_mem);

    std::cout << std::endl;
    std::cout << "Original F32 data " << std::endl;
    Matrix<float>(src_data, IW).print();
    std::cout << std::endl;

    std::cout << "Quantized F32 to INT8 " << std::endl;
    Matrix<int8_t>(dst_data, IW).print();
    std::cout << std::endl;

    std::cout << "Requantized INT8 back to F32 " << std::endl;
    Matrix<float>(src_data_scaled, IW).print();
    std::cout << std::endl;
}