#include "convolution.h"
#include "sandbox.cpp"

int main(int argc, char **argv)
{        
    std::vector<std::pair<Matrix<float>, Matrix<float>>> exampleData = GetExampleData();
    
    for (unsigned i = 0; i < exampleData.size(); i++)
    {
        std::cout << std::endl << "Example " << i + 1 << std::endl;
        std::cout << "float 32 data" << std::endl;
        exampleData[i].first.print();

        std::cout << "float 32 kernel" << std::endl;
        exampleData[i].second.print();

        std::cout << "convolution.h - reference float32 convolution result" << std::endl;
        Matrix<float> floatRefResult = Convolve(exampleData[i].first, exampleData[i].second, zeroPadding);
        floatRefResult.print();
        
        std::cout << "oneDNN - float32 convolution result" << std::endl;
        Matrix<float> floatOneResult = ConvolutionOneDNN(exampleData[i].first, exampleData[i].second);
        floatOneResult.print();

        std::cout << "oneDNN - int8 convolution result" << std::endl;
        Matrix<float> intOneResult = QuantizedConvolutionOneDNN(exampleData[i].first, exampleData[i].second);
        intOneResult.print();
    } 
}