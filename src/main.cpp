#include "convolution.h"

void ConsoleWriteLine(std::string text)
{
    std::cout << text << std::endl; 
}

int main(int argc, char **argv)
{        
    std::vector<Matrix<float>> exampleMatrices;
    std::vector<Matrix<float>> exampleKernels;    

    /* #region Examples  */
    exampleMatrices.push_back(
        Matrix<float>(3, 3, 
        {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        }
    ));
    
    exampleKernels.push_back(
        Matrix<float>(3, 3,
        {
            { 1.0,  2.0,  1.0},
            { 0.0,  0.0,  0.0},
            {-1.0, -2.0, -1.0}
        }
    ));
    
    // Second
    exampleMatrices.push_back(
        Matrix<float>(4, 4, 
        {
            {0.0, 2.0, 1.0, 3.0},
            {1.0, 1.0, 0.0, 2.0},
            {1.0, 0.0, 4.0, 1.0},
            {4.0, 2.0, 0.0, 1.0}
        }
    ));
    
    exampleKernels.push_back(
        Matrix<float>(3, 3,
        {
            { 0.0,  1.0,  0.0},
            { 1.0, -1.0,  1.0},
            { 0.0,  1.0,  0.0}
        }
    ));

    // Third
    exampleMatrices.push_back(
        Matrix<float>(5, 5, 
        {
            {2.0, 4.0, 9.0, 1.0, 4.0},
            {2.0, 1.0, 4.0, 4.0, 6.0},
            {1.0, 1.0, 2.0, 9.0, 2.0},
            {7.0, 3.0, 5.0, 1.0, 3.0},
            {2.0, 3.0, 4.0, 8.0, 5.0}
        }
    ));
    
    exampleKernels.push_back(
        Matrix<float>(3, 3,
        {
            { 1.0,  2.0,  3.0},
            {-4.0,  7.0,  4.0},
            { 2.0, -5.0,  1.0}
        }
    ));

    /* #endregion */
        
    ConsoleWriteLine("\n--------");
    for (unsigned f = 0; f < exampleMatrices.size(); f++)
    {
        std::cout << "Example " << f + 1 << std::endl;
        ConsoleWriteLine("float32 matrix"); 
        exampleMatrices[f].print();

        ConsoleWriteLine("float32 kernel"); 
        exampleKernels[f].print();

        ConsoleWriteLine("convolution.h - reference float32 convolution result"); 
        Matrix<float> floatRefResult = Convolve(exampleMatrices[f], exampleKernels[f], zeroPadding);
        floatRefResult.print();
        
        ConsoleWriteLine("oneDNN - float32 convolution result"); 
        Matrix<float> floatOneResult = ConvolutionOneDNN(exampleMatrices[f], exampleKernels[f]);
        floatOneResult.print();

        ConsoleWriteLine("oneDNN - int8 convolution result"); 
        // Matrix<float> intOneResult = QuantizedConvolutionOneDNN(exampleMatrices[f], exampleKernels[f]);
        // intOneResult.print();
        ConsoleWriteLine("-------- \n");
    }
        
    return 0;
}