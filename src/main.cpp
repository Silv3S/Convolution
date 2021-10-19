#include "convolution.h"

void ConsoleWriteLine(std::string text)
{
    std::cout << text << std::endl; 
}

int main(int argc, char **argv)
{        
    std::vector<Matrix<float>> exampleMatrices;
    std::vector<Matrix<float>> exampleKernels;    

    std::vector<Matrix<int8_t>> exampleMatricesI;
    std::vector<Matrix<int8_t>> exampleKernelsI;    

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

    exampleMatricesI.push_back(
        Matrix<int8_t>(3, 3, 
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        }
    ));
    
    exampleKernelsI.push_back(
        Matrix<int8_t>(3, 3,
        {
            { 1,  2,  1},
            { 0,  0,  0},
            {-1, -2, -1}
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

    exampleMatricesI.push_back(
        Matrix<int8_t>(4, 4, 
        {
            {0, 2, 1, 3},
            {1, 1, 0, 2},
            {1, 0, 4, 1},
            {4, 2, 0, 1}
        }
    ));
    
    exampleKernelsI.push_back(
        Matrix<int8_t>(3, 3,
        {
            { 0,  1,  0},
            { 1, -1,  1},
            { 0,  1,  0}
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

    exampleMatricesI.push_back(
        Matrix<int8_t>(5, 5, 
        {
            {2, 4, 9, 1, 4},
            {2, 1, 4, 4, 6},
            {1, 1, 2, 9, 2},
            {7, 3, 5, 1, 3},
            {2, 3, 4, 8, 5}
        }
    ));
    
    exampleKernelsI.push_back(
        Matrix<int8_t>(3, 3,
        {
            { 1,  2,  3},
            {-4,  7,  4},
            { 2, -5,  1}
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
        ConsoleWriteLine("-------- \n");

        ConsoleWriteLine("int8 matrix"); 
        exampleMatricesI[f].print();

        ConsoleWriteLine("int8 kernel"); 
        exampleKernelsI[f].print();

        ConsoleWriteLine("convolution.h - reference int8 convolution result"); 
        Matrix<int8_t> intRefResult = Convolve(exampleMatricesI[f], exampleKernelsI[f], zeroPadding);
        intRefResult.print();

        ConsoleWriteLine("oneDNN - int8 convolution result"); 
        Matrix<int8_t> intOneResult = ConvolutionOneDNN(exampleMatricesI[f], exampleKernelsI[f]);
        intOneResult.print();
        ConsoleWriteLine("-------- \n");
    }
        
    return 0;
}