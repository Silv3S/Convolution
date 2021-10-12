#include "convolution.h"

int main()
{
    std::vector<std::vector<float>> matrixValues = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        };

    std::vector<std::vector<float>> kernelValues = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
        };
        
    Matrix<float> originalMatrix(matrixValues.size(), matrixValues[0].size(), matrixValues);
    Matrix<float> originalKernel(kernelValues.size(), kernelValues[0].size(), kernelValues);
 
    originalMatrix.print();
    originalKernel.print();

    Matrix<float> result = Convolve(originalMatrix, originalKernel, zeroPadding);
    result.print();

    return 0;
}