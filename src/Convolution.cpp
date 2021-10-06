#include "Matrix.h"
#include "Matrix.cpp"
using namespace std;

int main()
{
    float sourceImageWidth = 32, sourceImageHeight = 24;
    float kernelWidth = 3;
    float kernelHeight = 5;

    Matrix<float> sourceImage = Matrix<float>(sourceImageHeight, sourceImageWidth);
    // sourceImage.print();

    // cout << endl << endl;

    Matrix<float> kernel = Matrix<float>(kernelHeight, kernelWidth);
    // kernel.print();

    sourceImage.convolute(kernel);
}