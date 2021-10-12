#include "convolution.h"

int main()
{
    std::vector<std::vector<double>> values = {
        {10.0,1.0,1.0,1.0,1.0,1.0,11.0},
        {2.0,2.0,2.0,2.0,2.0,2.0,2.0},
        {3.0,3.0,3.0,3.0,3.0,3.0,3.0},
        {4.0,4.0,4.0,4.0,4.0,4.0,4.0},
        {15.0,5.0,5.0,5.0,5.0,5.0,51.0}
        };
        
    Matrix<double> someMatrix(5,7, values);
    Matrix<double> dilatedMatrix = DilateMatrix(someMatrix,2);
    Matrix<double> paddedMatrix = AddPadding(someMatrix, replicatePadding, 7, 7);

    someMatrix.print();
    dilatedMatrix.print();
    paddedMatrix.print();

    return 0;
}