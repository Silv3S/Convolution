#include "convolution.h"

int main()
{
    std::vector<std::vector<double>> values = {{1.0,2.0,3.0,4.0,5.0},
        {6.0,7.0,8.0,9.0,10.0}, {11.0,12.0,13.0,14.0,15.0}};
        
    Matrix<double> someMatrix(3,5, values);
    Matrix<double> dilatedMatrix = DilateMatrix(someMatrix,2);
    Matrix<double> paddedMatrix = AddPadding(someMatrix, replicatePadding, 5, 5);

    someMatrix.print();
    dilatedMatrix.print();
    paddedMatrix.print();

    return 0;
}