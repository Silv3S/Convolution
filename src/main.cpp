#include "Matrix.h"
#include <iostream>

int main()
{
    std::vector<std::vector<double>> values = {{1.0,2.0,3.0}, {4.0,5.0,6.0}, {3.0,2.0,1.0}};
    Matrix<double> mat2(3,3, values);
    Matrix<double> dilatedMat2 = Matrix<double>::DilateMatrix(mat2,4);
    dilatedMat2.print();
   
    return 0;
}