#include "matrix.h"

int main()
{
    std::vector<std::vector<double>> values = {{1.0,2.0,3.0}, {4.0,5.0,6.0}, {3.0,2.0,1.0}};
    Matrix<double> mat2(3,3, values);

    mat2.print();
    return 0;
}