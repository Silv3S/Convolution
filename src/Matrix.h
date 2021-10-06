#include <iostream>
#include <vector>
#include <iomanip>
using std::vector;

// Solution to linking error
#ifndef _MATRIX_H_
#define _MATRIX_H_
template <class T>
class Matrix
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

    private:
        unsigned rows;
        unsigned cols;
        vector<vector<T>> data;

    public:
        Matrix(unsigned, unsigned, T);
        Matrix(unsigned, unsigned);
        Matrix(unsigned n);
        void print() const;
        T &operator()(unsigned &, unsigned &);
        Matrix operator*(Matrix &);
        static Matrix<T> compute_grad(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C);
        Matrix<T> convolute(Matrix<T>& kernel);
};
#endif