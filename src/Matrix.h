#include <iostream>
#include <vector>
#include <iomanip>

#ifndef _MATRIX_H_
#define _MATRIX_H_

template <class T>
class Matrix
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

    private:
        unsigned rows;
        unsigned cols;
        std::vector<std::vector<T>> data;

    public:
        Matrix(unsigned, unsigned _cols);
        Matrix(unsigned, unsigned _cols, const std::vector<std::vector<T>>& _data);
        void print() const;
        T &operator()(unsigned, unsigned);
        Matrix operator*(Matrix &);
        Matrix<T> Conv2D(Matrix<T>& kernel);
        virtual ~Matrix();

        static Matrix<T> DilateMatrix(Matrix<T>& A, unsigned tiles);
};

#include "Matrix.cpp"

#endif