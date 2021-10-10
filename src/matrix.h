#include <iostream>
#include <vector>
#include <iomanip>
#include <bits/stdc++.h>

#ifndef _MATRIX_H_
#define _MATRIX_H_

template <typename T>
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
};

template <typename T>
Matrix<T>::Matrix(unsigned _rows, unsigned _cols)
{
    rows = _rows;
    cols = _cols;

    data.resize(rows);
    for (unsigned i = 0; i < data.size(); i++)
    {
        data[i].resize(_cols, 0);
    }
}

template <typename T>
Matrix<T>::Matrix(unsigned _rows, unsigned _cols, const std::vector<std::vector<T>>& _data)
{
    rows = _rows;
    cols = _cols;

    data.resize(rows);
    for (unsigned i = 0; i < data.size(); i++)
    {
        data[i].resize(_cols);
    }

    data = _data;
}

template <typename T>
void Matrix<T>::print() const
{
    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            std::cout << std::setw(3) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
T &Matrix<T>::operator()(unsigned row, unsigned col)
{
    return this->data[row][col];
}

template <typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> &B)
{
    if (cols != B.rows)
    {
        throw std::invalid_argument("The dimensions of the matrix do not allow multiplication");
    }
    Matrix result(rows, B.cols, 0);
    unsigned i, j, k;
    T temp = 0.0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < B.cols; j++)
        {
            temp = 0.0;
            for (k = 0; k < cols; k++)
            {
                temp += data[i][k] * B(k, j);
            }
            result(i, j) = temp;
        }
    }
    return result;
}

#endif