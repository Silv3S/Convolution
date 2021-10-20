#pragma once

#include <iostream>
#include <vector>
#include <iomanip>
#include <bits/stdc++.h>

template <typename T>
class Matrix
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

    private:
        unsigned rows;
        unsigned cols;
        std::vector<std::vector<T>> data;

    public:
        unsigned getRowsCount();
        unsigned getColsCount();
        Matrix(unsigned, unsigned _cols);
        Matrix(unsigned, unsigned _cols, const std::vector<std::vector<T>>& _data);
        Matrix(const std::vector<T>& _data, unsigned _cols);
        void print() const;
        T &operator()(unsigned, unsigned);
        Matrix operator*(Matrix &);
        Matrix<T> DilateMatrix(Matrix<T>& A, unsigned tiles);
        std::vector<T> Flatten();   
        T AbsoluteMaximumValue();
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
Matrix<T>::Matrix(const std::vector<T>& _inputData, unsigned _cols)
{
    cols = _cols;
    rows = _inputData.size() / _cols;

    data.resize(rows);
    for (unsigned i = 0; i < rows; i++)
    {
        data[i].resize(_cols);
    }

    for (unsigned i = 0; i < _inputData.size(); i++)
    {
        unsigned currentJ = i % _cols;
        unsigned currentI = (i - currentJ) / _cols; 
        data[currentI][currentJ] = _inputData[i];
    }    
}

template <typename T>
void Matrix<T>::print() const
{
    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            std::cout << std::setw(10) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <>
void Matrix<int8_t>::print() const
{
    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            std::cout << std::setw(10) << int(data[i][j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <>
void Matrix<uint8_t>::print() const
{
    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            std::cout << std::setw(10) << int(data[i][j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
unsigned Matrix<T>::getRowsCount()
{
    return rows;
}

template <typename T>
unsigned Matrix<T>::getColsCount()
{
    return cols;
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
    Matrix<T> result(rows, B.cols);
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

template<typename T>
Matrix<T> DilateMatrix(Matrix<T>& A, unsigned tiles)
{
   if(tiles == 0 || tiles == 1 )
   {
       return A;
   }
   
   if(ceil(log2(tiles)) == floor(log2(tiles)))
   {
        unsigned dilatedRows = tiles * A.getRowsCount() + (tiles - 1);
        unsigned dilatedCols = tiles * A.getColsCount() + (tiles - 1);

        Matrix<T> dilatedMatrix = Matrix<T>(dilatedRows, dilatedCols);

        for (unsigned i = 0; i < A.getRowsCount(); i++)
        {
            for (unsigned j = 0; j < A.getColsCount(); j++)
            {
                dilatedMatrix( (i+1)*tiles-1, (j+1)*tiles-1 ) = A(i,j);
            }
        }        
        return dilatedMatrix;
    }
   
   return A;
}

template<typename T>
std::vector<T> Matrix<T>::Flatten() 
{
    std::vector<T> flatVec;

    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            flatVec.push_back(data[i][j]);
        }         
    }

    return flatVec;
}

template<typename T>
T Matrix<T>::AbsoluteMaximumValue()
{
    T maxAbsValue = data[0][0];

    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            if(abs(data[i][j]) > maxAbsValue)
            {
                maxAbsValue = abs(data[i][j]);
            }
        }        
    }    
    return maxAbsValue;
}
