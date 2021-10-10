#pragma once

#include "matrix.h"

enum paddingOptions {
    noPadding  = 0,
    zeroPadding = 1,
    replicatePadding = 2,
    mirrorPadding = 3
};

template <typename T>
class Convolution
{    
    public:
        Matrix<T> AddPadding(Matrix<T>& matrix, paddingOptions padding, unsigned kernelWidth, unsigned kernelHeight);
        static Matrix<T> Convolve(Matrix<T>& matrix, Matrix<T>& kernel, paddingOptions padding, unsigned stride = 1);
};


template <typename T>
Matrix<T> Convolve(Matrix<T>& matrix, Matrix<T>& kernel, paddingOptions padding, unsigned stride = 1)
{
    
    Matrix<T> paddingMatrix = AddPadding(matrix, padding, kernel.getColsCount(),kernel.getRowsCount());

    matrix.print();
    std::cout << std::endl << "Hello there"<< stride << "  i " << padding << std::endl ;
    kernel.print();

    return matrix;
}

template <typename T>
Matrix<T> AddPadding(Matrix<T>& matrix, paddingOptions padding, unsigned kernelWidth, unsigned kernelHeight)
{
    if(padding == noPadding)
    {
        return matrix;
    }

    unsigned paddingWidth = (kernelWidth - 1) / 2;
    unsigned paddingHeight = (kernelHeight - 1) / 2;
    Matrix<T> paddingMatrix = Matrix<T>(matrix.getRowsCount() + 2 * paddingHeight, matrix.getColsCount() + 2 * paddingWidth);
    
    for (unsigned i = 0; i < matrix.getRowsCount();  i++)
    {
        for (unsigned j = 0; j < matrix.getColsCount();  j++)
        {
            paddingMatrix(i + paddingWidth, j + paddingHeight) = matrix(i,j);
        }
    }
    
    switch(padding)
    {
        case replicatePadding:
        {
            // padding left and right
            for (unsigned i = 0; i < matrix.getRowsCount(); i++)
            {
                T currentLeftValue = matrix(i, 0);
                T currentRightValue = matrix(i, matrix.getColsCount()-1);
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i + paddingHeight, j) = currentLeftValue;
                    paddingMatrix(i + paddingHeight, j + matrix.getColsCount() + paddingWidth) = currentRightValue;
                }
            }

            // padding top and bottom
            for (unsigned j = 0; j < matrix.getColsCount(); j++)
            {
                T currentTopValue = matrix(matrix.getRowsCount()-1, j);
                T currentBottomValue = matrix(0, j);

                for (unsigned i = 0; i < paddingHeight; i++)
                {
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight, j + paddingWidth)  = currentTopValue;
                    paddingMatrix(i, j + paddingWidth) = currentBottomValue;
                }
            }

            // padding corners
            T leftTopValue = matrix(0, 0);
            T rightTopValue = matrix(matrix.getRowsCount() - 1, 0);
            T leftBottomValue = matrix(0, matrix.getColsCount() - 1);
            T rightBottomValue = matrix(matrix.getRowsCount() - 1, matrix.getColsCount() - 1);

            for (unsigned i = 0; i < paddingHeight; i++)
            {              
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i, j) = leftTopValue;
                    paddingMatrix(i, j + matrix.getColsCount() + paddingWidth) = rightTopValue;
                }
            }

            for (unsigned i = paddingMatrix.getRowsCount() - 1; i > paddingMatrix.getRowsCount() - 1 - paddingHeight; i--)
            {              
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i, j) = leftBottomValue;
                    paddingMatrix(i, j + matrix.getColsCount() + paddingWidth) = rightBottomValue;
                }
            }

        }
        break;
        
        case mirrorPadding:
        {
         
        }
        break;
    }

    return paddingMatrix;
}