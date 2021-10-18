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
    private:
        Matrix<T> GetFragmentCoveredByKernel(Matrix<T>& paddingMatrix, Matrix<T>& matrixCoveredByKernel, unsigned omi, unsigned omj);
        T SumNeighborhood(Matrix<T>& matrixCoveredByKernel, Matrix<T>& kernel, T bias);
    public:
        Matrix<T> AddPadding(Matrix<T>& matrix, paddingOptions padding, unsigned kernelWidth, unsigned kernelHeight);
        static Matrix<T> Convolve(Matrix<T>& matrix, Matrix<T>& kernel, paddingOptions padding, unsigned stride = 1);
};

template <typename T>
Matrix<T> Convolve(Matrix<T>& matrix, Matrix<T>& kernel, paddingOptions padding, unsigned stride = 1, T bias = 0.0)
{
    Matrix<T> paddingMatrix = AddPadding(matrix, padding, kernel.getColsCount(),kernel.getRowsCount());
    Matrix<T> featureMap(matrix.getRowsCount() / stride, matrix.getColsCount() / stride);
    Matrix<T> matrixCoveredByKernel(kernel.getRowsCount(), kernel.getColsCount());
  
    for (unsigned i = 0; i < matrix.getRowsCount(); i += stride)
    {
        for (unsigned j = 0; j < matrix.getColsCount(); j += stride)
        {
            matrixCoveredByKernel = GetFragmentCoveredByKernel(paddingMatrix, matrixCoveredByKernel, i, j);
            featureMap(i,j) = SumNeighborhood(matrixCoveredByKernel, kernel, bias);
        }
    }

    return featureMap;
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
            // padding left and right
            for (unsigned i = 0; i < matrix.getRowsCount(); i++)
            {
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i + paddingHeight, j) = matrix(i, paddingWidth - j - 1);
                    paddingMatrix(i + paddingHeight, j + matrix.getColsCount() + paddingWidth) = matrix(i, matrix.getColsCount() - 1- j);
                }
            }

            // padding top and bottom
            for (unsigned j = 0; j < matrix.getColsCount(); j++)
            {

                for (unsigned i = 0; i < paddingHeight; i++)
                {
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight, j + paddingWidth)  = matrix(matrix.getRowsCount() - 1 - i, j);;
                    paddingMatrix(i, j + paddingWidth) = matrix(paddingHeight - i - 1, j);;
                }
            }

            // padding corners

            for (unsigned i = 0; i < paddingHeight; i++)
            {              
                for (unsigned j = 0; j < paddingWidth; j++)
                {
                    paddingMatrix(i, j) = matrix(paddingHeight - i - 1, paddingWidth - j - 1);
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight , j + paddingWidth + matrix.getColsCount()) = matrix(matrix.getRowsCount() - 1 - i , matrix.getColsCount() - 1 - j);
                    paddingMatrix(i, j + matrix.getColsCount() + paddingWidth) = matrix(paddingHeight - i - 1, matrix.getColsCount() - j - 1);
                    paddingMatrix(i + matrix.getRowsCount() + paddingHeight, j) = matrix(matrix.getRowsCount() - i - 1, paddingWidth - j - 1);   
                }
            }         
        }
        break;
    }
    return paddingMatrix;
}

template <typename T>
Matrix<T> GetFragmentCoveredByKernel(Matrix<T>& paddingMatrix, Matrix<T>& matrixCoveredByKernel, unsigned omi, unsigned omj)
{
    for (unsigned i = 0; i < matrixCoveredByKernel.getRowsCount(); i++)
    {
        for (unsigned j = 0; j < matrixCoveredByKernel.getColsCount(); j++)
        {            
             matrixCoveredByKernel(i, j) = paddingMatrix(omi + i, omj + j);   
        }        
    }    
    
    return matrixCoveredByKernel;
}

template <typename T>
T SumNeighborhood(Matrix<T>& matrixCoveredByKernel, Matrix<T>& kernel, T bias = 0.0)
{
    T sum = bias;
    for (unsigned i = 0; i < matrixCoveredByKernel.getRowsCount(); i++)
    {
        for (unsigned j = 0; j < matrixCoveredByKernel.getColsCount(); j++)
        {
            sum += matrixCoveredByKernel(i,j) * kernel(i,j);   
        }        
    }    
    return sum;
}