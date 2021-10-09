#include<bits/stdc++.h>

template <class T>
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

template <class T>
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

template <class T>
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

template <class T>
T &Matrix<T>::operator()(unsigned row, unsigned col)
{
    return this->data[row][col];
}

template <class T>
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

template<class T>
Matrix<T> Matrix<T>::DilateMatrix(Matrix<T>& A, unsigned tiles)
{
   if(tiles == 0 || tiles == 1 )
   {
       return A;
   }
   
   if(ceil(log2(tiles)) == floor(log2(tiles)))
   {
        unsigned dilatedRows = tiles * A.rows + (tiles - 1);
        unsigned dilatedCols = tiles * A.cols + (tiles - 1);

        Matrix<T> dilatedMatrix = Matrix<T>(dilatedRows, dilatedCols);

        for (unsigned i = 0; i < A.rows; i++)
        {
            for (unsigned j = 0; j < A.cols; j++)
            {
                dilatedMatrix( (i+1)*tiles-1, (j+1)*tiles-1 ) = A(i,j);
            }
        }        
        return dilatedMatrix;
    }
   
   return A;

}

template<class T>
Matrix<T> Matrix<T>::Conv2D(Matrix<T>& kernel)
{
    int cutoutRows = (kernel.rows-1)/2;
    int cotoutCols = (kernel.cols-1)/2;

    std::cout << kernel.cols << std::endl;
    std::cout << kernel.rows << std::endl;

    std::cout << cotoutCols << std::endl;
    std::cout << cutoutRows << std::endl;
    
    Matrix<T> empty = Matrix<T>(0,0);
    return empty; 
}

template<class T>
Matrix<T>::~Matrix() {}