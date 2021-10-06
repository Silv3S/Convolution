#include "Matrix.h"
using namespace std;

template <class T>
Matrix<T>::Matrix(unsigned rowNum, unsigned colNum, T initial)
{
    rows = rowNum;
    cols = colNum;
    data.resize(rows);
    for (unsigned i = 0; i < rows; i++)
    {
        data[i].resize(cols, initial);
    }
}

template <class T>
Matrix<T>::Matrix(unsigned rowNum, unsigned colNum)
{
    rows = rowNum;
    cols = colNum;
    data.resize(rows);
    for (unsigned i = 0; i < rows; i++)
    {
        data[i].resize(cols);
        for (unsigned j = 0; j < cols; j++)
        {
            data[i][j] = (rand() % 10) * (rand() % 10);
        }
    }
}

// diagonal
template <class T>
Matrix<T>::Matrix(unsigned n)
{
    rows = n;
    cols = n;
    data.resize(rows);
    for (unsigned i = 0; i < rows; i++)
    {
        data[i].resize(cols);
        for (unsigned j = 0; j < cols; j++)
        {
            data[i][j] = (i == j) ? 1 : 0;
        }
    }
}

template <class T>
void Matrix<T>::print() const
{
    for (unsigned i = 0; i < rows; i++)
    {
        for (unsigned j = 0; j < cols; j++)
        {
            cout << std::setw(3) << data[i][j] << " ";
        }
        cout << endl;
    }
}

template <class T>
T &Matrix<T>::operator()(unsigned &row, unsigned &col)
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
Matrix<T> Matrix<T>::compute_grad(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C)
{
    Matrix<T> AB = Matrix<T>(0,0);
    try
    {
        AB = A * B;
        AB.print();
    }
    catch (const invalid_argument& e)
    {
        cout << e.what() << endl;
    }
    int totalEntries = C.rows * C.cols;
    Matrix<T> diagonal = Matrix<T>(totalEntries);
    Matrix<T> deltaAB_deltaC = A * diagonal;
    return deltaAB_deltaC;
}

template<class T>
Matrix<T> Matrix<T>::convolute(Matrix<T>& kernel)
{
    int cutoutRows = (kernel.rows-1)/2;
    int cotoutCols = (kernel.cols-1)/2;

    cout << kernel.cols << endl;
    cout << kernel.rows << endl;

    cout << cotoutCols << endl;
    cout << cutoutRows << endl;
    
    Matrix<T> empty = Matrix<T>(0,0);
    return empty; 
}