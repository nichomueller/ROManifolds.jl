using Gridap
using Test
using DrWatson
using SparseArrays
using ArraysOfArrays
using Gridap.MultiField
using Mabla.FEM
using Mabla.FEM.ParamDataStructures

A = sprand(Float64,100,100,0.5)
B = sprand(Float64,100,100,0.5)
AB = MatrixOfSparseMatricesCSC([A,A])

AA = [A,A]
m,n = ParamDataStructures.innersize(AA)
colptr,rowval,nzval = ParamDataStructures.innerpattern(AA)
matval = Matrix{Float64}(undef,ParamDataStructures.innersize(nzval)...,length(nzval))
copyto!(eachcol(matval),nzval)

mat = Matrix{Matrix{Float64}}(undef,2,2)
mat[1] = rand(3,3)
mat[2] = rand(3,3)
mat[3] = rand(3,3)
mat[4] = rand(3,3)

Mat = ArrayOfSimilarArrays(mat)

Vec = ArrayOfSimilarArrays([rand(3) for _ = 1:4])
Mat*Vec
