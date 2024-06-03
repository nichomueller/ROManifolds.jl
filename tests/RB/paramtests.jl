using Gridap
using Test
using DrWatson
using Gridap.MultiField
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using SparseArrays
using ArraysOfArrays

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
