using Gridap
using Gridap.Arrays
using Gridap.Helpers
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using BlockArrays

V1 = ParamArray([rand(3),rand(3)])
V2 = ParamArray([rand(4),rand(4)])
V = mortar([V1,V2])

V[3] = rand(4)

V[findblockindex.(axes(V),3)...] = rand(4)

param_getindex(V,1)
param_entry(V,1)
v2 = param_view(V,2)

MM = Matrix{MatrixOfMatrices{Float64,2,Array{Float64,3}}}(undef,2,2)
MM[1,1] = ParamArray([rand(3,3),rand(3,3)])
MM[1,2] = ParamArray([rand(3,4),rand(3,4)])
MM[2,1] = ParamArray([rand(4,3),rand(4,3)])
MM[2,2] = ParamArray([rand(4,4),rand(4,4)])
M = mortar(MM)

M[3,3] = rand(4,4)

param_getindex(M,1)
param_entry(M,1,2)
m2 = param_view(M,2)

M0 = zero(M)
M′ = similar(M,typeof(m2))
copyto!(M′,M)
@check M′ ≈ M

M1 = testitem(M)
V1 = testitem(V)

Mv = M*V
@check param_getindex(Mv,1) ≈ M1*V1

using LinearAlgebra
luM = lu(M)

W1 = ParamArray([rand(3),rand(3)])
W2 = ParamArray([rand(4),rand(4)])
W = mortar([W1,W2])

VW = V + W
@check VW.data ≈ V.data + W.data

i = 3
j = BlockArrays.findblockindex.(ParamDataStructures.inneraxes(V),i)[1]
@boundscheck blockcheckbounds(V,Block(j.I))
@inbounds bl = V.data[j.I...]
@inbounds ParamNumber(bl.data[j.α...,:])
