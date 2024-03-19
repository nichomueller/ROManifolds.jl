include("./SingleFieldUtilsFEMTests.jl")

module AssemblyTests

using LinearAlgebra

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools

using Mabla
using Mabla.FEM
using Mabla.Distributed

using Main.SingleFieldUtilsFEMTests
using GridapDistributed

import Gridap.Helpers: @check

ntimes = 3
nparams = 2
times = rand(ntimes)
params = realization(feop,nparams)

int_mat = ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ
matdata = collect_cell_matrix(trial0,test,int_mat)

int_mat_t = ∫(a(params[1],times[1])*∇(dv)⋅∇(du))dΩ
matdata_t = collect_cell_matrix(trial0,test,int_mat_t)

for np in 1:nparams
  for nt in 1:ntimes
    int_mat_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(du))dΩ
    matdata_t = collect_cell_matrix(trial0,test,int_mat_t)
    map(local_views(matdata),local_views(matdata_t)) do matdata,matdata_t
      check_ptarray(matdata[1][1],matdata_t[1][1];n = (np-1)*ntimes+nt)
      @check matdata[2] == matdata_t[2]
      @check matdata[3] == matdata_t[3]
    end
  end
end

# with jacobian
u = zero_free_values(trial(params,times))
u .= one(Float64)
b = copy(u)
x = similar(u)
xh = compute_xh(feop,params,times,(u,u))
int_mat = feop.jacs[1](params,times,xh,du,dv)
matdata = collect_cell_matrix(trial0,test,int_mat)
A = allocate_jacobian(feop,params,times,xh,1)
assemble_matrix_add!(A,feop.assem,matdata)
luA = lu(A)
ldiv!(x,luA,b)
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    u_t = zero_free_values(test)
    u_t .= one(Float64)
    b_t = copy(u_t)
    x_t = similar(u_t)
    xh_t = compute_xh_gridap(feop_t,times[nt],(u_t,u_t))
    int_mat_t = feop_t.jacs[1](times[nt],xh_t,du,dv)
    matdata_t = collect_cell_matrix(feop_t.trial[1](nothing),test,int_mat_t)
    A_t = allocate_jacobian(feop_t,times[nt],xh_t.cellfield,nothing)
    assemble_matrix_add!(A_t,feop_t.assem_t,matdata_t)
    luA_t = lu(A_t)
    ldiv!(x_t,luA_t,b_t)
    map(local_views(matdata),
        local_views(matdata_t),
        local_views(A),
        local_views(A_t),
        local_views(luA.lu_in_main),
        local_views(luA_t.lu_in_main),
        local_views(x),
        local_views(x_t)) do matdata,matdata_t,A,A_t,luA,luA_t,x,x_t
      check_ptarray(matdata[1][1],matdata_t[1][1];n = (np-1)*ntimes+nt)
      @check matdata[2] == matdata_t[2]
      @check A[(np-1)*ntimes+nt] ≈ A_t
      if !(isnothing(luA) && isnothing(luA_t))
        @check luA[(np-1)*ntimes+nt] ≈ luA_t
      end
      @check x[(np-1)*ntimes+nt] ≈ x_t "Failed at index $((np-1)*ntimes+nt)"
    end
  end
end

u = zero_free_values(trial(params,times))
u .= one(Float64)
xh = compute_xh(feop,params,times,(u,u))
int_vec = ∫(dv*∂t(xh))dΩ + ∫(aμt(params,times)*∇(dv)⋅∇(xh))dΩ
vecdata = collect_cell_vector(test,int_vec)
b = zero_free_values(trial(params,times))
assemble_vector_add!(b,feop.assem,vecdata)
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    u_t = zero_free_values(test)
    u_t .= one(Float64)
    xh_t = compute_xh_gridap(feop_t,times[nt],(u_t,u_t))
    int_vec_t = ∫(dv*∂t(xh_t))dΩ + ∫(a(params[np],times[nt])*∇(dv)⋅∇(xh_t))dΩ
    vecdata_t = collect_cell_vector(test,int_vec_t)
    b_t = zero_free_values(test)
    assemble_vector_add!(b_t,feop_t.assem_t,vecdata_t)
    map(local_views(vecdata),local_views(vecdata_t),local_views(b),local_views(b_t)
        ) do vecdata,vecdata_t,b,b_t
      check_ptarray(vecdata[1][1],vecdata_t[1][1];n = (np-1)*ntimes+nt)
      @check vecdata[2] == vecdata_t[2]
      @check b[(np-1)*ntimes+nt] ≈ b_t
    end
  end
end

# with residual
u = zero_free_values(trial(params,times))
u .= one(Float64)
xh = compute_xh(feop,params,times,(u,u))
int_vec = feop.res(params,times,xh,dv)
vecdata = collect_cell_vector(test,int_vec)
b = zero_free_values(trial(params,times))
assemble_vector_add!(b,feop.assem,vecdata)
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    u_t = zero_free_values(test)
    u_t .= one(Float64)
    xh_t = compute_xh_gridap(feop_t,times[nt],(u_t,u_t))
    int_vec_t = feop_t.res(times[nt],xh_t,dv)
    vecdata_t = collect_cell_vector(test,int_vec_t)
    b_t = zero_free_values(test)
    assemble_vector_add!(b_t,feop_t.assem_t,vecdata_t)
    map(local_views(vecdata),local_views(vecdata_t),local_views(b),local_views(b_t)
        ) do vecdata,vecdata_t,b,b_t
      check_ptarray(vecdata[1][1],vecdata_t[1][1];n = (np-1)*ntimes+nt)
      @check vecdata[2] == vecdata_t[2]
      @check b[(np-1)*ntimes+nt] ≈ b_t
    end
  end
end

end # module
