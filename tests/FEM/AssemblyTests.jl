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

using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check

ntimes = 3
nparams = 2
times = rand(ntimes)
params = realization(feop,nparams)

int_mat = ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ
matdata = collect_cell_matrix(trial0,test,int_mat)

for np in 1:nparams
  for nt in 1:ntimes
    int_mat_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(du))dΩ
    matdata_t = collect_cell_matrix(trial0,test,int_mat_t)
    check_ptarray(matdata[1][1],matdata_t[1][1];n = (np-1)*ntimes+nt)
    @check matdata[2] == matdata_t[2]
    @check matdata[3] == matdata_t[3]
  end
end

u = ones(num_free_dofs(test))
ptu = PArray([copy(u) for _ = 1:ntimes*nparams])
xh = compute_xh(feop,params,times,(ptu,ptu))
int_vec = ∫(dv*∂ₚt(xh))dΩ + ∫(aμt(params,times)*∇(dv)⋅∇(xh))dΩ
vecdata = collect_cell_vector(test,int_vec)
b = PArray([ones(num_free_dofs(test)) for _ = 1:ntimes*nparams])
assemble_vector_add!(b,feop.assem,vecdata)
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt],(u,u))
    int_vec_t = ∫(dv*∂t(xh_t))dΩ + ∫(a(params[np],times[nt])*∇(dv)⋅∇(xh_t))dΩ
    vecdata_t = collect_cell_vector(test,int_vec_t)
    b_t = ones(num_free_dofs(test))
    assemble_vector_add!(b_t,feop_t.assem_t,vecdata_t)
    check_ptarray(vecdata[1][1],vecdata_t[1][1];n = (np-1)*ntimes+nt)
    @check vecdata[2] == vecdata_t[2]
    @check b[(np-1)*ntimes+nt] ≈ b_t
  end
end

# with residual
u = ones(num_free_dofs(test))
ptu = PArray([copy(u) for _ = 1:ntimes*nparams])
xh = compute_xh(feop,params,times,(ptu,ptu))
int_vec = integrate(feop.res(params,times,xh,dv))
vecdata = collect_cell_vector(test,int_vec)
b = PArray([ones(num_free_dofs(test)) for _ = 1:ntimes*nparams])
assemble_vector_add!(b,feop.assem,vecdata)
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt],(u,u))
    int_vec_t = feop_t.res(times[nt],xh_t,dv)
    vecdata_t = collect_cell_vector(test,int_vec_t)
    b_t = ones(num_free_dofs(test))
    assemble_vector_add!(b_t,feop_t.assem_t,vecdata_t)
    check_ptarray(vecdata[1][1],vecdata_t[1][1];n = (np-1)*ntimes+nt)
    @check vecdata[2] == vecdata_t[2]
    @check b[(np-1)*ntimes+nt] ≈ b_t
  end
end

end # module
