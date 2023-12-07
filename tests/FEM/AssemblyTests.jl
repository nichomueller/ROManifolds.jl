include("./SingleFieldUtilsTests.jl")

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

using Main.SingleFieldUtilsTests

ntimes = 3
nparams = 2
times = rand(ntimes)
params = realization(feop,nparams)

int_mat = ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ
matdata = collect_cell_matrix(trial0,test,int)
global matdata_ok
for np in 1:nparams
  for nt in 1:ntimes
    int_mat_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(du))dΩ
    matdata_ok = collect_cell_matrix(trial0,test,int_ok)
    check_ptarray(matdata[1][1],matdata_ok[1][1];n = (np-1)*ntimes+nt)
  end
end

@check matdata[2] == matdata_ok[2]
@check matdata[3] == matdata_ok[3]

xh = compute_xh(feop,params,times)
int_vec = ∫(aμt(params,times)*∇(dv)⋅∇(xh))dΩ

vecdata = collect_cell_vector(test,int_vec)
global vecdata_ok
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt])
    int_vec_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(xh_t))dΩ
    vecdata_t = collect_cell_vector(test,int_vec_t)
    check_ptarray(vecdata[1][1],vecdata_t[1][1];n = (np-1)*ntimes+nt)
  end
end
@check vecdata[2] == vecdata_ok[2]

end # module
