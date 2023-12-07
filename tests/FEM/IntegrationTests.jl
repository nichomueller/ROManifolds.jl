module IntegrationTests

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

using SingleFieldUtilsTests

ntimes = 3
nparams = 2
times = rand(ntimes)
params = realization(feop,nparams)

int_mat = ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ

for np in 1:nparams
  for nt in 1:ntimes
    int_mat_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(du))dΩ
    check_ptarray(int[Ω],int_mat_t[Ω];n = (np-1)*ntimes+nt)
  end
end

xh = compute_xh(feop,params,times)
int_vec = ∫(aμt(params,times)*∇(dv)⋅∇(xh))dΩ

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt])
    int_vec_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(xh_t))dΩ
    check_ptarray(int_vec[Ω],int_vec_t[Ω];n = (np-1)*ntimes+nt)
  end
end

end # module
