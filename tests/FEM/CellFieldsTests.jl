module CellFieldTests

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

x = get_cell_points(Ω)

cf_mat = aμt(params,times)*∇(dv)⋅∇(du)
cfx_mat = cf_mat(x)

for np in 1:nparams
  for nt in 1:ntimes
    cf_mat_t = a(params[np],times[nt])*∇(dv)⋅∇(du)
    cfx_mat_t = cf_mat_t(x)
    check_ptarray(cfx_mat,cfx_mat_t;n = (np-1)*ntimes+nt)
  end
end

xh = compute_xh(feop,params,times)
cf_vec = aμt(params,times)*∇(dv)⋅∇(xh)
cfx_vec = cf_vec(x)

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt])
    cf_vec_t = a(params[np],times[nt])*∇(dv)⋅∇(xh_t)
    cfx_vec_t = cf_vec_t(x)
    check_ptarray(cfx_vec,cfx_vec_t;n = (np-1)*ntimes+nt)
  end
end

end # module
