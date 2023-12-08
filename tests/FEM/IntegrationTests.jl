include("./SingleFieldUtilsFEMTests.jl")

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

using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check

ntimes = 3
nparams = 2
times = rand(ntimes)
params = realization(feop,nparams)

int_mat = ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ

for np in 1:nparams
  for nt in 1:ntimes
    int_mat_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(du))dΩ
    check_ptarray(int_mat[Ω],int_mat_t[Ω];n = (np-1)*ntimes+nt)
  end
end

u = ones(num_free_dofs(test))
ptu = PTArray([copy(u) for _ = 1:ntimes*nparams])
xh = compute_xh(feop,params,times,(ptu,ptu))
int_vec = ∫(aμt(params,times)*∇(dv)⋅∇(xh))dΩ

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt],(u,u))
    int_vec_t = ∫(a(params[np],times[nt])*∇(dv)⋅∇(xh_t))dΩ
    check_ptarray(int_vec[Ω],int_vec_t[Ω];n = (np-1)*ntimes+nt)
  end
end

_int_vec = integrate(∫ₚ(aμt(params,times)*∇(dv)⋅∇(xh),dΩ))

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt in 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt],(u,u))
    _int_vec_t = integrate(∫ₚ(a(params[np],times[nt])*∇(dv)⋅∇(xh_t),dΩ))
    check_ptarray(_int_vec[Ω],_int_vec_t[Ω];n = (np-1)*ntimes+nt)
  end
end

x = Point(1,2)
for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  μ = params[np]
  for nt in 1:ntimes
    @check feop.trials[1].dirichlet_μt(x,μ,t) ≈ feop_t.trials[1].dirichlet_t(x,t) "(np,nt) = ($np,$nt)"
    @check feop.trials[2].dirichlet_μt(x,μ,t) ≈ feop_t.trials[2].dirichlet_t(x,t) "(np,nt) = ($np,$nt)"
    @check feop.trials[1](μ,t).dirichlet_values ≈ feop_t.trials[1](t).dirichlet_values
    @check feop.trials[2](μ,t).dirichlet_values ≈ feop_t.trials[2](t).dirichlet_values
  end
end

dv = get_fe_basis(test)
ptu0 = PTArray([u0 for _  = 1:ntimes*nparams])
xh = compute_xh(feop,params,times,(ptu0,ptu0))
dc = integrate(feop.res(params,times,xh,dv))
for np = 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  for nt = 1:ntimes
    xh_t = compute_xh_gridap(feop_t,times[nt],(u0,u0))
    dc_t = feop_t.res(times[nt],xh_t,dv)
    check_ptarray(dc[Ω],dc_t[Ω];n = (np-1)*ntimes+nt)
    check_ptarray(dc[Γn],dc_t[Γn];n = (np-1)*ntimes+nt)
  end
end

end # module
