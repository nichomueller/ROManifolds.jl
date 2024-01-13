include("../FEM/./SingleFieldUtilsFEMTests.jl")

module SingleFieldUtilsRBTests

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
using Mabla.RB
using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: TransientFEOperatorFromWeakForm
import Gridap.ODEs.TransientFETools: get_algebraic_operator
import Gridap.ODEs.TransientFETools: allocate_cache
import Gridap.ODEs.TransientFETools: update_cache!
import Gridap.ODEs.TransientFETools: get_order

ntimes = 60
tf = ntimes*dt
nparams = 50
ntests = 10

fesolver = ThetaMethod(LUSolver(),θ,dt)
times = get_stencil_times(fesolver)

ϵ = 1e-4
norm_style = :l2
st_mdeim = false
rbinfo = RBInfo(pwd();ϵ,norm_style,nsnaps_state=nparams,nsnaps_mdeim=nparams,
  nsnaps_test=ntests,st_mdeim)

sols,params, = collect_solutions(rbinfo,fesolver,feop)
rbspace = reduced_basis(rbinfo,feop,sols)

nzm = NnzMatrix(sols[1:nparams];nparams)
space_values = recast(nzm)
time_values = change_mode(nzm)

if norm_style == :l2
  nnorm = norm(space_values - rbspace.basis_space*rbspace.basis_space'*space_values)
  dnorm = norm(space_values)
  @check nnorm / dnorm <= ϵ*10
else
  norm_matrix = get_norm_matrix(rbinfo,feop)
  nnorm = norm(space_values - rbspace.basis_space*rbspace.basis_space'*norm_matrix*space_values)
  dnorm = norm(space_values,norm_matrix)
  @check nnorm / dnorm <= ϵ*10
end

nnorm = norm(time_values - rbspace.basis_time*rbspace.basis_time'*time_values)
dnorm = norm(time_values)
@check nnorm / dnorm <= ϵ*10

# offline
op = get_algebraic_operator(fesolver,feop,rbspace,params[1:nparams])
ress,trians = collect_residuals_for_trian(op)
jacs1,trians1 = collect_jacobians_for_trian(op;i=1)
jacs2,trians2 = collect_jacobians_for_trian(op;i=2)

u0 = zeros(num_free_dofs(test))
for (itrian,trian) in enumerate(trians)
  resi = ress[itrian]
  for np in 1:nparams
    feop_t = get_feoperator_gridap(feop,params[np])
    for nt in 1:ntimes
      resi_t = compute_res_gridap(feop_t,times[nt],(u0,u0),trian)
      _,resi_t_nnz = compress_array(resi_t)
      @check resi[:,(np-1)*ntimes+nt] ≈ resi_t_nnz "Failed for (np,nt) = ($np,$nt)"
    end
  end
end

for (i,trian) in enumerate(trians1)
  jac1i = jacs1[i]
  for np in 1:nparams
    feop_t = get_feoperator_gridap(feop,params[np])
    for nt in 1:ntimes
      jac1i_t = compute_jac_gridap(feop_t,times[nt],(u0,u0),trian,1)
      _,jac1i_t_nnz = compress_array(jac1i_t)
      @check jac1i[:,(np-1)*ntimes+nt] ≈ jac1i_t_nnz
    end
  end
end

function check_mdeim_spatial_basis(xfull_off,xfull_on,xreduced)
  basis_space = tpod(xfull_off)
  space_idx = get_interpolation_idx(basis_space)
  @check xreduced ≈ xfull_on[space_idx,:]
  @check norm(xfull_on - basis_space*basis_space'*xfull_on) / norm(xfull_on) ≤ ϵ*10
end

function check_mdeim_coefficient(coeff_cache,affdec,xfull_off,xfull_on,xreduced)
  basis_space = tpod(xfull_off)
  coeff_mdeim = rb_coefficient!(coeff_cache,affdec,xreduced;st_mdeim)
  for np in 1:ntests
    coeff = transpose(basis_space'*xfull_on[:,(np-1)*ntimes+1:np*ntimes])
    err_coeff = norm(coeff_mdeim[np] - coeff,Inf)
    println("Error coefficient is $err_coeff")
  end
end

function check_rb_contribution(k::RBVecContributionMap,cache,affdec,xfull_on,xreduced)
  nfree = length(get_free_dof_ids(test))
  coeff_cache,rb_cache = cache
  coeff_mdeim = rb_coefficient!(coeff_cache,affdec,xreduced;st_mdeim)
  basis_space_proj = affdec.basis_space
  basis_time = last(affdec.basis_time)
  for np in 1:ntests
    contrib_mdeim = evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff_mdeim[np])
    nzmidx = NnzMatrix(
      xfull_on.affinity,
      xfull_on[:,(np-1)*ntimes+1 : np*ntimes],
      xfull_on.nonzero_idx,
      nfree,
      1)
    contrib = space_time_projection(nzmidx,rbspace)
    err_contrib = norm(contrib-contrib_mdeim,Inf)
    println("Error contribution is $err_contrib")
  end
end

function check_rb_contribution(k::RBMatContributionMap,cache,affdec,xfull,xreduced;i=1)
  combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
  nfree = length(get_free_dof_ids(test))
  coeff_cache,rb_cache = cache
  coeff_mdeim = rb_coefficient!(coeff_cache,affdec,xreduced;st_mdeim)
  basis_space_proj = affdec.basis_space
  basis_time = last(affdec.basis_time)
  for np in 1:ntests
    contrib_mdeim = evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff_mdeim[np])
    nzmidx = NnzMatrix(
      xfull.affinity,
      xfull[:,(np-1)*ntimes+1 : np*ntimes],
      xfull.nonzero_idx,
      nfree,
      1)
    contrib = space_time_projection(nzmidx,rbspace,rbspace;combine_projections)
    err_contrib = norm(contrib-contrib_mdeim,Inf)
    println("Error contribution is $err_contrib")
    #@check err_contrib <= ϵ*10
  end
end

# online
op_online = get_algebraic_operator(fesolver,feop,rbspace,params[nparams+1:end])
ress_online,trians = collect_residuals_for_trian(op_online)
jacs1_online,trians1 = collect_jacobians_for_trian(op_online;i=1)
jacs2_online,trians2 = collect_jacobians_for_trian(op_online;i=2)

rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params[1:nparams])

(rhs_cache,lhs_cache), = allocate_cache(op_online,rbspace)
rhs_mdeim_cache,rhs_rb_cache = rhs_cache
rhs_collect_cache,rhs_coeff_cache = rhs_mdeim_cache
lhs_mdeim_cache,lhs_rb_cache = lhs_cache
lhs_collect_cache,lhs_coeff_cache = lhs_mdeim_cache

red_res = collect_reduced_residuals!(rhs_collect_cache,op_online,rbrhs)
red_jac1 = collect_reduced_jacobians!(lhs_collect_cache,op_online,rblhs[1];i=1)
red_jac2 = collect_reduced_jacobians!(lhs_collect_cache,op_online,rblhs[2];i=2)

for (itrian,trian) in enumerate(trians)
  resi_of = ress[itrian]
  resi_on = ress_online[itrian]
  red_resi = red_res[itrian]
  affdeci = rbrhs.affine_decompositions[itrian]
  check_mdeim_spatial_basis(resi_of,resi_on,red_resi)
  check_mdeim_coefficient(rhs_coeff_cache,affdeci,resi_of,resi_on,red_resi)
  check_rb_contribution(
    RBVecContributionMap(),
    (rhs_coeff_cache,rhs_rb_cache),
    affdeci,
    resi_on,
    red_resi)
end

for (itrian,trian) in enumerate(jacs1)
  jac1i_of = jacs1[itrian]
  jac1i_on = jacs1_online[itrian]
  red_jac1i = red_jac1[itrian]
  affdeci = rblhs[1].affine_decompositions[itrian]
  check_mdeim_spatial_basis(jac1i_of,jac1i_on,red_jac1i)
  check_mdeim_coefficient(lhs_coeff_cache,affdeci,jac1i_of,jac1i_on,red_jac1i)
  check_rb_contribution(
    RBMatContributionMap(),
    (lhs_coeff_cache,lhs_rb_cache),
    affdeci,
    jac1i_on,
    red_jac1i)
end

end # module
