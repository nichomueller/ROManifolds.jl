module PoissonEquation

using Gridap
using ROManifolds

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1,10,1,10,1,10)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> exp(-x[1]/sum(μ))
  aμ(μ) = ParamFunction(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
  gμ(μ) = ParamFunction(g,μ)

  h(μ) = x -> abs(cos(μ[3]*x[2]))
  hμ(μ) = ParamFunction(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    state_reduction = PODReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,3)
    state_reduction = Reduction(tolranks,energy;nparams,unsafe)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
  feop_uniform = LinearParamOperator(res,stiffness,pspace_uniform,trial,test,domains)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    pspace = ParamSpace(pdomain;sampling)
    feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

    fesnaps, = solution_snapshots(rbsolver,feop)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

    println(perf)
  end

end

main(:pod)
main(:ttsvd)

end


using Gridap
using Gridap.Algebra
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs
using ROManifolds
using ROManifolds.ParamDataStructures
using ROManifolds.ParamSteady
using Test
using FillArrays
import Gridap.MultiField: BlockMultiFieldStyle
using GridapSolvers
using GridapSolvers.NonlinearSolvers

method=:pod
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
unsafe=false

pdomain = (1,10,-1,5,1,2)

domain = (0,1,0,1)
partition = (10,10)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(μ) = x -> μ[1]*exp(-x[1])
aμ(μ) = ParamFunction(a,μ)

g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = ParamFunction(g,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

jac_lin(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res_lin(μ,(u,p),(v,q),dΩ) = jac_lin(μ,(u,p),(v,q),dΩ)

res_nlin(μ,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
state_reduction = SupremizerReduction(coupling,1e-4,energy;nparams=50,sketch=:sprn)

fesolver = NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

pspace = ParamSpace(pdomain)
feop_lin = LinearParamOperator(res_lin,jac_lin,pspace,trial,test,domains)
feop_nlin = ParamOperator(res_nlin,jac_nlin,pspace,trial,test,domains)
feop = LinearNonlinearParamOperator(feop_lin,feop_nlin)

fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

using ROManifolds.ParamAlgebra
r = μon
op = rbop
x̂ = zero_free_values(get_trial(op)(r))
nlop = parameterize(op,r)
syscache = allocate_systemcache(nlop,x̂)

using ROManifolds.RBSteady
A,b = syscache.A,syscache.b

fill!(x̂,zero(eltype(x̂)))
ParamAlgebra.update_systemcache!(nlop,x̂)

@unpack A,b = cache
residual!(b,nlop,x̂)

syscache_lin = get_linear_systemcache(nlop)
A_lin = get_matrix(syscache_lin)
b_lin = get_vector(syscache_lin)

op_nlin = get_nonlinear_operator(nlop)
residual!(b,op_nlin,x̂)

uh = EvaluationFunction(op_nlin.paramcache.trial,x̂)
test = get_test(rbop.op_nonlinear.op)
v = get_fe_basis(test)

trian_res = get_domains_res(rbop.op_nonlinear.op)
res = get_res(rbop.op_nonlinear.op)
Dc = res(r,uh,v)
map(trian_res) do strian
  b_strian = b.fecache[strian]
  rhs_strian = rbop.op_nonlinear.rhs[strian]
  vecdata = RBSteady.collect_cell_hr_vector(test,Dc,strian,rhs_strian)
  RBSteady.assemble_hr_vector_add!(b_strian,vecdata...)
end


residual!(b,nlop,x̂)
jacobian!(A,nlop,x̂)
ns = solve!(x̂,fesolver,A,b)

rbopl = rbop.op_linear
slvl = LUSolver()
residual!(b,rbopl,r,x̂,nlop.op_linear.paramcache)
CIAO
# uh = EvaluationFunction(nlop.paramcache.trial,x̂)
# v = get_fe_basis(test)

# trian_res = get_domains_res(rbop.op)
# dc = get_res(rbop.op)(μon,uh,v)

# strian = trian_res[1]
# b_strian = b.fecache[strian]
# rhs_strian = rbop.rhs[strian]
# cellvec,cellidsrows,icells = RBSteady.collect_cell_hr_vector(test,dc,strian,rhs_strian)
# RBSteady.assemble_hr_vector_add!(b_strian,cellvec,cellidsrows,icells)
# # i = 1
# # cellveci = lazy_map(RBSteady.BlockReindex(cellvec,i),icells.array[i])

# bfe = get_free_dof_values(zero(trial(r)))
# fill!(bfe,0.0)
# Ωv = view(Ω,[21,51])
# dΩv = Measure(Ωv,degree)
# dcv = get_res(rbop.op.op)(r,uh,v,dΩv)
# vvecdata = collect_cell_vector_for_trian(test,dcv,Ωv)
# assem = get_param_assembler(feop,r)
# assemble_vector_add!(bfe,assem,vvecdata)

CIAO
# b = syscache.b
# boh = copy(x̂.fe_data)
# fill!(boh,1.0)
# residual!(b,nlop,boh)

# uh = EvaluationFunction(nlop.paramcache.trial,boh)
# v = get_fe_basis(test)
# assem = get_param_assembler(nlop.op.op,r)

# b_strian = b.fecache[strian]
# trian_res = get_domains_res(nlop.op.op)
# strian = trian_res[1]
# rhs_strian = rbop.rhs[strian]
# dc = get_res(nlop.op.op)(r,uh,v)
# vecdata = RBSteady.collect_cell_hr_vector(test,dc,strian,rhs_strian)
# RBSteady.assemble_hr_vector_add!(b_strian,vecdata...)

# cellveci = lazy_map(RBSteady.BlockReindex(vecdata[1],1),vecdata[3].array[1])
# k = RBSteady.BlockReindex(vecdata[1],1)
# fi = map(testitem,(vecdata[3].array[1],))
# T = return_type(k,fi...)
# # lazy_map(k,T,vecdata[3].array[1])
# j_to_i = vecdata[3].array[1]
# i_to_maps = k.values.maps
# i_to_args = k.values.args
# j_to_maps = lazy_map(Reindex(i_to_maps),eltype(i_to_maps),j_to_i)
# j_to_args = map(i_to_fk->lazy_map(RBSteady.BlockReindex(i_to_fk,k.blockid),eltype(i_to_fk),j_to_i),i_to_args)
# LazyArray(T,j_to_maps,j_to_args...)

# b_trian = b.fecache[strian]
# cell_irows = RBSteady.get_cellids_rows(rhs_strian)
# scell_vec = get_contribution(dc,strian)
# cell_vec,trian = move_contributions(scell_vec,strian)
# @assert ndims(eltype(cell_vec)) == 1
# cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
# icells = RBSteady.get_owned_icells(rhs_strian)



# # RBSteady.assemble_hr_vector_add!(b_trian,vecdata...)
# cellvec,_cellidsrows = vecdata
# cellidsrows = FESpaces.map_cell_rows(RBSteady.HRAssemblyStrategy(),_cellidsrows)
# rows_cache = array_cache(cellidsrows)
# vals_cache = array_cache(cellvec)
# vals1 = getindex!(vals_cache,cellvec,1)
# rows1 = getindex!(rows_cache,cellidsrows,1)
# add! = RBSteady.AddHREntriesMap(+)
# add_cache = return_cache(add!,b,vals1,rows1)
# caches = add_cache,vals_cache,rows_cache
# RBSteady._numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows)

# A = syscache.A
# jacobian!(A,nlop,boh)

# du = get_trial_fe_basis(trial)
# trian_jac = get_domains_jac(op.op)
# dc = get_jac(op.op)(r,uh,du,v)
# strian = trian_jac[1]
# A_trian = A.fecache[strian]
# lhs_trian = op.lhs[strian]
# cell_irows = RBSteady.get_cellids_rows(lhs_trian)
# cell_icols = RBSteady.get_cellids_cols(lhs_trian)
# scell_mat = get_contribution(dc,strian)
# cell_mat,trian = move_contributions(scell_mat,strian)
# @assert ndims(eltype(cell_mat)) == 2
# cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
# cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
# matdata = cell_mat_rc,cell_irows,cell_icols
# # assemble_hr_matrix_add!(A_trian,assem,matdata)
# cellmat,_cellidsrows,_cellidscols = matdata
# cellidsrows = FESpaces.map_cell_rows(RBSteady.HRAssemblyStrategy(),_cellidsrows)
# cellidscols = FESpaces.map_cell_rows(RBSteady.HRAssemblyStrategy(),_cellidscols)
