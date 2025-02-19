module NavierStokesEquation

using Gridap
using Gridap.MultiField
using GridapSolvers
using GridapSolvers.NonlinearSolvers

using ROM

import Gridap.FESpaces: NonlinearFESolver

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1,10,-1,5,1,2)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  Re = 100
  a(μ) = x -> μ[1]/Re*exp(-x[1])
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
  trian_jac = (Ω,)
  domains_lin = FEDomains(trian_res,trian_jac)
  domains_nlin = FEDomains(trian_res,trian_jac)

  energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  trial_u = ParamTrialFESpace(test_u,gμ)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
    state_reduction = SupremizerReduction(coupling,tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,4)
    ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
    state_reduction = SupremizerReduction(ttcoupling,tolranks,energy;nparams)
  end

  fesolver = NonlinearFESolver(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true))
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
  feop_lin_uniform = LinearParamFEOperator(res_lin,jac_lin,pspace_uniform,trial,test,domains_lin)
  feop_nlin_uniform = ParamFEOperator(res_nlin,jac_nlin,pspace_uniform,trial,test,domains_nlin)
  feop_uniform = LinearNonlinearParamFEOperator(feop_lin_uniform,feop_nlin_uniform)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    pspace = ParamSpace(pdomain;sampling)
    feop_lin = LinearParamFEOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
    feop_nlin = ParamFEOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
    feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)

    fesnaps, = solution_snapshots(rbsolver,feop)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

    println(perf)
  end

end

# main(:pod)
main(:ttsvd)

end

using Gridap
using Gridap.MultiField
using GridapSolvers
using GridapSolvers.NonlinearSolvers

import Gridap.FESpaces: NonlinearFESolver

using ROM
using ROM.DofMaps

method=:ttsvd
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
pdomain = (1,10,-1,5,1,2)

domain = (0,1,0,1)
partition = (20,20)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Re = 100
a(μ) = x -> μ[1]/Re*exp(-x[1])
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
trian_jac = (Ω,)
domains_lin = FEDomains(trian_res,trian_jac)
domains_nlin = FEDomains(trian_res,trian_jac)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:C0,constraint=:zeromean)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

tolrank = tol_or_rank(tol,rank)
if method == :pod
  coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
  state_reduction = SupremizerReduction(coupling,tolrank,energy;nparams,sketch)
else method == :ttsvd
  tolranks = fill(tolrank,4)
  ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
  state_reduction = SupremizerReduction(ttcoupling,tolranks,energy;nparams)
end

fesolver = NonlinearFESolver(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true))
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
feop_lin_uniform = LinearParamFEOperator(res_lin,jac_lin,pspace_uniform,trial,test,domains_lin)
feop_nlin_uniform = ParamFEOperator(res_nlin,jac_nlin,pspace_uniform,trial,test,domains_nlin)
feop_uniform = LinearNonlinearParamFEOperator(feop_lin_uniform,feop_nlin_uniform)
μon = realization(feop_uniform;nparams=10)
x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

# sampling = :halton
# pspace = ParamSpace(pdomain;sampling)
# feop_lin = LinearParamFEOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
# feop_nlin = ParamFEOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
# feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)

# fesnaps, = solution_snapshots(rbsolver,feop)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# x̂,rbstats = solve(rbsolver,rbop,μon)
# perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

_model = CartesianDiscreteModel(domain,partition)
_Ω = Triangulation(_model)
_dΩ = Measure(_Ω,degree)
_trian_res = (_Ω,)
_trian_jac = (_Ω,)
_domains_lin = FEDomains(_trian_res,_trian_jac)
_domains_nlin = FEDomains(_trian_res,_trian_jac)
_test_u = TestFESpace(_Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
_trial_u = ParamTrialFESpace(_test_u,gμ)
_test_p = TestFESpace(_Ω,reffe_p;conformity=:C0,constraint=:zeromean)
_trial_p = ParamTrialFESpace(_test_p)
_test = MultiFieldParamFESpace([_test_u,_test_p];style=BlockMultiFieldStyle())
_trial = MultiFieldParamFESpace([_trial_u,_trial_p];style=BlockMultiFieldStyle())
_feop_lin_uniform = LinearParamFEOperator(res_lin,jac_lin,pspace_uniform,_trial,_test,_domains_lin)
_feop_nlin_uniform = ParamFEOperator(res_nlin,jac_nlin,pspace_uniform,_trial,_test,_domains_nlin)
_feop_uniform = LinearNonlinearParamFEOperator(_feop_lin_uniform,_feop_nlin_uniform)
_x,_festats = solution_snapshots(rbsolver,_feop_uniform,μon)

using LinearAlgebra
using Gridap.Arrays
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs
using ROM.ParamSteady
using ROM.ParamFESpaces
using ROM.ParamDataStructures

op = get_algebraic_operator(set_domains(feop_uniform))
nlop = ParamNonlinearOperator(op,μon)
x = get_free_dof_values(zero(trial(μon)))
# fill!(x,1.0)
# copyto!(x.data[1].data,rand(size(x.data[1].data)...))

A = param_getindex(jacobian(nlop,x),1)
b = param_getindex(residual(nlop,x),1)

_op = get_algebraic_operator(set_domains(_feop_uniform))
_nlop = ParamNonlinearOperator(_op,μon)
_A = param_getindex(jacobian(_nlop,x),1)
_b = param_getindex(residual(_nlop,x),1)

norm(A) ≈ norm(_A)
norm(b) ≈ norm(_b)

######################################################
b = allocate_residual(nlop,x)
uh = EvaluationFunction(trial(μon),x)
v = get_fe_basis(test)
# assem = get_param_assembler(op.op,μ)
dcΩ = res_nlin(μon,uh,v,dΩ.measure)[Ω.trian]

_b = allocate_residual(_nlop,x)
_uh = EvaluationFunction(_trial(μon),x)
_v = get_fe_basis(_test)
_dcΩ = res_nlin(μon,_uh,_v,_dΩ)[_Ω]

function _cmp(a,b)
  a[1][1] ≈ b[1][1]
end

sum(lazy_map(_cmp,dcΩ,_dcΩ))
# findall(iszero,lazy_map(_cmp,dcΩ,_dcΩ))

fill!(b,0.0)
vecdata = collect_cell_vector(test,res_nlin(μon,uh,v,dΩ.measure))
assem = get_param_assembler(nlop.op.op.op_nonlinear,μon)
assemble_vector_add!(b,assem,vecdata)
norm(param_getindex(b,1))

fill!(_b,0.0)
_vecdata = collect_cell_vector(_test,res_nlin(μon,_uh,_v,_dΩ))
_assem = get_param_assembler(_nlop.op.op.op_nonlinear,μon)
assemble_vector_add!(_b,_assem,_vecdata)
norm(param_getindex(_b,1))

using Gridap.Fields
using BlockArrays
b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
b2 = MultiField.expand_blocks(assem,b1)
# assemble_vector_add!(b2,assem,vecdata)
# numeric_loop_vector!(b2,assem,vecdata)
cellvec,cellids = vecdata[1][1],vecdata[2][1]
rows_cache = array_cache(cellids)
vals_cache = array_cache(cellvec)
vals1 = getindex!(vals_cache,cellvec,1)
rows1 = getindex!(rows_cache,cellids,1)
add! = AddEntriesMap(+)
add_cache = return_cache(add!,b2,vals1,rows1)
for cell = 1:101
  rows = getindex!(rows_cache,cellids,cell)
  vals = getindex!(vals_cache,cellvec,cell)
  evaluate!(add_cache,add!,b2,vals,rows)

  _rows = getindex!(_rows_cache,_cellids,cell)
  _vals = getindex!(_vals_cache,_cellvec,cell)
  evaluate!(_add_cache,add!,_b2,_vals,_rows)

  @assert norm(b2.array[1].data) ≈ norm(_b2.array[1].data)
end
cell = 2
rows = getindex!(rows_cache,cellids,cell)
vals = getindex!(vals_cache,cellvec,cell)
# evaluate!(add_cache,add!,b2,vals,rows)
# evaluate!(add_cache[1],add!,b2.array[1],vals.array[1],rows.array[1])
# add_entries!(add_cache[1],+,b2.array[1],vals.array[1],rows.array[1])
add_ordered_entries!(add_cache[1],+,b2.array[1],vals.array[1],rows.array[1])

_b1 = ArrayBlock(blocks(_b),fill(true,blocksize(_b)))
_b2 = MultiField.expand_blocks(_assem,_b1)
_cellvec,_cellids = _vecdata[1][1],_vecdata[2][1]
_rows_cache = array_cache(_cellids)
_vals_cache = array_cache(_cellvec)
_vals1 = getindex!(_vals_cache,_cellvec,1)
_rows1 = getindex!(_rows_cache,_cellids,1)
_add_cache = return_cache(add!,_b,_vals1,_rows1)
_rows = getindex!(_rows_cache,_cellids,cell)
_vals = getindex!(_vals_cache,_cellvec,cell)
# evaluate!(_add_cache,add!,_b,_vals,_rows)
add_entries!(_add_cache[1],+,_b2.array[1],_vals.array[1],_rows.array[1])
######################################################

uD = trial[1](μon).dirichlet_values[1]
_uD = _trial[1](μon).dirichlet_values[1]
uD ≈ _uD

z = A \ b
_z = _A \ _b

norm(z) ≈ norm(_z)

fill!(x,0.0)
b = residual(nlop,x)
A = jacobian(nlop,x)
A_item = testitem(A)
x_item = testitem(x)
dx = allocate_in_domain(A_item)
fill!(dx,zero(eltype(dx)))
ss = symbolic_setup(fesolver.nls.ls,A_item)
ns = numerical_setup(ss,A_item,x_item)

rmul!(b,-1)
xi = param_getindex(x,1)
Ai = param_getindex(A,1)
bi = param_getindex(b,1)
numerical_setup!(ns,Ai)
solve!(dx,ns,bi)
xi .+= dx

residual!(b,nlop,x)

_x = copy(x)
fill!(_x,0.0)
_b = residual(_nlop,_x)
_A = jacobian(_nlop,_x)
_A_item = testitem(_A)
_x_item = testitem(_x)
_dx = allocate_in_domain(_A_item)
fill!(_dx,zero(eltype(_dx)))
_ss = symbolic_setup(fesolver.nls.ls,_A_item)
_ns = numerical_setup(_ss,_A_item,_x_item)

rmul!(_b,-1)
_xi = param_getindex(_x,1)
_Ai = param_getindex(_A,1)
_bi = param_getindex(_b,1)
numerical_setup!(_ns,_Ai)
solve!(_dx,_ns,_bi)
_xi .+= _dx

residual!(_b,_nlop,_x)

mytest()

function mytest()
  @assert norm(param_getindex(x,1)) ≈ norm(param_getindex(_x,1))
  @assert norm(param_getindex(A,1)) ≈ norm(param_getindex(_A,1))
  @assert norm(param_getindex(b,1)) ≈ norm(param_getindex(_b,1))
end

# fill!(x,0.0)
# U = trial(μon)
# uh = EvaluationFunction(U,x)

# _x = copy(x)
# _U = _trial(μon)
# _uh = EvaluationFunction(_U,_x)

# uh[1].cell_dof_values[1]
# _uh[1].cell_dof_values[1]

# # EvaluationFunction(U,x)
# fv = restrict_to_field(U,x,1)
# # EvaluationFunction(U.spaces[1],fv)
# # FEFunction(U.spaces[1],fv)
# dv = get_dirichlet_dof_values(U.spaces[1])
# # cv = scatter_free_and_dirichlet_values(U.spaces[1],fv,dv)
# fe = FESpaces.get_fe_space(U.spaces[1])
# # scatter_free_and_dirichlet_values(fe,fv,dv)
# # scatter_free_and_dirichlet_values(fe.space,fv,dv)
# ofe = fe.space
# cell_dof_ids = get_cell_dof_ids(ofe)
# cell_values = lazy_map(Broadcasting(PosNegReindex(fv,dv)),cell_dof_ids)
# mycv = DofMaps.cell_ovalue_to_value(ofe,cell_values)

# _fv = restrict_to_field(_U,_x,1)
# _dv = get_dirichlet_dof_values(_U.spaces[1])
# _cv = scatter_free_and_dirichlet_values(_U.spaces[1],_fv,_dv)

cache = nlop.paramcache
A_lin = cache.A
b_lin = cache.b
paramcache = cache.paramcache
residual!(b,nlop.op.op_nonlinear,μon,x,paramcache)
mul!(b,A_lin,x,1,1)
axpy!(1,b_lin,b)

_cache = _nlop.paramcache
_A_lin = _cache.A
_b_lin = _cache.b
_paramcache = _cache.paramcache
residual!(_b,_nlop.op.op_nonlinear,μon,_x,_paramcache)
mul!(_b,_A_lin,_x,1,1)
axpy!(1,_b_lin,_b)

norm(param_getindex(b_lin,1)) ≈ norm(param_getindex(_b_lin,1))
