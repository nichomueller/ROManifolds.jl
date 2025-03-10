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
using ROManifolds.ParamODEs
using ROManifolds.Utils
using Test
using FillArrays
import Gridap.MultiField: BlockMultiFieldStyle
using GridapSolvers
using GridapSolvers.NonlinearSolvers

method=:ttsvd
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
unsafe=false

pdomain = (1,10,1,10,1,10)

domain = (0,1,0,1)
partition = (10,10)
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

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

state_reduction = Reduction(fill(1e-4,3),energy;nparams=50)

# coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
# state_reduction = SupremizerReduction(coupling,1e-4,energy;nparams=50,sketch=:sprn)
# ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
# state_reduction = SupremizerReduction(ttcoupling,fill(1e-4,4),energy;nparams=50)
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf

fesolver = ThetaMethod(LUSolver(),dt,θ)

rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

ptspace = TransientParamSpace(pdomain,tdomain)
feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

using ROManifolds.DofMaps
using ROManifolds.ParamAlgebra
using ROManifolds.RBSteady
using ROManifolds.RBTransient


op,r = rbop,μon
trial = get_trial(op)(r)
x̂ = zero_free_values(trial)
u = x̂

nlop = parameterize(op,r)
syscache = allocate_systemcache(nlop,x̂)
paramcache = nlop.paramcache
shift!(r,dt*(θ-1))
update_paramcache!(paramcache,nlop.op,r)
shift!(r,dt*(1-θ))

np = num_params(r)
hr_time_ids = RBTransient.get_common_time_domain(op.rhs)
hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
hr_uh = RBTransient._make_hr_uh_from_us(op.op,u,paramcache.trial,hr_param_time_ids)

v = get_fe_basis(test)
trian_res = get_domains_res(op.op)
μ = get_params(r)
hr_t = view(get_times(r),hr_time_ids)
dc = get_res(op.op)(μ,hr_t,hr_uh,v)

strian = trian_res[2]
rhs_strian = op.rhs[strian]
vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian,hr_param_time_ids)
cellvec,cellidsrows,icells,locations = vecdata
style = RBTransient.TransientHRStyle(rhs_strian)

b = syscache.b
b_strian = b.fecache[strian]
assemble_hr_vector_add!(b_strian,style,vecdata...)
# add! = RBTransient.AddTransientHREntriesMap(style,locations)

ress = residual_snapshots(rbsolver,feop,fesnaps)
basis = projection(rbsolver.residual_reduction.reduction,ress[2])
(rows,indices_time),interp = empirical_interpolation(basis)

# #
# cell_dof_ids = get_cell_dof_ids(test)
# dofs = [15,101,15,64]
# times = [1,10,8,3]

# using ROManifolds.RBSteady
# using ROManifolds.RBTransient

# cells = RBSteady.reduced_cells(test,Ω.trian,dofs)

# cache = array_cache(cell_dof_ids)

# ncells = length(cells)
# ptrs = Vector{Int32}(undef,ncells+1)
# @inbounds for (icell,cell) in enumerate(cells)
#   celldofs = getindex!(cache,cell_dof_ids,cell)
#   ptrs[icell+1] = length(celldofs)
# end
# length_to_ptrs!(ptrs)

# # count number of occurrences
# iudof_to_idof = RBTransient.get_iudof_to_idof(dofs,times)
# ucache = array_cache(iudof_to_idof)
# N = RBTransient.get_max_offset(iudof_to_idof)

# using ROManifolds.DofMaps
# _correct_idof(is,li) = li
# _correct_idof(is::OIdsToIds,li) = is.terms[li]

# using Gridap.TensorValues

# #z = zero(Mutable(VectorValue{N,Int32}))#zeros(Int32,N)
# data = map(_ -> copy(zeros(Int32,N)),1:ptrs[end]-1) #fill(z,ptrs[end]-1)
# for (icell,cell) in enumerate(cells)
#   celldofs = getindex!(cache,cell_dof_ids,cell)
#   for iudof in eachindex(iudof_to_idof)
#     idofs = getindex!(ucache,iudof_to_idof,iudof)
#     for (iuidof,idof) in enumerate(idofs)
#       dof = dofs[idof]
#       for (_icelldof,celldof) in enumerate(celldofs)
#         if dof == celldof
#           icelldof = _correct_idof(celldofs,_icelldof)
#           vv = data[ptrs[icell]-1+icelldof]
#           vv[iuidof] = idof
#         end
#       end
#     end
#   end
# end
# ye = Table(map(x->VectorValue(x),data),ptrs)

# vv = zeros(Int32,N)
# icell,cell = 5,cells[5]
# celldofs = getindex!(cache,cell_dof_ids,cell)
# iudof = 4
# idofs = getindex!(ucache,iudof_to_idof,iudof)
# for (iuidof,idof) in enumerate(idofs)
#   dof = dofs[idof]
#   for (_icelldof,celldof) in enumerate(celldofs)
#     if dof == celldof
#       icelldof = _correct_idof(celldofs,_icelldof)
#       println(ptrs[icell]-1+icelldof)
#       vv = data[ptrs[icell]-1+icelldof]
#       vv[iuidof] = idof
#     end
#   end
# end
