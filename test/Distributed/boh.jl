using Gridap
using GridapROMs
using DrWatson
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using GridapROMs.RBSteady
using PartitionedArrays
using Test

method=:pod
compression=:global
hypred_strategy=:deim
tol=1e-4
nparams=2
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
ncentroids=2

method = method ∈ (:pod,:ttsvd) ? method : :pod
compression = compression ∈ (:global,:local) ? compression : :global
hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

domain = (0,1,0,1)
partition = (8,8)

pdomain = (1,10,1,10)
pspace = ParamSpace(pdomain)

a(μ) = x -> μ[1]*exp(-x[1])
aμ(μ) = parameterise(a,μ)

g(μ) = x -> VectorValue(-μ[2]*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = parameterise(g,μ)

order = 2
degree = 2*order

energy = BlockNorm((H1(),L2()))
coupling = DivCoupling()
state_reduction = SupremizerReduction(coupling,tol,energy;nparams,compression,ncentroids)

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_u = ParamTrialFESpace(test_u,gμ)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldFESpace([test_u,test_p])
  trial = MultiFieldFESpace([trial_u,trial_p])

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  println("diagnostic | fesnaps built ok")
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  println("diagnostic | rbop (with supremizer enrichment) built ok")

  perr = RBSteady.projection_error(rbsolver,rbop,fesnaps)
  println("diagnostic | projection error (basis + project/inv_project, no HR): ", perr)

  # μon = realisation(feop;nparams=10,start=nparams+1)
  # x̂,rbstats = solve(rbsolver,rbop,μon)
  # x,festats = solution_snapshots(rbsolver,feop,μon)
  # perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)
  # println(perf)

  # rbsolverx = RBSteady.set_params(rbsolver;nparams=num_params(x))
  # res = residual_snapshots(rbsolverx,feop,x)
  # jac = jacobian_snapshots(rbsolverx,feop,x)
  # err_res,err_jac = RBSteady.hr_error(rbsolverx,rbop,res,jac,x)
  # println("diagnostic | hr error residual (per trian): ", err_res)
  # println("diagnostic | hr error jacobian (per trian): ", err_jac)

  # # per-rank save / load round-trip of the FE snapshots (distributed)
  # diagdir = mkpath(joinpath(@__DIR__,"boh_diag"))
  # save(diagdir,fesnaps)
  # fesnaps_loaded = load_snapshots(diagdir,ranks)
  # println("diagnostic | snapshots save/load round-trip ok: ",
  #   compute_relative_error(fesnaps,fesnaps_loaded) < 1e-12)
end

with_debug() do distribute
  main(distribute,(2,2))
end