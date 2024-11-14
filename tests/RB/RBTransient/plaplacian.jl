using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapSolvers
import GridapSolvers.NonlinearSolvers: NewtonSolver
using PartitionedArrays
using GridapPETSc
using GridapPETSc: PETSC
using BenchmarkTools
using SparseMatricesCSR

domain = (0,1,0,1)
cells  = (10,10)
model  = CartesianDiscreteModel(domain,cells)

order = 1

g(x,μ,t) = sin(t)*(x[1]*μ[1] + x[2]*μ[2])^order
g(μ,t) = x -> g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

σ(∇u,μ,t) = exp(-sum(μ)*t)*(1.0+∇u⋅∇u)*∇u
σ(μ,t) = ∇u -> σ(∇u,μ,t)
σμt(μ,t) = TransientParamFunction(σ,μ,t)

dσ(∇du,∇u) = exp(-sum(μ)*t)*(2*∇u⋅∇du + (1.0+∇u⋅∇u)*∇du)
dσ(μ,t) = (∇du,∇u) -> dσ(∇du,∇u,μ,t)
dσμt(μ,t) = TransientParamFunction(dσ,μ,t)

f(x,μ,t) = 0
f(μ,t) = x -> f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

order = 1
Ω  = Triangulation(model)
dΩ = Measure(Ω,2order+1)

mass(μ,t,ut,dut,v,dΩ) = ∫(v*dut)dΩ
jac(μ,t,u,du,v,dΩ) = ∫(∇(v)⋅(dσμt(μ,t)∘(∇(du),∇(u))))dΩ
res(μ,t,u,v,dΩ) = ∫(v*∂t(u))dΩ + ∫(∇(v)⋅(σμt(μ,t)∘∇(u)))dΩ - ∫(v*fμt(μ,t))dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_mass = (Ω,)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)

assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},
                              Vector{PetscScalar},U,V)

feop = TransientFEOperator(r,j,U,V,assem)

function mykspsetup′(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
end

options′ = "-ksp_error_if_not_converged true -ksp_converged_reason"

uh′ = GridapPETSc.with(args=split(options′)) do
  ls = PETScLinearSolver(mykspsetup′)
  nls = NewtonSolver(ls;maxiter=20,atol=1e-10,rtol=1.e-12,verbose=true)
  fesolver = FESolver(nls)
  uh = zero(U)
  solve!(uh,fesolver,op)
  return uh
end

################################################################################

using ReducedOrderModels
import ReducedOrderModels.ParamSteady: ParamFEOpFromWeakForm

pspace = ParamSpace(fill([1,10],2))

σσ(∇u) = (1.0 .+ ∇u⋅∇u) * ∇u
dσσ(∇du,∇u) = 2 .* ∇u⋅∇du + (1.0 .+ ∇u⋅∇u) * ∇du

uu((x,y),μ) = (μ[1]*x+μ[2]*y)^k
uu(μ) = x -> uu(x,μ)
uμ(μ) = ParamFunction(uu,μ)

ff(x,μ) = -divergence(y->σ(∇(uu(μ),y)),x)
ff(μ) = x -> ff(x,μ)
fμ(μ) = ParamFunction(ff,μ)

# conv(u,∇u) = (∇u')⋅u
rr(μ,u,v) = ∫( ∇(v)⋅(σσ∘∇(u)) - v*fμ(μ) )dΩ
jj(μ,u,du,v) = ∫( ∇(v)⋅(dσσ∘(∇(du),∇(u))) )dΩ

Uμ = ParamTrialFESpace(V,uμ)
dof_map = FEDofMap(Uμ,V)
assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},
                              Vector{PetscScalar},Uμ,V)
feop = ParamFEOpFromWeakForm{NonlinearParamEq}(rr,jj,pspace,assem,dof_map,Uμ,V)

μ = realization(pspace)

uμh = GridapPETSc.with(args=split(options)) do
  nls = PETScNonlinearSolver(mysnessetup)
  fesolver = FESolver(nls)
  uh = zero(Uμ(μ))
  uh,cache = solve!(uh,fesolver,feop,μ)
  uh
end

uμh′ = GridapPETSc.with(args=split(options′)) do
  ls = PETScLinearSolver(mykspsetup′)
  nls = NewtonSolver(ls;maxiter=20,atol=1e-10,rtol=1.e-12,verbose=true)
  fesolver = FESolver(nls)
  uh = zero(Uμ(μ))
  uh,cache = solve!(uh,fesolver,feop,μ)
  return uh
end

fesolver = NewtonSolver(LUSolver();maxiter=20,atol=1e-10,rtol=1.e-12,verbose=true)
uhμ = zero(Uμ(μ))
solve!(uhμ,fesolver,feop,μ)
