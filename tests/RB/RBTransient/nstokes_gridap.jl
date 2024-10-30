using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, BiformBlock, BlockTriangularSolver

using Gridap.ODEs

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 2*dt

model_dir = datadir(joinpath("models","model_circle_fine.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

order = 2
degree = 2*order+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

μ = [5.,5.,5.]

const Re = 1000.0
const Re′ = 100.0
a(x,t) = μ[1]/Re′
a(t) = x->a(x,t)

const W = 0.5
inflow(t) = abs(1-cos(2π*t/(5*60*dt))+μ[3]*sin(μ[2]*2π*t/(5*60*dt))/100)
g_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(t),0.0,0.0)
g_in(t) = x->g_in(x,t)
g_0(x,t) = VectorValue(0.0,0.0,0.0)
g_0(t) = x->g_0(x,t)

u0(x) = VectorValue(0.0,0.0,0.0)
p0(x) = 0.0

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

# stiffness(t,(u,p),(v,q),dΩ) = ∫(a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
# mass(t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
# res(t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(t,(u,p),(v,q),dΩ)

djac(t,(uₜ,pₜ),(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ
jac(t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ) + ∫(a(t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ #+ graddiv(du,v,dΩ)
res(t,(u,p),(v,q),dΩ) = c(u,v,dΩ) + ∫(v⋅∂t(u))dΩ + ∫(a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ #+ graddiv(u,v,dΩ)

djac(t,ut,dut,v) = djac(t,ut,dut,v,dΩ)
jac(t,u,du,v) = jac(t,u,du,v,dΩ)
res(t,u,v) = res(t,u,v,dΩ)

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialFESpace(test_u,[g_in,g_in,g_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientFEOperator(res,(jac,djac),trial,test)

nlsolver = GridapSolvers.NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
odesolver = ThetaMethod(nlsolver,dt,θ)
xh0 = interpolate_everywhere([u0,p0],trial(t0))
sol = solve(odesolver,feop,t0,tf,xh0)
dir = datadir("plts_gridap")

createpvd(dir) do pvd
  for (t_n,xh_n) in sol
    # println(t_n)
    file = dir*"/sol$(t_n)"*".vtu"
    uh_n,ph_n = xh_n
    pvd[t_n] = createvtk(Ω,file,cellfields=["u"=>uh_n,"p"=>ph_n])
  end
end

# PETSC
using GridapPETSc
using GridapPETSc: PETSC

assembler = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},trial,test)
feop = TransientFEOperator(res,(jac,djac),trial,test;assembler)

function mykspsetup(ksp)
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

options = "-ksp_error_if_not_converged true -ksp_converged_reason"

GridapPETSc.with(args=split(options)) do
  ls = PETScLinearSolver(mykspsetup)
  nls = GridapSolvers.NewtonSolver(ls;maxiter=20,atol=1e-10,rtol=1.e-12,verbose=true)
  odesolver = ThetaMethod(nls,dt,θ)
  sol = solve(odesolver,feop,t0,tf,xh0)
  for (t_n,xh_n) in sol
  end
end

# STEADY
t = tf
jac′((u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ) + ∫(a(t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
res′((u,p),(v,q),dΩ) = c(u,v,dΩ) + ∫(a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

jac′(u,du,v) = jac′(u,du,v,dΩ)
res′(u,v) = res′(u,v,dΩ)

trial_u′ = TrialFESpace(test_u,[g_in(t),g_in(t),g_0(t)])
trial_p′ = trial_p
test′ = test
trial′ = MultiFieldFESpace([trial_u′,trial_p′];style=BlockMultiFieldStyle())
feop′ = FEOperator(res′,jac′,trial′,test′)

solver_u = LUSolver()
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)
solver_p.log.depth = 4

bblocks  = [NonlinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock()    BiformBlock((p,q) -> ∫(-1.0*p*q)dΩ,test_p,test_p)]
coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-11,rtol=1.e-8,verbose=true)
solver.log.depth = 2

using GridapSolvers.NonlinearSolvers
nlsolver = NewtonSolver(solver;maxiter=20,atol=1e-10,rtol=1.e-12,verbose=true)
xh = solve(nlsolver,feop′)

uh,ph = xh
writevtk(Ω,dir*"/ssol.vtu",cellfields=["u"=>uh,"p"=>ph])
