using Gridap
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)
order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

# weak formulation
a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# solvers
fesolver = ThetaMethod(LUSolver(),dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# RB method
# we can load & solve directly, if the offline structures have been previously saved to file
try
  results = load_solve(rbsolver,feop,test_dir)
catch
  @warn "Loading offline structures failed: running offline phase"
  fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

  save(test_dir,fesnaps)
  save(test_dir,rbop)
  save(test_dir,results)
end

# post process
println(compute_error(results))
println(compute_speedup(results))
average_plot(rbop,results;dir=joinpath(test_dir,"plots"))

# # NEED TO IMPROVE:
using Gridap
using BenchmarkTools

# A = fesnaps
# A′ = flatten_snapshots(A)
# @btime A′*A′
# B = copy(A′)
# @btime B'*B

fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
A = flatten_snapshots(fesnaps)

using Mabla.FEM.IndexMaps
A = ModeTransientSnapshots(Snapshots(fesnaps.data,TrivialIndexMap(collect(1:647)),fesnaps.realization))

using BenchmarkTools
At = A'
@btime A'*A

A′ = copy(A)
@btime A′'*A′

using LinearAlgebra

function LinearAlgebra.:*(A::ModeTransientSnapshots,B::Adjoint{T,<:ModeTransientSnapshots}) where T
  C = adjoint(A)*B.parent
  return adjoint(C)
end

function LinearAlgebra.:*(A::Adjoint{T,<:ModeTransientSnapshots},B::ModeTransientSnapshots) where T
  a = A.parent.snaps.data
  b = B.snaps.data
  ns,nt,np = num_space_dofs(B),num_times(B),num_params(B)
  ntnp = nt*np
  c = zeros(eltype(A),(ntnp,ntnp))

  @inbounds for itA in 1:nt
    row_block = a[itA]
    @inbounds for itB in 1:nt
      col_block = b[itB]
      @inbounds for ipA in 1:np
        @fastmath iA = (itA-1)*np + ipA
        row = row_block[ipA]
        @inbounds for ipB in 1:np
          @fastmath iB = (itB-1)*np + ipB
          col = col_block[ipB]
          c[iA,iB] = dot(row,col)
        end
      end
    end
  end
  return c
end

At*A
@btime $At*$A

@btime begin
  A′ = copy(A)
  A′'*A′
end

nt = 10
np = 60
v = A.snaps.data[1].data[1]
@btime begin
  @inbounds for itA in 1:nt
    @inbounds for itB in 1:nt
      @inbounds for ipA in 1:np
        @inbounds for ipB in 1:np
          dot($v,$v)
        end
      end
    end
  end
end
