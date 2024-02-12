using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
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

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
rbinfo = RBInfo(dir;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=5)

rbsolver = RBSolver(rbinfo,fesolver)

snaps,comp = RB.collect_solutions(rbsolver,feop,uh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)
son = select_snapshots(snaps,6)
ron = get_realization(son)
# xrb = solve(rbsolver,red_op,ron)

θ == 0.0 ? dtθ = dt : dtθ = dt*θ

red_test = get_test(red_op)
red_trial = get_trial(red_op)(ron)
fe_trial = trial(ron)
red_x = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

ode_cache = allocate_cache(red_op,ron)
nl_cache = nothing
mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(red_op,ron,y,ode_cache)
ode_cache = update_cache!(ode_cache,red_op,ron)
A,b = ODETools._matrix_and_vector!(mat_cache,vec_cache,red_op,ron,dtθ,y,ode_cache,z)
afop = AffineOperator(A,b)
solve!(red_x,fesolver.nls,afop)

xproj = compress(red_trial,son)

trial0 = trial(nothing)
pA = ParamArray([assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial0,test) for (μ,t) in ron])
pM = ParamArray([assemble_matrix((u,v)->∫(v*u)dΩ,trial0,test)/dt for (μ,t) in ron])
snapsA = Snapshots(pA,ron)
snapsM = Snapshots(pM,ron)
Arb = RB.compress(red_trial,red_test,snapsA)
Mrb = RB.compress(red_trial,red_test,snapsM;combine=(x,y) -> θ*(x-y))
AMrb = Arb + Mrb

# errA = norm(A[1] - AMrb)

# μon = FEM._get_params(ron)[1]
# g_t(x,t) = μon[1]*exp(-x[1]/μon[2])*abs(sin(t/μon[3]))
# g_t(t) = x->g_t(x,t)
# fs_t = TransientTrialFESpace(test,g_t)
# dfs_t = ∂t(fs_t)
# R(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μon,t)*∇(v)⋅∇(u))dΩ - ∫(f(μon,t)*v)dΩ - ∫(h(μon,t)*v)dΓn
# function get_res(t)
#   fs = fs_t(t)
#   dfs = dfs_t(t)
#   x0 = zeros(3)
#   xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
#   assemble_vector(v->R(t,xh,v),test)
# end
# pR = ParamArray([get_res(t) for (μ,t) in ron])
# snapsR = Snapshots(pR,ron)
# Rrb = RB.compress(red_test,snapsR)

# errb = norm(b[1] + Rrb)

function my_solve_step!(
  uf::AbstractVector,
  solver::ThetaMethod,
  op::AffineODEParamOperator,
  r::TransientParamRealization,
  u0::AbstractVector,
  cache)

  dt = solver.dt
  θ = solver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  FEM.shift_time!(r,dtθ)

  if isnothing(cache)
    ode_cache = allocate_cache(op,r)
    vθ = similar(u0)
    vθ .= 0.0
    l_cache = nothing
    A,b = ODETools._allocate_matrix_and_vector(op,r,u0,ode_cache)
    E,f = ODETools._allocate_matrix_and_vector(op,r,u0,ode_cache)
  else
    ode_cache,vθ,A,b,E,f,l_cache = cache
  end
  trial = get_trial(op.feop)
  test = get_test(op.feop)
  M1 = assemble_matrix((u,v) -> ∫(u*v)dΩ,trial(nothing),test)/dtθ
  M = ParamArray([M1 for _ = 1:length(r)])

  ode_cache = update_cache!(ode_cache,op,r)

  ODETools._matrix_and_vector!(A,b,op,r,dtθ,u0,ode_cache,vθ)
  ODETools._matrix_and_vector!(E,f,op,r,dtθ,vθ,ode_cache,vθ)
  afop = AffineOperator(A,b)

  newmatrix = true
  l_cache = solve!(uf,solver.nls,afop,l_cache,newmatrix)

  uf = uf + u0
  _uf = E \ (M*u0 + f)
  @check uf ≈ _uf "$uf != $_uf at time $(get_times(r))"

  if 0.0 < θ < 1.0
    @. uf = uf*(1.0/θ)-u0*((1-θ)/θ)
  end

  cache = (ode_cache,vθ,A,b,E,f,l_cache)
  FEM.shift_time!(r,dt*(1-θ))
  return (uf,r,cache)
end

sol = solve(fesolver,feop,uh0μ,ron)
odesol = sol.odesol
wf = copy(odesol.u0)
w0 = copy(odesol.u0)
r = FEM.get_at_time(odesol.r,:initial)
cache = nothing

ye = []
sA = []
sb = []
while get_times(r) < FEM.get_final_time(odesol.r) - 100*eps()
  wf,r,cache = my_solve_step!(wf,odesol.solver,odesol.op,r,w0,cache)
  ode_cache,vθ,A,b,E,F,l_cache = cache
  w0 .= wf
  push!(ye,copy(w0).array...)
  push!(sA,copy(E).array...)
  push!(sb,copy(F).array...)
end
snap_ok = stack(ye)

son ≈ snap_ok

snaps
