using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 20
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,6])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]*exp(sin(π*t/tf)*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = 1-cos(π*t/tf)+sin(π*t/(μ[2]*tf))/μ[3]
g_in(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)
g_c(x,μ,t) = VectorValue(0.0,0.0)
g_c(μ,t) = x->g_c(x,μ,t)
gμt_c(μ,t) = TransientParamFunction(g_c,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

α = 1.e6
Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=degree)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = (∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ)*(1/dt)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

solver_u = LUSolver()
solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
fesolver = ThetaMethod(solver,dt,θ)

ϵ = 1e-5
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)

#=  =#using Gridap.ODEs

r = realization(feop)
r0 = ParamDataStructures.get_at_time(r,:initial)
r = r0
odeop = get_algebraic_operator(feop.op.op)
solver = fesolver
us0 = (get_free_dof_values(xh0μ(r0.params)),)
odecache = allocate_odecache(solver,odeop,r0,us0)
state0,cache = ode_start(solver,odeop,r0,us0,odecache)
statef = copy.(state0)

odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

sysslvr = solver.sysslvr
x = statef[1]
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r,dtθ)
usx = (state0[1],x)
ws = (dtθ,1)
update_odeopcache!(odeopcache,odeop,r)
stageop = LinearParamStageOperator(odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

# sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
A = stageop.A
ss = symbolic_setup(sysslvr,A)
ns = numerical_setup(ss,A)

ns1 = param_getindex(ns,1)

typeof(ns.A)
typeof(ns1.A)

typeof(ns.Pl_ns)
typeof(ns1.Pl_ns)

typeof(ns.Pr_ns)
typeof(ns1.Pr_ns)

typeof(ns.caches)
typeof(ns1.caches)

typeof(ns.solver)
typeof(ns1.solver)

ms = ns.Pr_ns
ms1 = ns1.Pr_ns

typeof(ms.block_caches)
typeof(ms1.block_caches)

typeof(ms.block_ns)
typeof(ms1.block_ns)

typeof(ms.solver)
typeof(ms1.solver)

typeof(ms.work_caches)
typeof(ms1.work_caches)

ks = ms.block_ns[2]
ks1 = ms1.block_ns[2]

ks.A
ks1.A

ks.Pl_ns
ks1.Pl_ns

ks.caches
ks1.caches

ks.solver
ks1.solver

function Gridap.Algebra.solve!(x::AbstractVector,ns::LinearSolvers.FGMRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, Pr, caches = ns.solver, ns.A, ns.Pl_ns, ns.Pr_ns, ns.caches
  V, Z, zl, H, g, c, s = caches
  m   = LinearSolvers.krylov_cache_length(ns)
  log = solver.log

  fill!(V[1],zero(eltype(V[1])))
  fill!(zl,zero(eltype(zl)))

  # Initial residual
  # println(norm(A))
  LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  β    = norm(V[1])
  println(β)
  # done = LinearSolvers.init!(log,β)
  done =false
  while !done
    # Arnoldi process
    j = 1
    V[1] ./= β
    fill!(H,0.0)
    fill!(g,0.0); g[1] = β
    while !done && !LinearSolvers.restart(solver,j)
      # Expand Krylov basis if needed
      if j > m
        H, g, c, s = LinearSolvers.expand_krylov_caches!(ns)
        m = LinearSolvers.krylov_cache_length(ns)
      end

      # Arnoldi orthogonalization by Modified Gram-Schmidt
      fill!(V[j+1],zero(eltype(V[j+1])))
      fill!(Z[j],zero(eltype(Z[j])))
      LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)
      for i in 1:j
        H[i,j] = dot(V[j+1],V[i])
        V[j+1] .= V[j+1] .- H[i,j] .* V[i]
      end
      H[j+1,j] = norm(V[j+1])
      V[j+1] ./= H[j+1,j]
      println(H[j+1,j])
      # Update QR
      for i in 1:j-1
        γ = c[i]*H[i,j] + s[i]*H[i+1,j]
        H[i+1,j] = -s[i]*H[i,j] + c[i]*H[i+1,j]
        H[i,j] = γ
        println(γ)
      end

      # New Givens rotation, update QR and residual
      c[j], s[j], _ = LinearAlgebra.givensAlgorithm(H[j,j],H[j+1,j])
      H[j,j] = c[j]*H[j,j] + s[j]*H[j+1,j]; H[j+1,j] = 0.0
      g[j+1] = -s[j]*g[j]; g[j] = c[j]*g[j]

      β  = abs(g[j+1])
      println(β)
      j += 1
      done = LinearSolvers.update!(log,β)
    end
    j = j-1

    # Solve least squares problem Hy = g by backward substitution
    for i in j:-1:1
      g[i] = (g[i] - dot(H[i,i+1:j],g[i+1:j])) / H[i,i]
    end

    # Update solution & residual
    for i in 1:j
      x .+= g[i] .* Z[i]
    end
    LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  end

  # LinearSolvers.finalize!(log,β)
  return x
end

# GRIDAP
μ = get_params(r0).params[1]
t = 0.005

_g_in(x) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
_g_w(x) = VectorValue(0.0,0.0)
_lhs((u,p),(v,q)) = ∫(a(μ,t0)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
_rhs((v,q)) = ∫(_g_w⋅v)dΩ

_trial_u = TrialFESpace(test_u,[_g_in,_g_w])
_test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
_trial = MultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = AffineFEOperator(_lhs,_rhs,_trial,_test)
_solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
_uh = solve(_solver,_feop)

_uh = zero(_trial)
_x = get_free_dof_values(_uh)
_op = get_algebraic_operator(_feop)
# solve!(_x,_solver,_op,nothing)
# _A = _op.matrix
# _b = _op.vector
# _ss = symbolic_setup(_solver,_A)
# _ns = numerical_setup(_ss,_A)
# solve!(_x,_ns,_b)
# OR
_x = get_free_dof_values(_uh)
_A = param_getindex(stageop.A,1)
_b = param_getindex(stageop.b,1)
_ss = symbolic_setup(_solver,_A)
_ns = numerical_setup(_ss,_A)

solve!(_x,_ns,_b)
_solver, _A, _Pl, _Pr, _caches = _ns.solver, _ns.A, _ns.Pl_ns, _ns.Pr_ns, _ns.caches
_V, _Z, _zl, _H, _g, _c, _s = _caches
_m   = GridapSolvers.LinearSolvers.krylov_cache_length(_ns)

fill!(_V[1],zero(eltype(_V[1])))
fill!(_zl,zero(eltype(_zl)))

# Initial residual
LinearSolvers.krylov_residual!(_V[1],_x,_A,_b,_Pl,_zl)
_β    = norm(_V[1])

# Arnoldi process
j = 1
_V[1] ./= _β
fill!(_H,0.0)
fill!(_g,0.0); _g[1] = _β

if j > m
  _H, _g, _c, _s = LinearSolvers.expand_krylov_caches!(_ns)
  _m = LinearSolvers.krylov_cache_length(_ns)
end

# Arnoldi orthogonalization by Modified Gram-Schmidt
fill!(_V[j+1],zero(eltype(_V[j+1])))
fill!(_Z[j],zero(eltype(_Z[j])))
LinearSolvers.krylov_mul!(_V[j+1],_A,_V[j],_Pr,_Pl,_Z[j],_zl)
for i in 1:j
  _H[i,j] = dot(_V[j+1],_V[i])
  # println(norm(_V[j+1] .- _H[i,j] .* _V[i]))
  _V[j+1] .= _V[j+1] .- _H[i,j] .* _V[i]
end
_H[j+1,j] = norm(_V[j+1])
_V[j+1] ./= _H[j+1,j]

# Update QR
for i in 1:j-1
  _γ = _c[i]*_H[i,j] + _s[i]*_H[i+1,j]
  _H[i+1,j] = -_s[i]*_H[i,j] + _c[i]*_H[i+1,j]
  _H[i,j] = _γ
end

# New Givens rotation, update QR and residual
_c[j], _s[j], _ = LinearAlgebra.givensAlgorithm(_H[j,j],_H[j+1,j])
_H[j,j] = _c[j]*_H[j,j] + _s[j]*_H[j+1,j]; _H[j+1,j] = 0.0
_g[j+1] = -_s[j]*_g[j]; _g[j] = _c[j]*_g[j]

_β  = abs(_g[j+1])
j += 1

j = j-1

# Solve least squares problem Hy = g by backward substitution
for i in j:-1:1
  _g[i] = (_g[i] - dot(_H[i,i+1:j],_g[i+1:j])) / _H[i,i]
end

# Update solution & residual
for i in 1:j
  _x .+= _g[i] .* _Z[i]
end
krylov_residual!(_V[1],_x,_A,_b,_Pl,_zl)
