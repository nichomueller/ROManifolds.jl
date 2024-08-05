using Gridap
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using LinearAlgebra
using Test
using DrWatson
using Serialization

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

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:2dt
ptspace = TransientParamSpace(pranges,tdomain)

n = 5
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,6])

order = 2
degree = 2*(order+1)
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = -μ[2]*t
g_in(x,μ,t) = VectorValue(inflow(μ,t)*x[2]*(1-x[2]),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)

α = 1#.e2
Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=degree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ #+ graddiv(u,v,dΩ)
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = (∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ)*(1/dt)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0) #)#
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
solver_u = LUSolver()
# solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-7,verbose=false)
solver_p = LUSolver()
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-7,verbose=false)
odesolver = ThetaMethod(solver,dt,θ)

r = realization(feop;nparams=1)

ϵ = 1e-5
rbsolver = RBSolver(odesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ;r)

# fesolver′ = LUSolver()
# odesolver′ = ThetaMethod(fesolver′,dt,θ)
# rbsolver′ = RBSolver(odesolver′,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# fesnaps′,festats′ = fe_solutions(rbsolver′,feop,xh0μ;r)

# using Gridap.ODEs
# using Gridap.Algebra
sol = solve(odesolver,feop,xh0μ;r)
vals,_ = collect(sol.odesol)
x1 = vals[1]
U1 = trial(r)
xh1 = FEFunction(U1,x1)
uh1,ph1 = xh1
sum(∫(ph1)dΩ)

r0 = ParamDataStructures.get_at_time(r,:initial)
odeop = get_algebraic_operator(feop.op)
us0 = (get_free_dof_values(xh0μ(r0.params)),)
odecache = allocate_odecache(odesolver,odeop,r0,us0)
state0,cache = ode_start(odesolver,odeop,r0,us0,odecache)
statef = copy.(state0)

odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

sysslvr = odesolver.sysslvr
x = statef[1]
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r0,dtθ)
usx = (state0[1],x)
ws = (dtθ,1)
update_odeopcache!(odeopcache,odeop,r0)
stageop = LinearParamStageOperator(odeop,odeopcache,r0,usx,ws,A,b,reuse,sysslvrcache)
# sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
A = stageop.A
ss = symbolic_setup(sysslvr,A)
ns = numerical_setup(ss,A)
b = stageop.b
rmul!(b,-1)

using BlockArrays
VV = fesnaps[1].data[1][1]
vels = fesnaps[1].
X = mortar([x[1][:,1,1],x[2][:,1,1]])

err = param_getindex(A,1)*X-param_getindex(b,1)
ciao
# # solve!(x,ns,b)
# solver,A,Pl,Pr,caches = ns.solver,ns.A,ns.Pl_ns,ns.Pr_ns,ns.caches
# V,Z,zl,H,g,c,s = copy.(caches)
# m   = LinearSolvers.krylov_cache_length(ns)
# log = solver.log

# plength = param_length(x)

# fill!(V[1],zero(eltype(V[1])))
# fill!(zl,zero(eltype(zl)))

# # Initial residual
# LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
# β    = norm(V[1])
# done = LinearSolvers.init!(log,maximum(β))

# # Arnoldi process
# j = 1
# V[1] ./= β
# fill!(H,0.0)
# fill!(g,0.0); g.data[1,:] = β
# while j < 10
#   # Expand Krylov basis if needed
#   if j > m
#     H,g,c,s = ParamAlgebra.expand_param_krylov_caches!(ns)
#     m = LinearSolvers.krylov_cache_length(ns)
#   end

#   # Arnoldi orthogonalization by Modified Gram-Schmidt
#   fill!(V[j+1],zero(eltype(V[j+1])))
#   fill!(Z[j],zero(eltype(Z[j])))
#   LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)

#   using Mabla.FEM.ParamAlgebra
#   for k in 1:plength
#     Vk = map(V->param_getindex(V,k),V)
#     Zk = map(Z->param_getindex(Z,k),Z)
#     zlk = param_getindex(zl,k)
#     Hk = param_getindex(H,k)
#     gk = param_getindex(g,k)
#     ck = param_getindex(c,k)
#     sk = param_getindex(s,k)
#     ParamAlgebra._gs_qr_givens!(Vk,Zk,zlk,Hk,gk,ck,sk;j)
#   end

#   β  = abs.(g.data[j+1,:])
#   j += 1
#   done = LinearSolvers.update!(log,maximum(β))
# end
# j = j-1

# for k in 1:plength
#   # Solve least squares problem Hy = g by backward substitution
#   for i in j:-1:1
#     g.data[i,k] = (g.data[i,k] - dot(H.data[i,i+1:j,k],g.data[i+1:j,k])) / H.data[i,i,k]
#   end

#   # Update solution & residual
#   xk = param_getindex(x,k)
#   for i in 1:j
#     Zik = param_getindex(Z[i],k)
#     xk .+= g.data[i,k] .* Zik
#   end
# end
# LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)

#################################################################################
#################################################################################

μ = get_params(r).params[1]

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)
_stiffness(t,(u,p),(v,q),dΩ) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
_stiffness(t,u,v) = _stiffness(t,u,v,dΩ)
_mass(t,(u,p),(v,q),dΩ) = ∫(v⋅u)dΩ
_mass(t,u,v) = _mass(t,u,v,dΩ)
_g_in(x,t) = g_in(x,μ,t)
_g_in(t) = x->_g_in(x,t)
_g_w(x,t) = g_w(x,μ,t)
_g_w(t) = x->_g_w(x,t)
_rhs(t,(v,q),dΩ) = ∫(VectorValue(0.0,0.0)⋅v)dΩ
_rhs(t,v) = _rhs(t,v,dΩ)

_trial_u = TransientTrialFESpace(test_u,[_g_in,_g_w])
_trial = TransientMultiFieldParamFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = TransientLinearFEOperator((_stiffness,_mass),_rhs,_trial,test)

_xh0 = interpolate_everywhere([u0μ(get_params(r)[1]),p0μ(get_params(r)[1])],_trial(0.0))

fesltn = solve(odesolver,_feop,t0,2dt,_xh0)
UU,PP = [],[]
for (t_n,xhs_n) in fesltn
  uhs_n,phs_n = xhs_n
  push!(UU,copy(get_free_dof_values(uhs_n)))
  push!(PP,copy(get_free_dof_values(phs_n)))
end

for (t_n,xhs_n) in fesltn
  uhs_n,phs_n = xhs_n
  println(sum(∫(phs_n)dΩ))
end

# _odeop = get_algebraic_operator(_feop)
# _us0 = (get_free_dof_values(_xh0),)
# _odecache = allocate_odecache(odesolver,_odeop,t0,_us0)
# _state0,_cache = ode_start(odesolver,_odeop,t0,_us0,_odecache)
# _statef = copy.(_state0)

# _odeslvrcache,_odeopcache = _odecache
# _reuse,_A,_b,_sysslvrcache = _odeslvrcache

# _x = _statef[1]
# fill!(_x,zero(eltype(_x)))
# _usx = (_state0[1],_x)
# t = t0+θ*dt
# update_odeopcache!(_odeopcache,_odeop,t)
# _stageop = LinearStageOperator(_odeop,_odeopcache,t,_usx,ws,_A,_b,_reuse,_sysslvrcache)
# # _sysslvrcache = solve!(_x,sysslvr,_stageop,_sysslvrcache)
# _A = _stageop.J
# _ss = symbolic_setup(sysslvr,_A)
# _ns = numerical_setup(_ss,_A)
# _b = _stageop.r
# rmul!(_b,-1)

# # solve!(_x,_ns,_b)
# _solver, _A, _Pl, _Pr, _caches = _ns.solver, _ns.A, _ns.Pl_ns, _ns.Pr_ns, _ns.caches
# _V, _Z, _zl, _H, _g, _c, _s = copy.(_caches)
# _m   = LinearSolvers.krylov_cache_length(_ns)
# _log = _solver.log

# fill!(_V[1],zero(eltype(_V[1])))
# fill!(_zl,zero(eltype(_zl)))

# # Initial residual
# LinearSolvers.krylov_residual!(_V[1],_x,_A,_b,_Pl,_zl)
# _β    = norm(_V[1])
# _done = LinearSolvers.init!(_log,_β)
# # while !done
# # Arnoldi process
# _j = 1
# _V[1] ./= _β
# fill!(_H,0.0)
# fill!(_g,0.0); _g[1] = _β
# while _j < 10
#   # Expand Krylov basis if needed
#   if _j > _m
#     _H, _g, _c, _s = LinearSolvers.expand_krylov_caches!(_ns)
#     _m = LinearSolvers.krylov_cache_length(_ns)
#   end

#   # Arnoldi orthogonalization by Modified Gram-Schmidt
#   fill!(_V[_j+1],zero(eltype(_V[_j+1])))
#   fill!(_Z[_j],zero(eltype(_Z[_j])))
#   LinearSolvers.krylov_mul!(_V[_j+1],_A,_V[_j],_Pr,_Pl,_Z[_j],_zl)
#   for i in 1:_j
#     _H[i,_j] = dot(_V[_j+1],_V[i])
#     _V[_j+1] .= _V[_j+1] .- _H[i,_j] .* _V[i]
#   end
#   _H[_j+1,_j] = norm(_V[_j+1])
#   _V[_j+1] ./= _H[_j+1,_j]

#   # Update QR
#   for i in 1:_j-1
#     _γ = _c[i]*_H[i,_j] + _s[i]*_H[i+1,_j]
#     _H[i+1,_j] = -_s[i]*_H[i,_j] + _c[i]*_H[i+1,_j]
#     _H[i,_j] = _γ
#   end

#   # New Givens rotation, update QR and residual
#   _c[_j], _s[_j], _ = LinearAlgebra.givensAlgorithm(_H[_j,_j],_H[_j+1,_j])
#   _H[_j,_j] = _c[_j]*_H[_j,_j] + _s[_j]*_H[_j+1,_j]; _H[_j+1,_j] = 0.0
#   _g[_j+1] = -_s[_j]*_g[_j]; _g[_j] = _c[_j]*_g[_j]

#   _β  = abs(_g[_j+1])
#   _j += 1
#   _done = LinearSolvers.update!(_log,_β)
# end
# _j = _j-1

# # Solve least squares problem Hy = g by backward substitution
# for i in _j:-1:1
#   _g[i] = (_g[i] - dot(_H[i,i+1:_j],_g[i+1:_j])) / _H[i,i]
# end

# # Update solution & residual
# for i in 1:_j
#   _x .+= _g[i] .* _Z[i]
# end
# LinearSolvers.krylov_residual!(_V[1],_x,_A,_b,_Pl,_zl)




































































function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"bottom",[1,2,5])
  add_tag_from_tags!(labels,"walls",[7,8])
end

# Geometry
Dc = 2
n = 5
domain = (0,1,0,1)
partition = (n,n)

model = CartesianDiscreteModel(domain,partition)
add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
add_labels!(get_face_labeling(model))

# FE spaces
order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

u_bottom = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

V = TestFESpace(model,reffe_u,dirichlet_tags=["bottom","top"]);
U = TrialFESpace(V,[u_bottom,u_top]);
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

# Weak formulation
α = 1.e2
f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
poly = (Dc==2) ? QUAD : HEX
Π_Qh = LocalProjectionMap(poly,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ + graddiv(u,v,dΩ)
biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
liform((v,q),dΩ) = ∫(v⋅f)dΩ

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,X,Y)
A, b = get_matrix(op), get_vector(op);

# Solver
solver_u = LUSolver() # or mumps
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)
solver_p.log.depth = 2

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8)
ns = numerical_setup(symbolic_setup(solver,A),A)

using Gridap.Algebra
x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)
uh,ph = FEFunction(X,x)
