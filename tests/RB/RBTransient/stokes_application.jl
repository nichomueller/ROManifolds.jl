using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra, Gridap.ODEs
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

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"top",[3,4,6])
add_tag_from_tags!(labels,"bottom",[1,2,5])
add_tag_from_tags!(labels,"walls",[7,8])

order = 2
degree = 2*(order+1)
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = μ[2]
g_in(x,μ,t) = VectorValue(inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)

α = 1.e2
Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=degree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
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
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["top","bottom"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)#conformity=:L2,constraint=:zeromean)
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
# solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=true)
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
odesolver = ThetaMethod(solver,dt,θ)

r = realization(feop)

ϵ = 1e-5
rbsolver = RBSolver(odesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ;r)

fesolver′ = LUSolver()
odesolver′ = ThetaMethod(fesolver′,dt,θ)
rbsolver′ = RBSolver(odesolver′,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
fesnaps′,festats′ = fe_solutions(rbsolver′,feop,xh0μ;r)

################################################################################
#################################################################################
#################################################################################
# CASE j=4
solver,A,Pl,Pr,caches = ns.solver,ns.A,ns.Pl_ns,ns.Pr_ns,ns.caches
V,Z,zl,H,g,c,s = copy.(caches)
m   = LinearSolvers.krylov_cache_length(ns)
log = solver.log

plength = param_length(x)

fill!(V[1],zero(eltype(V[1])))
fill!(zl,zero(eltype(zl)))

# Initial residual
LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
β    = norm(V[1])
done = LinearSolvers.init!(log,maximum(β))

# Arnoldi process
j = 1
V[1] ./= β
fill!(H,0.0)
fill!(g,0.0); g.data[1,:] = β
for j = 1:1
  # Expand Krylov basis if needed
  if j > m
    H,g,c,s = expand_param_krylov_caches!(ns)
    m = LinearSolvers.krylov_cache_length(ns)
  end

  # Arnoldi orthogonalization by Modified Gram-Schmidt
  fill!(V[j+1],zero(eltype(V[j+1])))
  fill!(Z[j],zero(eltype(Z[j])))
  LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)

  using Mabla.FEM.ParamAlgebra
  for k in 1:plength
    Vk = map(V->param_getindex(V,k),V)
    Zk = map(Z->param_getindex(Z,k),Z)
    zlk = param_getindex(zl,k)
    Hk = param_getindex(H,k)
    gk = param_getindex(g,k)
    ck = param_getindex(c,k)
    sk = param_getindex(s,k)
    ParamAlgebra._gs_qr_givens!(Vk,Zk,zlk,Hk,gk,ck,sk;j)
  end

  β  = abs.(g.data[j+1,:])
  j += 1
  done = LinearSolvers.update!(log,maximum(β))
end

j = 2
fill!(V[j+1],zero(eltype(V[j+1])))
fill!(Z[j],zero(eltype(Z[j])))

wr = copy(Z[j])
y = copy(V[j])
# solve!(wr,Pr,y)
using BlockArrays
NB = length(Pr.block_ns)
cc = Pr.solver.coeffs
ww,yy = Pr.work_caches
mats = Pr.block_caches
iB = 1
wi = ww[Block(iB)]
copy!(wi,y[Block(iB)])
for jB in iB+1:NB
  println("dio porco")
  cij = cc[iB,jB]
  if abs(cij) > eps(cij)
    xj = wr[Block(jB)]
    mul!(wi,mats[iB,jB],xj,-cij,1.0)
  end
end
###
iB=1
jB=2
cij = cc[iB,jB]
xj = wr[Block(jB)]
mul!(wi,mats[iB,jB],xj,-cij,1.0)
_xj = _wr[Block(jB)]
mul!(_wi,_mats[iB,jB],_xj,-cij,1.0)
###
# Solve diagonal block
nsi = Pr.block_ns[iB]
xi  = wr[Block(iB)]
yi  = yy[Block(iB)]
solve!(yi,nsi,wi)
copy!(xi,yi)

# ssolver,AA,Pl,caches = nsi.solver,nsi.A,nsi.Pl_ns,nsi.caches
# flexible,log = ssolver.flexible,ssolver.log
# w,p,z,r = caches

# plength = param_length(y)

# # Initial residual
# mul!(w,AA,y); r .= b .- w
# fill!(p,zero(eltype(p)))
# fill!(z,zero(eltype(z)))
# γ = ones(eltype2(p),plength)

# res  = norm(r)
# done = LinearSolvers.init!(log,maximum(res))
# while !done

#   if !flexible # β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
#     solve!(z,Pl,r)
#     β = γ; γ = dot(z,r); β = γ / β
#   else         # β = (zₖ₊₁ ⋅ (rₖ₊₁-rₖ))/(zₖ ⋅ rₖ)
#     δ = dot(z,r)
#     solve!(z,Pl,r)
#     β = γ; γ = dot(z,r); β = (γ-δ) / β
#   end

#   for k in 1:plength
#     xk = param_getindex(y,k)
#     wk = param_getindex(w,k)
#     Ak = param_getindex(AA,k)
#     zk = param_getindex(z,k)
#     pk = param_getindex(p,k)
#     rk = param_getindex(r,k)

#     pk .= zk .+ β[k] .* pk

#     # w = A⋅p
#     mul!(wk,Ak,pk)
#     α = γ[k] / dot(pk,wk)

#     # Update solution and residual
#     xk .+= α .* pk
#     rk .-= α .* wk
#   end

#   res  = norm(r)
#   done = LinearSolvers.update!(log,maximum(res))
# end

################################################################################
# GRIDAP

# solve!(_x,_ns,_b)
_solver, _A, _Pl, _Pr, _caches = _ns.solver, _ns.A, _ns.Pl_ns, _ns.Pr_ns, _ns.caches
_V, _Z, _zl, _H, _g, _c, _s = copy.(_caches)
_m   = LinearSolvers.krylov_cache_length(_ns)
_log = _solver.log

fill!(_V[1],zero(eltype(_V[1])))
fill!(_zl,zero(eltype(_zl)))

# Initial residual
LinearSolvers.krylov_residual!(_V[1],_x,_A,_b,_Pl,_zl)
_β    = norm(_V[1])
done = LinearSolvers.init!(_log,_β)

_j = 1
_V[1] ./= _β
fill!(_H,0.0)
fill!(_g,0.0); _g[1] = _β
for _j = 1:1
  # Expand Krylov basis if needed
  if _j > _m
    _H, _g, _c, _s = LinearSolvers.expand_krylov_caches!(_ns)
    _m = LinearSolvers.krylov_cache_length(_ns)
  end

  # Arnoldi orthogonalization by Modified Gram-Schmidt
  fill!(_V[_j+1],zero(eltype(_V[_j+1])))
  fill!(_Z[_j],zero(eltype(_Z[_j])))
  LinearSolvers.krylov_mul!(_V[_j+1],_A,_V[_j],_Pr,_Pl,_Z[_j],_zl)
  for i in 1:_j
    _H[i,_j] = dot(_V[_j+1],_V[i])
    _V[_j+1] .= _V[_j+1] .- _H[i,_j] .* _V[i]
  end
  _H[_j+1,_j] = norm(_V[_j+1])
  _V[_j+1] ./= _H[_j+1,_j]

  # Update QR
  for i in 1:_j-1
    _γ = _c[i]*_H[i,_j] + _s[i]*_H[i+1,_j]
    _H[i+1,_j] = -_s[i]*_H[i,_j] + _c[i]*_H[i+1,_j]
    _H[i,_j] = _γ
  end

  # New Givens rotation, update QR and residual
  _c[_j], _s[_j], _ = LinearAlgebra.givensAlgorithm(_H[_j,_j],_H[_j+1,_j])
  _H[_j,_j] = _c[_j]*_H[_j,_j] + _s[_j]*_H[_j+1,_j]; _H[_j+1,_j] = 0.0
  _g[_j+1] = -_s[_j]*_g[_j]; _g[_j] = _c[_j]*_g[_j]

  _β  = abs(_g[_j+1])
  _j += 1
  done = LinearSolvers.update!(_log,_β)
end

_j = 2
fill!(_V[_j+1],zero(eltype(_V[_j+1])))
fill!(_Z[_j],zero(eltype(_Z[_j])))

_wr = copy(_Z[_j])
_y = copy(_V[_j])

# solve!(_wr,_Pr,_y)
_cc = _Pr.solver.coeffs
_ww,_yy = _Pr.work_caches
_mats = _Pr.block_caches
_wi = _ww[Block(iB)]
copy!(_wi,_y[Block(iB)])
@assert _ww[Block(1)] ≈ param_getindex(ww[Block(1)],1)
@assert _ww[Block(2)] ≈ param_getindex(ww[Block(2)],1)
@assert _y[Block(1)] ≈ param_getindex(y[Block(1)],1)
@assert _y[Block(2)] ≈ param_getindex(y[Block(2)],1)
for jB in iB+1:NB
  println("dio porco")
  _cij = _cc[iB,jB]
  if abs(_cij) > eps(_cij)
    _xj = _wr[Block(jB)]
    mul!(_wi,_mats[iB,jB],_xj,-_cij,1.0)
  end
end
# Solve diagonal block
_nsi = _Pr.block_ns[iB]
_xi  = _wr[Block(iB)]
_yi  = _yy[Block(iB)]
solve!(_yi,_nsi,_wi)
copy!(_xi,_yi)

# _ssolver,_AA,_Pl,_caches = _Pr.solver,_Pr.A,_Pr.Pl_ns,_Pr.caches
# _flexible,_log = _ssolver.flexible,_ssolver.log
# _w,_p,_z,_r = _caches

# # Initial residual
# mul!(_w,_AA,_y); _r .= _b .- _w
# fill!(_p,zero(eltype(_p)))
# fill!(_z,zero(eltype(_z)))
# _γ = 1.0

# _res  = norm(_r)
# _done = LinearSolvers.init!(_log,_res)
# while !_done

#   if !_flexible # β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
#     solve!(_z,_Pl,_r)
#     _β = _γ; _γ = dot(_z,_r); _β = _γ / _β
#   else         # β = (zₖ₊₁ ⋅ (rₖ₊₁-rₖ))/(zₖ ⋅ rₖ)
#     _δ = dot(_z,_r)
#     solve!(_z,_Pl,_r)
#     _β = _γ; _γ = dot(_z,_r); _β = (_γ-_δ) / _β
#   end

#   _p .= _z .+ _β .* _p

#   # w = A⋅p
#   mul!(_w,_A,_p)
#   _α = _γ / dot(_p,_w)

#   # Update solution and residual
#   _x .+= _α .* _p
#   _r .-= _α .* _w

#   _res  = norm(_r)
#   _done = LinearSolvers.update!(_log,_res)
# end

function Algebra.solve!(x::AbstractBlockVector,ns::GridapSolvers.BlockSolvers.BlockTriangularSolverNS{Val{:upper}},b::AbstractBlockVector)
  @assert blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w, y = ns.work_caches
  mats = ns.block_caches
  for iB in NB:-1:1
    # Add upper off-diagonal contributions
    wi  = blocks(w)[iB]
    copy!(wi,blocks(b)[iB])
    for jB in iB+1:NB
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = blocks(x)[jB]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
        # println(norm(wi))
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB]
    xi  = blocks(x)[iB]
    yi  = blocks(y)[iB]
    solve!(yi,nsi,wi)
    copy!(xi,yi) # Remove this with PA 0.4
  end
  return x
end
