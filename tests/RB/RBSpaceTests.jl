using LinearAlgebra
using Plots
using Test
using Gridap
using Gridap.Helpers
using Mabla.FEM
using Mabla.RB

ns = 100
nt = 10
np = 5
pranges = fill([0,1],3)
tdomain = 0:1:10
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

v = [rand(ns) for i = 1:np*nt]
a = ParamArray(v)
s = Snapshots(a,r)
Us,Ss,Vs = svd(s)
s2 = RB.change_mode(s)
Us2,Ss2,Vs2 = svd(s2)

w = [v[(i-1)*np+1:i*np] for i = 1:nt]
b = ParamArray.(w)
t = Snapshots(b,r)
Ut,St,Vt = svd(t)
t2 = RB.change_mode(t)
Ut2,St2,Vt2 = svd(t2)

A = stack(v)
UA,SA,VA = svd(A)
x = map(1:np) do ip
  hcat(v[ip:np:nt*np]...)'
end
B = hcat(x...)
UB,SB,VB = svd(B)

@check Ut ≈ Us ≈ UA
@check St ≈ Ss ≈ SA
@check Ut2 ≈ Us2 ≈ UB
@check St2 ≈ Ss2 ≈ SB

v1 = A[:,rand(axes(A,2))]
w1 = B[:,rand(axes(B,2))]

@check norm(UA*UA'*v1 - v1)/sqrt(ns) ≤ 1e-12
@check norm(UB*UB'*w1 - w1)/sqrt(nt) ≤ 1e-12

nparts = 2
nrowsA = floor(Int,ns/nparts)
A_parts = [A[(i-1)*nrows+1:i*nrows,:] for i = 1:nparts]
v1_parts = [v1[(i-1)*nrows+1:i*nrows] for i = 1:nparts]

UA_parts = map(A_parts) do A
  U,V,S = svd(A)
  U
end

solA = hcat(UA[:,1],vcat(map(x->x[:,1],UA_parts)...))
plot(solA)

v1_rec_parts = map(UA_parts,v1_parts) do U,v1
  U*U'*v1
end

@check norm(vcat(v1_rec_parts...) - v1)/sqrt(ns) ≤ 1e-12

# supremizer check
using Gridap
using Gridap.FESpaces
using ForwardDiff
using BlockArrays
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.Fields
using Gridap.MultiField
using BlockArrays
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
jac(μ,t,u,(du,dp),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ + ∫(q*(∇⋅(du)))dΩ
jac_t(μ,t,u,(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("stokes","toy_mesh"))
info = RBInfo(dir;norm_style=[:l2,:l2],nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,
  st_mdeim=true,compute_supremizers=true)

rbsolver = RBSolver(info,fesolver)

snaps,comp = ode_solutions(rbsolver,feop,xh0μ)

norm_matrix = RB.assemble_norm_matrix(feop)
soff = select_snapshots(snaps,RB.offline_params(info))
bases = reduced_basis(soff,norm_matrix;ϵ=RB.get_tol(info))

# RB.enrich_basis(feop,bases,norm_matrix)
_basis_space,_basis_time = bases
supr_op = RB.compute_supremizer_operator(feop)
# basis_space = add_space_supremizers(_basis_space,supr_op,norm_matrix)
basis_primal,basis_dual = _basis_space.array
norm_matrix_primal = first(norm_matrix)
supr_i = supr_op * basis_dual
gram_schmidt!(supr_i,basis_primal,norm_matrix_primal)
basis_primal = hcat(basis_primal,supr_i)

C = basis_primal'*supr_op *basis_dual
Uc,Sc,Vc = svd(C)
@check all(abs.(Sc) .> 1e-2)

# basis_time = add_time_supremizers(_basis_time)
basis_primal,basis_dual = _basis_time.array
basis_pd = basis_primal'*basis_dual
for ntd = axes(basis_dual,2)
  proj = ntd == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd[:,ntd],basis_pd[:,1:ntd-1])
  dist = norm(basis_pd[:,1]-proj)
  println(dist > 1e-2)
end

basis_time = RB.add_time_supremizers(_basis_time)
basis_primal,basis_dual = basis_time.array
basis_pd = basis_primal'*basis_dual
Upd,Spd,Vpd = svd(basis_pd)
@check all(abs.(Spd) .> 1e-2)
