using LinearAlgebra
using BlockArrays
using SparseArrays
using Plots
using Test
using DrWatson

using Gridap
using Gridap.FESpaces
using Gridap.Helpers

using Mabla.FEM
using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

ns = 100
nt = 10
np = 5
pranges = fill([0,1],3)
tdomain = 0:1:10
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)
v = [ParamArray([rand(ns) for _ = 1:np]) for _ = 1:nt]

X̃ = sprand(Float64,ns,ns,0.5)
X′ = (X̃ + X̃') / 2
X = X′'*X′ / norm(X′)^2

# case 1
i = TrivialIndexMap(LinearIndices((ns,)))
s = Snapshots(v,i,r)
basis = projection(s,X)

bs = get_basis_space(basis)
bt = get_basis_time(basis)
s1 = flatten_snapshots(s)
s2 = RBTransient.change_mode(s1)

@check norm(bs*bs'*X*s1 - s1)/sqrt(ns) ≤ 1e-12
@check norm(bt*bt'*s2 - s2)/sqrt(nt) ≤ 1e-12
@check norm(bs'*X*bs - I) ≤ 1e-12
@check norm(bt'*bt - I) ≤ 1e-12

# case 2
nsx,nxy = 10,10
i = IndexMap(collect(LinearIndices((nsx,nxy))))
s = Snapshots(v,i,r)
basis = projection(s)

bs = get_basis_space(basis)
bt = get_basis_time(basis)
s1 = flatten_snapshots(s)
s2 = RBTransient.change_mode(s1)
bst = RBTransient.get_basis_spacetime(basis)

s12 = map(eachslice(s;dims=4)) do s
  vec(collect(s))
end |> stack

@check norm(bst'*bst - I) ≤ 1e-12
@check norm(bst*bst'*s12 - s12)/sqrt(ns) ≤ 1e-12

# FEM problem

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.05

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (5,5)
model = TProductModel(domain,partition)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ
res(μ,t,u,v) = mass(μ,t,∂t(u),v) + stiffness(μ,t,u,v) - rhs(μ,t,v)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1)
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,trial,test)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nparams_state=50,nparams_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

fesnaps,festats = fe_snapshots(rbsolver,feop,uh0μ)
X = assemble_norm_matrix(feop)

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
basis = RBSteady.get_basis(red_test)
bs = RBTransient.get_basis_space(basis)

@check bs ≈ cores2basis(RBSteady.get_cores_space(basis)...)
@check norm(bs'*X*bs - I) ≤ 1e-12

u = get_values(fesnaps)
r = get_realization(fesnaps)
op = get_algebraic_operator(feop)
cache = allocate_odecache(fesolver,op,r,(u,))
A,_ = jacobian_and_residual(fesolver,op,r,(u,),cache)
iA = get_matrix_index_map(feop)
sA = Snapshots(A,iA,r)
basis = reduced_basis(sA)

interpolation,integration_domain = mdeim(basis)

# supremizer check

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
info = RBInfo(dir;norm_style=[:l2,:l2],nparams_state=50,nparams_test=10,nsnaps_mdeim=20,
  st_mdeim=true,compute_supremizers=true)

rbsolver = RBSolver(info,fesolver)

snaps,comp = fe_snapshots(rbsolver,feop,xh0μ)

norm_matrix = RB.assemble_norm_matrix(feop)
soff = select_snapshots(snaps,RB.offline_params(info))
bases = reduced_basis(soff,norm_matrix;ϵ=RB.get_tol(info))

# RB.enrich_basis(feop,bases,norm_matrix)
_basis_space,_basis_time = bases
supr_op = RB.assemble_coupling_matrix(feop)
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
  dist = norm(basis_pd[:,i]-proj)
  println(dist > 1e-2)
end

basis_time = RB.add_time_supremizers(_basis_time)
basis_primal,basis_dual = basis_time.array
basis_pd = basis_primal'*basis_dual
Upd,Spd,Vpd = svd(basis_pd)
@check all(abs.(Spd) .> 1e-2)
