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

# domain = (0,1,0,1)
# partition = (2,2)
# model = CartesianDiscreteModel(domain,partition)
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
# Γn = BoundaryTriangulation(model,tags=[7,8])
Γn = BoundaryTriangulation(model,tags=["neumann"])
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
# test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
info = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,st_mdeim=true)

rbsolver = RBSolver(info,fesolver)

snaps,comp = RB.fe_solutions(rbsolver,feop,uh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)

son = select_snapshots(snaps,RB.online_params(info))
ron = get_realization(son)
xrb, = solve(rbsolver,red_op,ron)
son_rev = reverse_snapshots(son)
norm(xrb - son_rev)

info_space = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
rbsolver_space = RBSolver(info_space,fesolver)
red_op_space = reduced_operator(rbsolver_space,feop,snaps)
xrb_space, = solve(rbsolver_space,red_op_space,ron)
norm(xrb_space - son_rev)

xrb_loaded = load_solve(rbsolver,feop)
xrb_space_loaded = load_solve(rbsolver_space,feop)

#
# red_trial,red_test = reduced_fe_space(info,feop,snaps)
ϵ = RB.get_tol(info)
norm_matrix = RB.get_norm_matrix(info,feop)
soff = select_snapshots(snaps,RB.offline_params(info))
odeop = get_algebraic_operator(feop)

χ = rand(647,500)
@time svd(χ)
@time svd(soff)
# basis_space,basis_time = reduced_basis(soff,norm_matrix;ϵ)
flag,s = RB._return_flag(soff)
b1 = tpod(soff,norm_matrix;ϵ)
compressed_s = compress(b1,s)
compressed_s = change_mode(compressed_s)
b2 = tpod(compressed_s;ϵ)
RB._return_bases(flag,b1,b2)

struct S1{T,F} <: AbstractMatrix{T}
  f::F
  S1(f::F) where F = new{eltype(f),F}(f)
end
Base.size(s::S1) = size(s.f)
Base.eltype(::S1{T}) where T = T
Base.eltype(::Type{S1{T}}) where T = T
Base.@propagate_inbounds Base.getindex(s::S1,i...) = getindex(s.f,i...)
Base.@propagate_inbounds Base.getindex(s::S1,i) = getindex(s.f,i)
Base.@propagate_inbounds Base.getindex(s::S1,i,j) = getindex(s.f,i,j)
Base.IndexStyle(::Type{<:S1}) = IndexLinear() #IndexCartesian()

t = rand(1000,100)
s = S1(t)
@btime $s'*$s
@btime $t'*$t
