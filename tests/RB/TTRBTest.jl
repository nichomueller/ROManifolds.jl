using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.Helpers
using Gridap.TensorValues
using BlockArrays
using DrWatson
using Kronecker
using Mabla.FEM
using Mabla.RB

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (10,10)
model = TProductModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

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

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=20,nsnaps_test=1,nsnaps_mdeim=10)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

params = [
  [0.1,0.9,0.5],
  [0.2,0.4,0.8],
  [0.3,0.7,0.4],
  [0.9,0.2,0.4],
  [0.5,0.5,0.6],
  [0.8,0.4,0.2],
  [0.3,0.4,0.3],
  [0.1,0.2,0.9],
  [0.9,0.2,0.1],
  [0.4,0.6,0.5],
  [0.2,0.5,0.5],
  [0.1,0.2,1.0],
  [0.2,0.7,0.1],
  [0.2,0.2,0.2],
  [0.9,0.5,0.1],
  [0.8,0.7,0.2],
  [0.1,0.1,0.7],
  [0.1,0.7,0.9],
  [0.4,0.4,0.1],
  [0.4,0.3,0.5],
  [0.2,0.3,0.6]
]
r = TransientParamRealization(ParamRealization(params),t0:dt:tf)

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ;r)

rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))

soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
red_trial,red_test = reduced_fe_space(rbsolver,feop,soff)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
jjac,rres = jacobian_and_residual(rbsolver,pop,smdeim)

mdeim_style = rbsolver.mdeim_style
basis = reduced_basis(jjac[1][1];ϵ=RB.get_tol(rbsolver))
lu_interp,integration_domain = mdeim(mdeim_style,basis)
proj_basis = reduce_operator(mdeim_style,basis,red_trial,red_test;combine=(x,y)->x)
coefficient = RB.allocate_coefficient(rbsolver,basis)
result = RB.allocate_result(rbsolver,red_trial,red_test)

basis_spacetime = get_basis_spacetime(basis)
indices_spacetime = RB.get_mdeim_indices(basis_spacetime)

aad = rbop.lhs[1][1]
bas = aad.basis
coeff = aad.coefficient

jcores = RB.get_cores(basis)
Vcores = RB.get_cores(RB.get_basis(red_test))
Ucores = RB.get_cores(RB.get_basis(red_trial))
ccores = map((a,b...)->RB.compress_core(a,b...;combine=(x,y)->x),jcores,Ucores,Vcores)
ccore = RB.multiply_cores(ccores...)

cspace = dropdims(RB.multiply_cores(ccores[1],ccores[2]);dims=(1,2,3))

ccore1 = RB.compress_core(jcores[1],Ucores[1],Vcores[1];combine=(x,y)->x)
ccore2 = RB.compress_core(jcores[2],Ucores[2],Vcores[2];combine=(x,y)->x)
cspace = dropdims(RB.multiply_cores(ccore1,ccore2);dims=(1,2,3))

X = assemble_norm_matrix(feop)
M1 = X.arrays_1d[1]
M2 = X.arrays_1d[2]
A1 = X.gradients_1d[1]
A2 = X.gradients_1d[2]

A = X.array

nzi1 = RB.get_nonzero_indices(A1)
nzi2 = RB.get_nonzero_indices(A2)
nzi = nzi1 .+ (nzi2.-1)'.*length(nzi1)
nzi12 = RB.get_nonzero_indices(X.array)

space_dofs = reshape(collect((10,9,10,9)),:,2)
tensor_indices = RB.tensorize_indices(nzi12,vec(prod(space_dofs;dims=1)))

_,_,nzvmx = findnz(M1)
_,_,nzvmy = findnz(M2)
_,_,nzvax = findnz(A1)
_,_,nzvay = findnz(A2)
nzv = nzvmx*nzvmy' + nzvmx*nzvay' + nzvax*nzvmy'

I,J,V = findnz(X.array)

Nx,Ny = 10,9
ix = fast_index(I,Nx)
iy = slow_index(I,Nx)
ix,iy = RB.tens

sp = FEM.univariate_sparsities(trial(nothing),test)

using SparseArrays
U,V = trial(nothing),test
sparsity = FEM.tp_sparsity(U,V)
I,J,_ = findnz(sparsity)
Ix,Jx,_ = findnz(sparsity.matrices_1d[1])
Iy,Jy,_ = findnz(sparsity.matrices_1d[2])
IJ = get_nonzero_indices(sparsity)

unrows = FEM.univariate_num_rows(sparsity)
uncols = FEM.univariate_num_cols(sparsity)
i = 50
irows = Tuple(tensorize_indices(I[i],unrows))
icols = Tuple(tensorize_indices(J[i],uncols))

rowx,colx = irows[1],icols[1]
px = findfirst(Tuple.(CartesianIndex.(Ix,Jx)).== [(rowx,colx)])
rowy,coly = irows[2],icols[2]

# FEM.sparse_index_map(U,V)
function _global2local(i::Vector{<:CartesianIndex},j::CartesianIndex)
  findfirst(i.==[j])
end
i,j,_ = FEM.univariate_findnz(sparsity)
lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

findall(IJ .== 820)
