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
n = 20
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 2
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

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(compute_error(results))
println(compute_speedup(results))

using Gridap.CellData
using Gridap.FESpaces
using Gridap.ODEs
using Mabla.FEM.IndexMaps

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
op = get_algebraic_operator(feop)
op = TransientPGOperator(op,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
j,r = jacobian_and_residual(rbsolver,op,smdeim)
red_jac = reduced_jacobian(rbsolver,op,j)
red_res = reduced_residual(rbsolver,op,r)

j1 = j[1][1] #r[1]#
basis = reduced_basis(j1)
proj_basis = reduce_operator(rbsolver.mdeim_style,basis,red_trial,red_test)

cB,cU,cV = basis.cores_space[1],red_trial.basis.cores_space[1],red_test.basis.cores_space[1]
cc = compress_core(cB,cU,cV)
oldcc = old_compress_core(cB,cU,cV)

w = zeros(Float64,size(cB,2))
mul!(w,cB[ia1,:,:,ia4],cU[ibU1,:,ibU3])

err = cc - oldcc
ibV1,ia1,ibU1,ibV3,ia4,ibU3 = 1,1,1,1,2,1
wold = zeros(Float64,size(cB,2))
mul!(wold,cB[ia1,:,:,ia4],cU[ibU1,:,ibU3])

s = j1[:,:,1,1]
bs = get_basis_space(basis)

spod = change_index_map(TrivialIndexMap,fesnaps)
red_trialpod,red_testpod = reduced_fe_space(rbsolver,feop,spod)
jpod = change_index_map(TrivialIndexMap,j1)
bpod = reduced_basis(jpod)
proj_bpod = reduce_operator(rbsolver.mdeim_style,bpod,red_trialpod,red_testpod)

function old_compress_core(a::AbstractArray{T,4},btrial::AbstractArray{S,3},btest::AbstractArray{S,3};
  kwargs...) where {T,S}

  TS = promote_type(T,S)
  bab = zeros(TS,size(btest,1),size(a,1),size(btrial,1),size(btest,3),size(a,4),size(btrial,3))
  w = zeros(TS,size(a,2))
  @inbounds for i = CartesianIndices(size(bab))
    ibV1,ia1,ibU1,ibV3,ia4,ibU3 = Tuple(i)
    mul!(w,a[ia1,:,:,ia4],btrial[ibU1,:,ibU3])
    bab[i] = btest[ibV1,:,ibV3]'*w
  end
  return bab
end
