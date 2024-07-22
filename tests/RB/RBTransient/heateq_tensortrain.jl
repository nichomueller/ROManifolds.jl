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
lu_interp,integration_domain = mdeim(rbsolver.mdeim_style,basis)

cores_space = get_cores_space(basis)
core_time = RBTransient.get_core_time(basis)
_indices_spacetime,_interp_basis_spacetime = empirical_interpolation(cores_space...,core_time)

i = get_index_map(j1)
bst = RBTransient.get_basis_spacetime(i,cores_space,core_time)
# indices_spacetime,interp_basis_spacetime = empirical_interpolation(bst)

b1 = vec(select_snapshots(j1,1))
b1 - bst*(_interp_basis_spacetime\b1[_indices_spacetime])

function old_mdeim(mdeim_style,b::TransientTTSVDCores)
  i = get_index_map(b)
  bst = RBTransient.get_basis_spacetime(i,get_cores_space(b),RBTransient.get_core_time(b))
  indices_spacetime,interp_basis_spacetime = empirical_interpolation(bst)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(interp_basis_spacetime)
  integration_domain = TransientIntegrationDomain(indices_space,indices_time)
  return lu_interp,integration_domain
end

old_lu_interp,old_integration_domain = old_mdeim(rbsolver.mdeim_style,basis)

index_map = get_index_map(basis)
cores = get_cores(basis)
C,I,r,Iv = RBSteady._eim_cache(first(cores))
for i = eachindex(cores)
  _,Ai = RBSteady.empirical_interpolation!((I,r,Iv),C)
  if i < length(cores)
    C = RBSteady._next_core(Ai,cores[i+1])
  else
    Ig = RBSteady._to_global_indices(Iv,index_map)
    return Ig,Ai
  end
end

local_indices = Iv

Is...,It = local_indices
Igt = It
Igs = copy(It)
for (i,ii) in enumerate(Igt)
  igti = RBSteady._global_index(ii,last(Is))
  Igt[i] = igti
  Igs[i] = index_map[RBSteady._global_index(igti,Is)]
end

# TPOD
using Mabla.FEM.IndexMaps
using BlockArrays
using SparseArrays
using LinearAlgebra
_dΩ = dΩ.measure

_induced_norm(u,v) = ∫(v*u)_dΩ + ∫(∇(v)⋅∇(u))_dΩ

_test = TestFESpace(model.model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial = TransientTrialParamFESpace(_test,gμt)
_feop = TransientParamLinearFEOperator((stiffness,mass),res,_induced_norm,ptspace,
  _trial,_test,trian_res,trian_stiffness,trian_mass)

_fesnaps,_festats = fe_solutions(rbsolver,_feop,uh0μ;r)

_rbop = reduced_operator(rbsolver,_feop,_fesnaps)
_rbsnaps,_rbstats = solve(rbsolver,_rbop,_fesnaps)
_results = rb_results(rbsolver,_rbop,_fesnaps,_rbsnaps,festats,_rbstats)

println(compute_error(_results))

ad = rbop.lhs[1][1]
_ad = _rbop.lhs[1][1]


# old mdeim
using Gridap.CellData
using Gridap.FESpaces
using Gridap.ODEs
using Mabla.FEM.IndexMaps

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
op = get_algebraic_operator(feop)
op = TransientPGOperator(op,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
j,r = jacobian_and_residual(rbsolver,op,smdeim)
red_jac = old_reduced_jacobian(rbsolver,op,j)
red_res = old_reduced_residual(rbsolver,op,r)

red_lhs,red_rhs = red_jac,red_res
trians_rhs = get_domains(red_rhs)
trians_lhs = map(get_domains,red_lhs)
new_op = change_triangulation(op,trians_rhs,trians_lhs)
rbop_old = TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)
rbsnaps_old,rbstats_old = solve(rbsolver,rbop_old,fesnaps)
results_old = rb_results(rbsolver,rbop_old,fesnaps,rbsnaps_old,festats,rbstats_old)
println(compute_error(results_old))

using PartitionedArrays

function old_mdeim(mdeim_style,b::TransientTTSVDCores)
  i = get_index_map(b)
  bst = RBTransient.get_basis_spacetime(i,get_cores(b)...)
  indices_spacetime,interp_basis_spacetime = empirical_interpolation(bst)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(interp_basis_spacetime)
  integration_domain = TransientIntegrationDomain(indices_space,indices_time)
  return lu_interp,integration_domain
end

function old_reduced_form(
  solver::RBSolver,
  s::AbstractSnapshots,
  trian::Triangulation,
  args...;
  kwargs...)

  mdeim_style = solver.mdeim_style
  basis = reduced_basis(s;ϵ=RBSteady.get_tol(solver))
  lu_interp,integration_domain = old_mdeim(mdeim_style,basis)
  proj_basis = reduce_operator(mdeim_style,basis,args...;kwargs...)
  red_trian = RBSteady.reduce_triangulation(trian,integration_domain,args...)
  coefficient = RBSteady.allocate_coefficient(solver,basis)
  result = RBSteady.allocate_result(solver,args...)
  ad = AffineDecomposition(proj_basis,lu_interp,integration_domain,coefficient,result)
  return ad,red_trian
end

function old_reduced_residual(solver::RBSolver,op,s::AbstractSnapshots,trian::Triangulation)
  test = get_test(op)
  old_reduced_form(solver,s,trian,test)
end

function old_reduced_jacobian(solver::RBSolver,op,s::AbstractSnapshots,trian::Triangulation;kwargs...)
  trial = get_trial(op)
  test = get_test(op)
  old_reduced_form(solver,s,trian,trial,test;kwargs...)
end

function old_reduced_jacobian(
  solver::ThetaMethodRBSolver,
  op::TransientRBOperator,
  contribs::Tuple{Vararg{Any}})

  fesolver = get_fe_solver(solver)
  θ = fesolver.θ
  a = ()
  for (i,c) in enumerate(contribs)
    combine = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*(x-y)
    a = (a...,old_reduced_jacobian(solver,op,c;combine))
  end
  return a
end

function old_reduced_residual(solver::RBSolver,op,c::ArrayContribution)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    old_reduced_residual(solver,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function old_reduced_jacobian(solver::RBSolver,op,c::ArrayContribution;kwargs...)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    old_reduced_jacobian(solver,op,values,trian;kwargs...)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end
