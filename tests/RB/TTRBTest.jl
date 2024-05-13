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

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.05

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (10,10)
model = CartesianDiscreteModel(domain,partition)

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

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ;tt_format=true)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
println(RB.speedup(results))

save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,trial,test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
A,b = jacobian_and_residual(rbsolver,pop,smdeim)

perm = get_dof_permutation(Float64,model,test,order)

vvreffe = ReferenceFE(lagrangian,VectorValue{2,Float64},1)
vvtest = TestFESpace(model,vvreffe;conformity=:H1,dirichlet_tags=["dirichlet"])
tptest = TProductFESpace(model,vvreffe;conformity=:H1,dirichlet_tags=["dirichlet"])

import Mabla.FEM.TProduct
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
reffe = ReferenceFE(lagrangian,Float64,2)
test = TestFESpace(model,reffe;conformity=:H1)
trial = TrialFESpace(test,x->0)
perm = TProduct.get_dof_permutation(Float64,model,test,2)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

domain1d = (0,1)
partition1d = (2,)
model1d = CartesianDiscreteModel(domain1d,partition1d)
reffe1d = ReferenceFE(lagrangian,Float64,2)
test1d = TestFESpace(model1d,reffe1d;conformity=:H1)
trial1d = TrialFESpace(test1d,x->0)
Ω1d = Triangulation(model1d)
dΩ1d = Measure(Ω1d,2)

# test 1
F = assemble_vector(v->∫(v)dΩ,test)
F1d = assemble_vector(v->∫(v)dΩ1d,test1d)
TPF = kronecker(F1d,F1d)
TPF ≈ F
TPF[perm[:]] ≈ F

# test 2
f1d(x) = x[1]
f(x) = x[1]*x[2]
F = assemble_vector(v->∫(f*v)dΩ,test)
F1d = assemble_vector(v->∫(f1d*v)dΩ1d,test1d)
kronecker(F1d,F1d) ≈ F

# test 3
M = assemble_matrix((u,v)->∫(v*u)dΩ,trial,test)
M1d = assemble_matrix((u,v)->∫(v*u)dΩ1d,trial1d,test1d)
kronecker(M1d,M1d) ≈ M

# test 4
M = assemble_matrix((u,v)->∫(f*v*u)dΩ,trial,test)
M1d = assemble_matrix((u,v)->∫(f1d*v*u)dΩ1d,trial1d,test1d)
kronecker(M1d,M1d) ≈ M

_model = TProduct.TProductModel(domain,partition)
_test = TProduct.TProductFESpace(_model,reffe;conformity=:H1)

_perm = _test.dof_permutation

CIAO
# # 1d connectivity
# cell_dof_ids = get_cell_dof_ids(test)
# c1 = get_cell_dof_ids(test1d)
# _cell_dof_ids_1d = copy(c1),copy(c1)
# # _cell_dof_ids_1d = TProduct._setup_1d_connectivities([test1d,test1d])

# v = [1,5,2,6,3,7,4]
# tpv = Vector{typeof(v)}(undef,3)

# function my_recursive_fun(cell_ids,spaces,order,D)
#   function _my_recursive_fun(cell_ids,::Val{1},::Val{d′}) where d′
#     @assert d′ == D
#     return _my_recursive_fun(cell_ids,Val(2),Val(d′-1))
#   end
#   function _my_recursive_fun(cell_ids_prev,::Val{d},::Val{d′}) where {d,d′}
#     space_d = spaces[d]
#     cell_ids_d = get_cell_dof_ids(space_d)
#     ncells_prev = length(cell_ids_prev)
#     ncells_d = ncells_prev*length(cell_ids_d)
#     vec_cell_ids = Vector{eltype(cell_ids_d)}(undef,ncells_d)

#     orders = tfill(order,Val(d))
#     cache = zeros(eltype(eltype(cell_ids_d)),orders.+1)

#     for iprev = 1:ncells_prev
#       cell_prev = cell_ids_prev[iprev]
#       for id = eachindex(cell_ids_d)
#         for idof in CartesianIndices(orders.+1)
#           tidof = Tuple(idof)
#           cache[idof] = cell_prev[tidof[d-1]] + (tidof[d]-1)*ncells_prev
#         end
#         i = (id-1)*ncells_prev+iprev
#         vec_cell_ids[i] = vec(copy(cache))
#       end
#     end
#     _my_recursive_fun(vec_cell_ids,Val(d+1),Val(d′-1))
#   end
#   function _my_recursive_fun(cell_ids,::Val{d},::Val{0}) where d
#     @assert d == D+1
#     return cell_ids
#   end
#   return _my_recursive_fun(cell_ids,Val(1),Val(D))
# end

# cellids = get_cell_dof_ids(test1d)
# spaces = (test1d,test1d)
# order = 2
# D = 2
# diocan = my_recursive_fun(cellids,spaces,order,D)


# d = 2
# cell_ids_prev = get_cell_dof_ids(test1d)
# order = 2

# ncells_prev = length(cell_ids_prev)
# space_d = test1d
# ndofs_prev = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
# cell_ids_d = get_cell_dof_ids(space_d)
# ncells_d = ncells_prev*length(cell_ids_d)
# vec_cell_ids_d = Vector{eltype(cell_ids_d)}(undef,ncells_d)

# orders = tfill(order,Val(d))
# cache = zeros(eltype(eltype(cell_ids_d)),orders.+1)

# for iprev = 1:ncells_prev
#   cell_prev = cell_ids_prev[iprev]
#   for id = eachindex(cell_ids_d)
#     for idof in CartesianIndices(orders.+1)
#       tidof = Tuple(idof)
#       cache[idof] = cell_prev[tidof[d-1]] + (tidof[d]-1)*ndofs_prev
#     end
#     i = (id-1)*ncells_prev+iprev
#     vec_cell_ids_d[i] = vec(copy(cache))
#   end
# end

# new attempt
d = 2
cell_ids_prev = get_cell_dof_ids(test1d)
order = 2

space_d = test1d
cell_ids_prev = get_cell_dof_ids(space_d)
cell_ids_d = get_cell_dof_ids(space_d)

ndofs_d = num_free_dofs(space_d)
ndofs_prev = num_free_dofs(space_d)
ncells_prev = length(cell_ids_prev)

# inner reorder
dof_permutations_1d = TProduct._get_dof_permutation(model1d,cell_ids_prev,order)

# add a (reordered) dimension
add_dim = ndofs_prev .* collect(0:ndofs_d)
add_dim_reorder = add_dim[dof_permutations_1d]
temp = dof_permutations_1d .+ add_dim_reorder'

# NEW RECURSION

function my_recursive_fun(models,spaces,ord,D)
  function _tensor_product(aprev::AbstractArray{Tp,M},a::AbstractVector{Td}) where {Tp,Td,M}
    T = promote_type(Tp,Td)
    N = M+1
    s = (size(aprev)...,length(a))
    atp = zeros(T,s)
    slicesN = eachslice(atp,dims=N)
    @inbounds for (iN,sliceN) in enumerate(slicesN)
      sliceN .= aprev .+ a[iN]
    end
    return atp
  end
  function _my_recursive_fun(::Val{1},::Val{d′}) where d′
    @assert d′ == D
    model_d = models[1]
    space_d = spaces[1]
    ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
    ndofs = ndofs_d
    cell_ids_d = get_cell_dof_ids(space_d)
    dof_permutations_1d = TProduct._get_dof_permutation(model_d,cell_ids_d,ord)
    return _my_recursive_fun(dof_permutations_1d,ndofs,Val(2),Val(d′-1))
  end
  function _my_recursive_fun(node2dof_prev,ndofs_prev,::Val{d},::Val{d′}) where {d,d′}
    model_d = models[d]
    space_d = spaces[d]
    ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
    ndofs = ndofs_prev*ndofs_d
    cell_ids_d = get_cell_dof_ids(space_d)

    dof_permutations_1d = TProduct._get_dof_permutation(model_d,cell_ids_d,ord)

    add_dim = ndofs_prev .* collect(0:ndofs_d)
    add_dim_reorder = add_dim[dof_permutations_1d]
    node2dof_d = _tensor_product(node2dof_prev,add_dim_reorder)

    _my_recursive_fun(node2dof_d,ndofs,Val(d+1),Val(d′-1))
  end
  function _my_recursive_fun(node2dof,ndofs,::Val{d},::Val{0}) where d
    @assert d == D+1
    return node2dof
  end
  return _my_recursive_fun(Val(1),Val(D))
end

my_recursive_fun((model1d,model1d,model1d),(test1d,test1d,test1d),2,3)
