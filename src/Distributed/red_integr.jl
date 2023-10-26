root = pwd()
include("$root/src/Utils/Files.jl")
include("$root/src/FEM/FEM.jl")

domain = (0,1,0,1)
part = (4,4)
order = 1
degree = 2

model = CartesianDiscreteModel(domain,part)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,7,8])
add_tag_from_tags!(labels,"neumann",[6,])

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

ranges = fill([1.,2.],3)
sampling = UniformSampling()
pspace = PSpace(ranges,sampling)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂ₚt(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(a(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

res(μ,t,u,v) = res(μ,t,u,v,dΩ,dΓn)
jac(μ,t,u,du,v) = jac(μ,t,u,du,v,dΩ)
jac_t(μ,t,u,dut,v) = jac_t(μ,t,u,dut,v,dΩ)

reffe = ReferenceFE(lagrangian,Float,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = PTransientTrialFESpace(test,g)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.05,0.005,1
uh0(μ) = interpolate_everywhere(u0(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),t0,tf,dt,θ,uh0)

using PartitionedArrays
parts = (2,2)
ranks = DebugArray(LinearIndices((prod(parts),)))
Dmodel = CartesianDiscreteModel(ranks,parts,domain,part)
Dlabels = get_face_labeling(Dmodel)
add_tag_from_tags!(Dlabels,"dirichlet",[1,2,3,4,5,7,8])
add_tag_from_tags!(Dlabels,"neumann",[6,])
Dtest = FESpace(Dmodel,reffe,conformity=:H1,dirichlet_tags="dirichlet")
Dtrial = PTTrialFESpace(Dtest,g)
DΩ = Triangulation(Dmodel)
DdΩ = Measure(DΩ,degree)
DΓn = BoundaryTriangulation(Dmodel,tags=["neumann"])
DdΓn = Measure(DΓn,degree)
Dres(μ,t,u,v) = ∫(v*∂ₚt(u))DdΩ + ∫(a(μ,t)*∇(v)⋅∇(u))DdΩ - ∫(f(μ,t)*v)DdΩ - ∫(h(μ,t)*v)DdΓn
Djac(μ,t,u,du,v) = ∫(a(μ,t)*∇(v)⋅∇(du))DdΩ
Djac_t(μ,t,u,dut,v) = ∫(v*dut)DdΩ
Dfeop = AffinePTFEOperator(Dres,Djac,Djac_t,pspace,Dtrial,Dtest)
Duh0(μ) = interpolate_everywhere(u0(μ),Dtrial(μ,t0))
Dfesolver = ThetaMethod(LUSolver(),t0,tf,dt,θ,Duh0)

function GridapDistributed.allocate_jacobian(
  op::PTFEOperatorFromWeakForm,
  duh::GridapDistributed.DistributedCellField,
  cache)
  _matdata_jacobians = Gridap.ODEs.TransientFETools.fill_initial_jacobians(op,duh)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  allocate_matrix(op.assem,matdata)
end

function GridapDistributed.allocate_jacobian(
  op::PTFEOperatorFromWeakForm,
  duh::GridapDistributed.DistributedMultiFieldFEFunction,
  cache)
  _matdata_jacobians = Gridap.ODEs.TransientFETools.fill_initial_jacobians(op,duh)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  allocate_matrix(op.assem,matdata)
end

function Gridap.ODEs.ODETools.jacobians!(
  A::AbstractMatrix,
  op::Gridap.ODEs.TransientFETools.TransientFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::Real,
  xh::GridapDistributed.TransientDistributedCellField,
  γ::Tuple{Vararg{Real}},
  cache)
  _matdata_jacobians = Gridap.ODEs.TransientFETools.fill_jacobians(op,μ,t,xh,γ)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function collect_snapshots(
  ::typeof(assemble_vector),
  op::PTFEOperator{Affine},
  solver::ODESolver,
  trian::GridapType,
  pinfo::Vector{<:AbstractArray},
  tinfo::Vector{<:Real})

  μt = Iterators.product(tinfo,pinfo)

  pop = get_algebraic_operator(op)
  dv = get_fe_basis(op.test)
  cache = allocate_cache(pop)

  function _vector(μt)
    t,μ = μt
    update_cache!(cache,pop,μ,t)
    x0 = _setup_initial_condition(pop,solver,μ)
    xh = _evaluation_function(op,(x0,x0),cache)
    vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv),trian)

    r = allocate_residual(pop,x0,cache)
    assemble_vector_add!(r,op.assem,vecdata)
  end

  # lazy_map(_vector,[μt...])
  map(_vector,[μt...])
end

function collect_snapshots(
  ::typeof(assemble_matrix),
  op::PTFEOperator{Affine},
  solver::ODESolver,
  trian::GridapType,
  pinfo::Vector{<:AbstractArray},
  tinfo::Vector{<:Real};
  i::Int=1)

  T = SparseMatrixCSC{Float64,Int}
  μt = Iterators.product(tinfo,pinfo)

  pop = get_algebraic_operator(op)
  dv = get_fe_basis(op.test)
  trial = get_trial(op)
  trial_hom = allocate_trial_space(trial)
  du = get_trial_fe_basis(trial_hom)
  cache = allocate_cache(pop)
  γ = (1.0,1/(solver.dt*solver.θ))

  function _matrix(μt)
    t,μ = μt
    update_cache!(cache,pop,μ,t)
    trial, = cache[1]
    x0 = _setup_initial_condition(pop,solver,μ)
    xh = _evaluation_function(op,(x0,x0),cache)
    matdata = collect_cell_matrix(trial,op.test,γ[i]*op.jacs[i](μ,t,xh,du,dv),trian)

    J = allocate_jacobian(pop,x0,cache)
    assemble_matrix_add!(J,op.assem,matdata)
  end

  lazy_map(_matrix,T,[μt...])
end

function _setup_initial_condition(
  op::PTFEOperator,
  solver::ODESolver,
  μ::AbstractArray=realization(op))

  solver.uh0(μ)
end

function _setup_initial_condition(
  pop::PODEOpFromFEOp,
  solver::ODESolver,
  μ::AbstractArray=realization(pop.feop))

  uh0 = _setup_initial_condition(pop.feop,solver,μ)
  get_free_dof_values(uh0)
end

function _evaluation_function(
  op::PTFEOperator,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
end

function Gridap.FESpaces.collect_cell_vector(
  test::GridapDistributed.DistributedFESpace,
  a::GridapDistributed.DistributedDomainContribution,
  trian::GridapDistributed.DistributedTriangulation)

  map(collect_cell_vector,local_views(test),local_views(a),trian.trians)
end

function Gridap.FESpaces.collect_cell_matrix(
  trial::GridapDistributed.DistributedFESpace,
  test::GridapDistributed.DistributedFESpace,
  a::GridapDistributed.DistributedDomainContribution,
  trian::GridapDistributed.DistributedTriangulation)

  map(collect_cell_matrix,local_views(trial),local_views(test),local_views(a),trian.trians)
end

_create_from_nz(x) = create_from_nz(x)

_create_from_nz(x::Vector{T}) where {T<:AbstractArray} = map(_create_from_nz,x)

pinfo = [rand(3) for _ = 1:10]
tinfo = get_times(fesolver)
res_vec = collect_snapshots(assemble_vector,feop,fesolver,Ω,pinfo,tinfo)
res_Dvec = collect_snapshots(assemble_vector,Dfeop,Dfesolver,DΩ,pinfo,tinfo)

using SparseArrays
res_mat = collect_snapshots(assemble_matrix,feop,fesolver,Ω,pinfo,tinfo)
res_Dmat = collect_snapshots(assemble_matrix,Dfeop,Dfesolver,DΩ,pinfo,tinfo)

map(local_views(res_Dmat[1])) do res_Dmat_1p
  println(typeof(res_Dmat_1p))
end

# SVD
map(local_views(res_Dmat[1])) do res_Dmat_1p
  println(typeof(res_Dmat_1p))
end

function _my_collecting_function(a)
  t = ()
  for ela = a
    t = (t...,ela)
  end
  t
end
