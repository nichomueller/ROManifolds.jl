function reduced_operator(
  solver::RBSolver,
  feop::ParamFEOperator,
  args...;
  dir=datadir(),
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  save(fesnaps,dir)
  save(festats,dir;label="fe")
  reduced_operator(solver,feop,fesnaps)
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamFEOperator,
  s)

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  op = get_algebraic_operator(feop)
  reduced_operator(solver,op,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::FESubspace,
  red_test::FESubspace,
  s)

  red_lhs,red_rhs = reduced_weak_form(solver,op,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  GenericRBOperator(new_op,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial::FESubspace,
  red_test::FESubspace,
  s)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

abstract type RBOperator{T} <: ParamOperatorWithTrian{T} end

function allocate_rbcache(
  op::RBOperator,
  r::Realization)

  lhs_cache = allocate_jacobian(op,r)
  rhs_cache = allocate_residual(op,r)
  return lhs_cache,rhs_cache
end

struct GenericRBOperator{T} <: RBOperator{T}
  op::ParamOperatorWithTrian{T}
  trial::FESubspace
  test::FESubspace
  lhs::AffineContribution
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
ParamDataStructures.realization(op::GenericRBOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::GenericRBOperator) = ParamSteady.get_fe_operator(op.op)
IndexMaps.get_vector_index_map(op::GenericRBOperator) = get_vector_index_map(op.op)
IndexMaps.get_matrix_index_map(op::GenericRBOperator) = get_matrix_index_map(op.op)
get_fe_trial(op::GenericRBOperator) = get_trial(op.op)
get_fe_test(op::GenericRBOperator) = get_test(op.op)

function ParamSteady.allocate_paramcache(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::GenericRBOperator,
  r::Realization)

  update_paramcache!(paramcache,op.op,r)
end

function Algebra.allocate_residual(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  rhs_fe = allocate_jacobian(op.op,r,u,paramcache)
  rhs_rb = allocate_hypred_cache(op.rhs,r)
  return (rhs_fe,rhs_rb)
end

function Algebra.allocate_jacobian(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  lhs_fe = allocate_jacobian(op.op,r,u,paramcache)
  lhs_rb = allocate_hypred_cache(op.lhs,r)
  return (lhs_fe,lhs_rb)
end

function Algebra.residual!(
  b̂,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  b = paramcache.b
  feb = fe_residual!(b,op,r,u,paramcache)
  inv_project!(b̂,op.rhs,feb)
end

function Algebra.jacobian!(
  Â,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  A = paramcache.A
  feA = fe_jacobian!(A,op,r,u,paramcache)
  inv_project!(Â,op.lhs,feA)
end

function fe_jacobian!(
  A,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  jacobian!(A,op.op,r,u,paramcache)
  Ai = select_at_indices(A,op.lhs)
  return Ai
end

function fe_residual!(
  b,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  residual!(b,op.op,r,u,paramcache)
  bi = select_at_indices(b,op.rhs)
  return bi
end

function allocate_rbcache(op::GenericRBOperator,r::Realization)
  lhs_cache = allocate_hypred_cache(op.lhs,r)
  rhs_cache = allocate_hypred_cache(op.rhs,r)
  return lhs_cache,rhs_cache
end

"""
    struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq} end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications

"""
struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq}
  op_linear::GenericRBOperator{<:LinearParamEq}
  op_nonlinear::GenericRBOperator{NonlinearParamEq}
end

ParamSteady.get_linear_operator(op::LinearNonlinearRBOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearRBOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearRBOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearRBOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearRBOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function ParamSteady.get_fe_operator(op::LinearNonlinearRBOperator)
  join_operators(ParamSteady.get_fe_operator(op.op_linear),ParamSteady.get_fe_operator(op.op_nonlinear))
end

function IndexMaps.get_vector_index_map(op::LinearNonlinearRBOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  get_vector_index_map(op.op_linear)
end

function IndexMaps.get_matrix_index_map(op::LinearNonlinearRBOperator)
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_matrix_index_map(op.op_linear)
end

function get_fe_trial(op::LinearNonlinearRBOperator)
  @check get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  get_fe_trial(op.op_nonlinear)
end

function get_fe_test(op::LinearNonlinearRBOperator)
  @check get_fe_test(op.op_linear) === get_fe_test(op.op_nonlinear)
  get_fe_test(op.op_nonlinear)
end

function Algebra.allocate_residual(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  @notimplemented
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  @notimplemented
end

function Algebra.residual!(
  cache,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache;
  kwargs...)

  @notimplemented
end

function Algebra.jacobian!(
  A::Tuple,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  @notimplemented
end

function ParamSteady.allocate_paramcache(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector)

  paramcache_lin = allocate_paramcache(get_linear_operator(op),r,u)
  paramcache_nlin = allocate_paramcache(get_nonlinear_operator(op),r,u)
  return (paramcache_lin,paramcache_nlin)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::LinearNonlinearRBOperator,
  r::Realization)

  paramcache_lin,paramcache_nlin = paramcache
  paramcache_lin = ParamSteady.update_paramcache!(paramcache_lin,get_linear_operator(op),r)
  paramcache_nlin = ParamSteady.update_paramcache!(paramcache_nlin,get_nonlinear_operator(op),r)
  return (paramcache_lin,paramcache_nlin)
end

function allocate_rbcache(op::LinearNonlinearRBOperator,r::Realization)
  cache_lin = allocate_rbcache(get_linear_operator(op),r)
  cache_nlin = allocate_rbcache(get_nonlinear_operator(op),r)
  return (cache_lin,cache_nlin)
end

# Solve a POD-MDEIM problem

function init_online_cache!(solver::RBSolver,op::RBOperator,r::Realization)
  fe_trial = get_fe_trial(op)(r)
  trial = get_trial(op)(r)
  y = zero_free_values(fe_trial)
  x̂ = zero_free_values(trial)

  paramcache = allocate_paramcache(op,r,y)
  rbcache = allocate_rbcache(op,r)

  cache = solver.cache
  cache.fecache = (y,paramcache)
  cache.rbcache = (x̂,rbcache)
  return
end

function online_cache!(solver::RBSolver,op::RBOperator,r::Realization)
  cache = solver.cache
  y,paramcache = cache.fecache
  if param_length(r) != param_length(y)
    init_online_cache!(solver,op,r)
  else
    paramcache = update_paramcache!(paramcache,op,r)
    cache.fecache = y,paramcache
  end
  return
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator{NonlinearParamEq},
  r::AbstractRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator,
  r::AbstractRealization;
  kwargs...)

  cache = solver.cache
  if isnothing(cache.fecache) || isnothing(cache.rbcache)
    RBSteady.init_online_cache!(solver,op,r)
  else
    RBSteady.online_cache!(solver,op,r)
  end
  solve!(cache,solver,op,r;kwargs...)
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::RBOperator,
  r::AbstractRealization;
  kwargs...)

  y,paramcache = cache.fecache
  x̂,rbcache = cache.rbcache

  t = @timed solve!(x̂,solver,op,r,y,paramcache,rbcache)
  stats = CostTracker(t,nruns=num_params(r))

  return x̂,stats
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::RBSolver,
  op::RBOperator,
  r::Realization,
  x::AbstractVector,
  paramcache,
  rbcache;
  kwargs...)

  fesolver = get_fe_solver(solver)
  Âcache,b̂cache = rbcache
  Â = jacobian!(Âcache,op,r,x,paramcache)
  b̂ = residual!(b̂cache,op,r,x,paramcache)
  solve!(x̂,fesolver.ls,Â,b̂)
  return x̂
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::RBSolver,
  op::RBOperator{LinearNonlinearParamEq},
  r::Realization,
  x::AbstractVector,
  paramcache,
  rbcache;
  kwargs...)

  fesolver = get_fe_solver(solver)

  # linear + nonlinear cache
  paramcache_lin,paramcache_nlin = paramcache
  rbcache_lin,rbcache_nlin = rbcache

  # linear cache
  Âcache_lin,b̂cache_lin = rbcache_lin
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  Â_lin = jacobian!(Âcache_lin,op_lin,r,x,paramcache_lin)
  b̂_lin = residual!(b̂cache_lin,op_lin,r,x,paramcache_lin)

  # nonlinear cache
  syscache_nlin = rbcache_nlin
  trial = get_trial(op)(r)
  cache = syscache_nlin,trial

  nlop = RBNewtonRaphsonOperator(op_nlin,paramcache_nlin,r,Â_lin,b̂_lin,cache)
  solve!(x̂,fesolver.nls,nlop,r,x;kwargs...)

  return x̂
end

# cache utils

function select_fe_space_at_indices(fs::FESpace,indices)
  @notimplemented
end

function select_fe_space_at_indices(fs::TrivialParamFESpace,indices)
  TrivialParamFESpace(fs.space,Val(length(indices)))
end

function select_fe_space_at_indices(fs::SingleFieldParamFESpace,indices)
  dvi = ConsecutiveParamArray(fs.dirichlet_values.data[:,indices])
  TrialParamFESpace(dvi,fs.space)
end

function select_evalcache_at_indices(u::ConsecutiveParamArray,paramcache,indices)
  @unpack Us,Ups,pfeopcache,form = paramcache
  new_Us = select_fe_space_at_indices(Us,indices)
  new_XhF = ConsecutiveParamArray(u.data[:,indices])
  new_paramcache = ParamCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_paramcache
end

function select_evalcache_at_indices(u::BlockConsecutiveParamVector,paramcache,indices)
  @unpack Us,Ups,pfeopcache,form = paramcache
  VT = Us.vector_type
  style = Us.multi_field_style
  spaces = select_fe_space_at_indices(Us,indices)
  new_Us = MultiFieldFESpace(VT,spaces,style)
  new_XhF = mortar([ConsecutiveParamArray(b.data[:,indices]) for b in blocks(u)])
  new_paramcache = ParamCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_paramcache
end

function select_slvrcache_at_indices(b::ConsecutiveParamArray,indices)
  ConsecutiveParamArray(b.data[:,indices])
end

function select_slvrcache_at_indices(A::ConsecutiveParamSparseMatrixCSC,indices)
  ConsecutiveParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,A.data[:,indices])
end

function select_slvrcache_at_indices(A::BlockParamArray,indices)
  map(a -> select_slvrcache_at_indices(a,indices),blocks(A)) |> mortar
end

function select_slvrcache_at_indices(cache::ArrayContribution,indices)
  contribution(cache.trians) do trian
    select_slvrcache_at_indices(cache[trian],indices)
  end
end

# selects the entries of the snapshots relevant to the reduced integration domain
# in `a`
function select_at_indices(s::AbstractArray,a::HyperReduction)
  s[get_integration_domain(a)]
end

function Arrays.return_cache(::typeof(select_at_indices),s::AbstractArray,a::HyperReduction,args...)
  select_at_indices(s,a,args...)
end

function Arrays.return_cache(
  ::typeof(select_at_indices),
  s::Union{BlockArray,BlockParamArray},
  a::BlockHyperReduction,
  args...)

  @check size(blocks(s)) == size(a)
  @notimplementedif isempty(findall(a.touched))
  i = findfirst(a.touched)
  cache = return_cache(select_at_indices,blocks(s)[i],a[i],args...)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function select_at_indices(s::Union{BlockArray,BlockParamArray},a::BlockHyperReduction,args...)
  s′ = return_cache(select_at_indices,s,a,args...)
  for i = eachindex(a)
    if a.touched[i]
      s′[i] = select_at_indices(blocks(s)[i],a[i],args...)
    end
  end
  return ArrayBlock(s′,a.touched)
end

function select_at_indices(s::ArrayContribution,a::AffineContribution)
  contribution(s.trians) do trian
    select_at_indices(s[trian],a[trian])
  end
end
