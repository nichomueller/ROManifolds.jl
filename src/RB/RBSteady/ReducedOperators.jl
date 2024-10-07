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
  reduced_operator(solver,feop,red_trial,red_test,s)
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

function ParamSteady.update_paramcache!(
  opcache,
  op::RBOperator,
  r::Realization)

  msg = "The cache can be correctly initialized before the call to solve!"
  @notimplemented msg
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
ParamSteady.get_vector_index_map(op::GenericRBOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::GenericRBOperator) = get_matrix_index_map(op.op)
get_fe_trial(op::GenericRBOperator) = get_trial(op.op)
get_fe_test(op::GenericRBOperator) = get_test(op.op)

function ParamSteady.allocate_paramcache(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
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
  cache,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  b,b̂ = cache
  fe_sb = fe_residual!(b,op,r,u,paramcache)
  inv_project!(b̂,op.rhs,fe_sb)
end

function Algebra.jacobian!(
  cache,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  A,Â = cache
  fe_sA = fe_jacobian!(A,op,r,u,paramcache)
  inv_project!(Â,op.lhs,fe_sA)
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
  join_operators(get_fe_operator(op.op_linear),get_fe_operator(op.op_nonlinear))
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

  (rhs_fe_lin,rhs_rb_lin) = allocate_residual(op.op_linear,r,u,paramcache)
  (rhs_fe_nlin,rhs_rb_nlin) = allocate_residual(op.op_nonlinear,r,u,paramcache)
  _,rhs_hypred_lin = rhs_rb_lin
  _,rhs_hypred_nlin = rhs_rb_nlin
  rhs_hypred_lin_nlin = rhs_hypred_lin + rhs_hypred_nlin
  return (rhs_fe_lin,rhs_rb_lin),(rhs_fe_nlin,rhs_rb_nlin),rhs_hypred_lin_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  (lhs_fe_lin,lhs_rb_lin) = allocate_residual(op.op_linear,r,u,paramcache)
  (lhs_fe_nlin,lhs_rb_nlin) = allocate_residual(op.op_nonlinear,r,u,paramcache)
  _,lhs_hypred_lin = lhs_rb_lin
  _,lhs_hypred_nlin = lhs_rb_nlin
  lhs_hypred_lin_nlin = lhs_hypred_lin + lhs_hypred_nlin
  return (lhs_fe_lin,lhs_rb_lin),(lhs_fe_nlin,lhs_rb_nlin),lhs_hypred_lin_nlin
end

function Algebra.residual!(
  cache,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache;
  kwargs...)

  (_,b̂_lin),(b_nlin,b̂_nlin),b̂_lin_nlin = cache
  feb_nlin = fe_residual!(b_nlin,op.op_nonlinear,r,u,paramcache)
  b̂_nlin = inv_project!(op.op_nonlinear.rhs,feb_nlin)
  @. b̂_lin_nlin = b̂_nlin + b̂_lin
  return b̂_lin_nlin
end

function Algebra.jacobian!(
  A::Tuple,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  (_,Â_lin),(A_nlin,Â_nlin),Â_lin_nlin = cache
  feA_nlin = fe_jacobian!(A_nlin,op.op_nonlinear,r,u,paramcache)
  Â_nlin = inv_project!(op.op_nonlinear.lhs,feA_nlin)
  @. Â_lin_nlin = Â_nlin + Â_lin
  return Â_lin_nlin
end

# Solve a POD-MDEIM problem

function init_online_cache!(solver::RBSolver,op::RBOperator,r::Realization,y::AbstractParamVector)
  @check param_length(r) == param_length(y)

  fesolver = get_fe_solver(solver)
  paramcache = allocate_paramcache(fesolver,op,r)
  rbcache = allocate_rbcache(op,r)

  cache = solver.cache
  cache.fecache = (y,paramcache)
  cache.rbcache = rbcache
  return
end

function online_cache!(solver::RBSolver,op::RBOperator,r::Realization)
  cache = solver.cache
  (y,paramcache) = cache.fecache
  if param_length(r) != param_length(y)
    y′ = array_of_consecutive_arrays(testitem(y),param_length(r))
    init_online_cache!(solver,op,r,y′)
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
  r::AbstractRealization)

  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  x̂ = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  cache = x̂,y
  solve!(cache,solver,op,r)
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::RBOperator,
  r::AbstractRealization)

  x̂,y = cache
  fesolver = get_fe_solver(solver)

  t = @timed solve!((x̂,),fesolver,op,r,(y,))
  stats = CostTracker(t,nruns=num_params(r))

  trial = get_trial(op)(r)
  x = inv_project(trial,x̂)

  return x,stats
end

# cache utils

function select_fe_space_at_indices(fs::FESpace,indices)
  @notimplemented
end

function select_fe_space_at_indices(fs::TrivialParamFESpace,indices)
  TrivialParamFESpace(fs.space,Val(length(indices)))
end

function select_fe_space_at_indices(fs::SingleFieldParamFESpace,indices)
  dvi = ConsecutiveArrayOfArrays(fs.dirichlet_values.data[:,indices])
  TrialParamFESpace(dvi,fs.space)
end

function select_evalcache_at_indices(u::ConsecutiveArrayOfArrays,paramcache,indices)
  @unpack Us,Ups,pfeopcache,form = paramcache
  new_Us = select_fe_space_at_indices(Us,indices)
  new_XhF = ConsecutiveArrayOfArrays(u.data[:,indices])
  new_paramcache = ParamOpFromFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_paramcache
end

function select_evalcache_at_indices(u::BlockVectorOfVectors,paramcache,indices)
  @unpack Us,Ups,pfeopcache,form = paramcache
  VT = Us.vector_type
  style = Us.multi_field_style
  spaces = select_fe_space_at_indices(Us,indices)
  new_Us = MultiFieldFESpace(VT,spaces,style)
  new_XhF = mortar([ConsecutiveArrayOfArrays(b.data[:,indices]) for b in blocks(u)])
  new_paramcache = ParamOpFromFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_paramcache
end

function select_slvrcache_at_indices(b::ConsecutiveArrayOfArrays,indices)
  ConsecutiveArrayOfArrays(b.data[:,indices])
end

function select_slvrcache_at_indices(A::MatrixOfSparseMatricesCSC,indices)
  MatrixOfSparseMatricesCSC(A.m,A.n,A.colptr,A.rowval,A.data[:,indices])
end

function select_slvrcache_at_indices(A::BlockArrayOfArrays,indices)
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
  s::Union{BlockArray,BlockArrayOfArrays},
  a::BlockHyperReduction,
  args...)

  @check size(blocks(s)) == size(a)
  @notimplementedif isempty(findall(a.touched))
  i = findfirst(a.touched)
  cache = return_cache(select_at_indices,blocks(s)[i],a[i],args...)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function select_at_indices(s::Union{BlockArray,BlockArrayOfArrays},a::BlockHyperReduction,args...)
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
