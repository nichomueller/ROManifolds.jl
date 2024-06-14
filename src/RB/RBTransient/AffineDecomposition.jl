# OFFLINE PHASE

get_indices_time(i::AbstractIntegrationDomain) = @abstractmethod
union_indices_time(i::AbstractIntegrationDomain...) = union(map(get_indices_time,i)...)

struct TransientIntegrationDomain{S<:AbstractVector,T<:AbstractVector} <: AbstractIntegrationDomain
  indices_space::S
  indices_time::T
end

get_indices_space(i::TransientIntegrationDomain) = i.indices_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function RBSteady.allocate_coefficient(
  solver::RBSolver{S,SpaceOnlyMDEIM} where S,
  b::TransientProjection)

  nspace = num_reduced_space_dofs(b)
  ntime = num_times(b)
  nparams = num_online_params(solver)
  coeffmat = allocate_matrix(Vector{Float64},nspace,ntime)
  coeff = array_of_similar_arrays(coeffmat,nparams)
  return coeff
end

const TransientAffineDecomposition{A,B,C<:TransientIntegrationDomain,D,E} = AffineDecomposition{A,B,C,D,E}

get_indices_time(a::TransientAffineDecomposition) = get_indices_time(get_integration_domain(a))

function _time_indices_and_interp_matrix(::SpaceTimeMDEIM,interp_basis_space,basis_time)
  indices_time = get_mdeim_indices(basis_time)
  interp_basis_time = view(basis_time,indices_time,:)
  interp_basis_space_time = kronecker(interp_basis_time,interp_basis_space)
  lu_interp = lu(interp_basis_space_time)
  return indices_time,lu_interp
end

function _time_indices_and_interp_matrix(::SpaceOnlyMDEIM,interp_basis_space,basis_time)
  indices_time = axes(basis_time,1)
  lu_interp = lu(interp_basis_space)
  return indices_time,lu_interp
end

function RBSteady.mdeim(mdeim_style::MDEIMStyle,b::TransientPODBasis)
  basis_space = get_basis_space(b)
  basis_time = get_basis_time(b)
  indices_space = get_mdeim_indices(basis_space)
  recast_indices_space = recast_indices(indices_space,basis_space)
  interp_basis_space = view(basis_space,indices_space,:)
  indices_time,lu_interp = _time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
  integration_domain = TransientIntegrationDomain(recast_indices_space,indices_time)
  return lu_interp,integration_domain
end

function RBSteady.mdeim(mdeim_style::MDEIMStyle,b::TransientTTSVDCores)
  basis_spacetime = get_basis_spacetime(b)
  indices_spacetime = get_mdeim_indices(basis_spacetime)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(view(basis_spacetime,indices_spacetime,:))
  integration_domain = TransientIntegrationDomain(indices_space,indices_time)
  return lu_interp,integration_domain
end

const TransientAffineContribution{A<:TransientAffineDecomposition,V,K} = AffineContribution{A,V,K}

function union_reduced_times(a::TransientAffineContribution)
  idom = ()
  for values in get_values(a)
    idom = (idom...,get_integration_domain(values))
  end
  union_indices_time(idom...)
end

function union_reduced_times(a::NTuple)
  union([union_reduced_times(ai) for ai in a]...)
end

function RBSteady.reduced_jacobian(
  solver::ThetaMethodRBSolver,
  op::RBOperator,
  contribs::Tuple{Vararg{Any}})

  fesolver = get_fe_solver(solver)
  θ = fesolver.θ
  a = ()
  for (i,c) in enumerate(contribs)
    combine = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*(x-y)
    a = (a...,reduced_jacobian(solver,op,c;combine))
  end
  return a
end

# ONLINE PHASE

function RBSteady.coefficient!(
  a::TransientAffineDecomposition{<:ReducedAlgebraicOperator{T}},
  b::AbstractParamArray
  ) where T<:SpaceTimeMDEIM

  coefficient = a.coefficient
  mdeim_interpolation = a.mdeim_interpolation
  ldiv!(coefficient,mdeim_interpolation,vec(b))
end

function RBSteady.mdeim_result(a::TupOfArrayContribution,b::TupOfArrayContribution)
  sum(map(mdeim_result,a,b))
end

# multi field interface

const BlockTransientAffineDecomposition{A<:TransientAffineDecomposition,N,C} = BlockAffineDecomposition{A,N,C}

function RBSteady.get_integration_domain(a::BlockTransientAffineDecomposition)
  active_block_ids = get_touched_blocks(a)
  block_indices_space = get_indices_space(a)
  union_indices_space = union([block_indices_space[i] for i in active_block_ids]...)
  union_indices_time = get_indices_time(a)
  TransientIntegrationDomain(union_indices_space,union_indices_time)
end

function get_indices_time(a::BlockTransientAffineDecomposition)
  indices_time = Any[get_indices_time(a[i]) for i = get_touched_blocks(a)]
  union(indices_time...)
end

function ParamDataStructures.num_times(a::BlockTransientAffineDecomposition)
  num_times(testitem(a))
end

function num_reduced_times(a::BlockTransientAffineDecomposition)
  length(get_indices_time(a))
end
