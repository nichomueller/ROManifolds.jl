# OFFLINE PHASE

get_indices_time(i::AbstractIntegrationDomain) = @abstractmethod
union_indices_time(i::AbstractIntegrationDomain...) = union(map(get_indices_time,i)...)

"""
"""
struct TransientIntegrationDomain{S<:AbstractVector,T<:AbstractVector} <: AbstractIntegrationDomain
  indices_space::S
  indices_time::T
end

RBSteady.get_indices_space(i::TransientIntegrationDomain) = i.indices_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function RBSteady.allocate_coefficient(
  solver::RBSolver{S,SpaceOnlyMDEIM} where S,
  b::TransientProjection)

  nspace = RBSteady.num_reduced_space_dofs(b)
  ntime = num_times(b)
  nparams = RBSteady.num_online_params(solver)
  coeffmat = allocate_matrix(Vector{Float64},nspace,ntime)
  coeff = array_of_consecutive_arrays(coeffmat,nparams)
  return coeff
end

const TransientAffineDecomposition{A,B,C<:TransientIntegrationDomain,D,E} = AffineDecomposition{A,B,C,D,E}

const TupOfAffineContribution = Tuple{Vararg{AffineContribution{T}}} where T

get_indices_time(a::TransientAffineDecomposition) = get_indices_time(get_integration_domain(a))

function _time_indices_and_interp_matrix(::SpaceTimeMDEIM,interp_basis_space,basis_time)
  indices_time,interp_basis_time = empirical_interpolation(basis_time)
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
  indices_space,interp_basis_space = empirical_interpolation(basis_space)
  indices_time,lu_interp = _time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
  integration_domain = TransientIntegrationDomain(indices_space,indices_time)
  return lu_interp,integration_domain
end

function RBSteady.mdeim(mdeim_style::MDEIMStyle,b::TransientTTSVDCores)
  basis_spacetime = get_basis_spacetime(b)
  indices_spacetime,interp_basis_spacetime = empirical_interpolation(basis_spacetime)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(interp_basis_spacetime)
  integration_domain = TransientIntegrationDomain(indices_space,indices_time)
  return lu_interp,integration_domain
end

function union_reduced_times(a::AffineContribution)
  idom = ()
  for values in get_values(a)
    idom = (idom...,get_integration_domain(values))
  end
  union_indices_time(idom...)
end

function union_reduced_times(a::TupOfAffineContribution)
  union(union_reduced_times.(a)...)
end

function RBSteady.reduced_jacobian(
  solver::ThetaMethodRBSolver,
  op::TransientRBOperator,
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

function RBSteady.mdeim_result(a::TupOfAffineContribution,b::TupOfArrayContribution)
  sum(map(mdeim_result,a,b))
end

# multi field interface

function union_indices_time(i::ArrayBlock{<:TransientIntegrationDomain}...)
  times = map(i) do i
    union_indices_time(i.array[findall(i.touched)]...)
  end
  union(times...)
end

function get_indices_time(a::BlockAffineDecomposition{<:TransientAffineDecomposition})
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  blocks = [get_indices_time(a[i]) for i in active_block_ids]
  return_cache(block_map,blocks...)
end

# for testing/visualization purposes

function RBSteady.project(A::AbstractMatrix,r::TransientRBSpace)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)

  a = (basis_space'*A)*basis_time
  v = vec(a)
  return v
end

function RBSteady.project(A::ParamSparseMatrix,trial::TransientPODBasis,test::TransientPODBasis;combine=(x,y)->x)
  function compress_basis_space(A,B,C)
    map(param_data(A)) do a
      C'*a*B
    end
  end
  basis_space_test = get_basis_space(test)
  basis_time_test = get_basis_time(test)
  basis_space_trial = get_basis_space(trial)
  basis_time_trial = get_basis_time(trial)
  ns_test,ns_trial = size(basis_space_test,2),size(basis_space_trial,2)
  nt_test,nt_trial = size(basis_time_test,2),size(basis_time_trial,2)

  red_xvec = compress_basis_space(A,get_basis_space(trial),get_basis_space(test))
  a = stack(vec.(red_xvec))'  # Nt x ns_test*ns_trial
  st_proj = zeros(eltype(A),nt_test,nt_trial,ns_test*ns_trial)
  st_proj_shift = zeros(eltype(A),nt_test,nt_trial,ns_test*ns_trial)
  @inbounds for ins = 1:ns_test*ns_trial, jt = 1:nt_trial, it = 1:nt_test
    st_proj[it,jt,ins] = sum(basis_time_test[:,it].*basis_time_trial[:,jt].*a[:,ins])
    st_proj_shift[it,jt,ins] = sum(basis_time_test[2:end,it].*basis_time_trial[1:end-1,jt].*a[2:end,ins])
  end
  st_proj = combine(st_proj,st_proj_shift)
  st_proj_a = zeros(T,ns_test*nt_test,ns_trial*nt_trial)
  @inbounds for i = 1:ns_trial, j = 1:ns_test
    st_proj_a[j:ns_test:ns_test*nt_test,i:ns_trial:ns_trial*nt_trial] = st_proj[:,:,(i-1)*ns_test+j]
  end
  return st_proj_a
end

function RBSteady.project(fesolver,fes::TupOfAffineContribution,trial,test)
  cmp = ()
  for i = eachindex(fes)
    combine = (x,y) -> i == 1 ? fesolver.θ*x+(1-fesolver.θ)*y : fesolver.θ*(x-y)
    cmp = (cmp...,RBSteady.project(fesolver,fes[i],trial,test;combine))
  end
  sum(cmp)
end

function RBSteady.interpolation_error(
  a::AffineDecomposition,
  fes::AbstractTransientSnapshots,
  rbs::AbstractTransientSnapshots)

  ids_space,ids_time = RBSteady.get_indices_space(a),get_indices_time(a)
  fes_ids = select_snapshots_entries(fes,ids_space,ids_time)
  rbs_ids = select_snapshots_entries(rbs,ids_space,ids_time)
  norm(fes_ids - rbs_ids)
end

function RBSteady.interpolation_error(a::Tuple,fes::Tuple,rbs::Tuple)
  @check length(a) == length(fes) == length(rbs)
  err = ()
  for i = eachindex(a)
    err = (err...,RBSteady.interpolation_error(a[i],fes[i],rbs[i]))
  end
  err
end

function RBSteady.interpolation_error(solver,feop::LinearNonlinearTransientParamFEOperator,rbop,s)
  err_lin = RBSteady.interpolation_error(solver,feop.op_linear,rbop.op_linear,s;name="linear")
  err_nlin = RBSteady.interpolation_error(solver,feop.op_nonlinear,rbop.op_nonlinear,s;name="non linear")
  return err_lin,err_nlin
end

function RBSteady.linear_combination_error(solver,feop::LinearNonlinearTransientParamFEOperator,rbop,s)
  err_lin = RBSteady.linear_combination_error(solver,feop.op_linear,rbop.op_linear,s;name="linear")
  err_nlin = RBSteady.linear_combination_error(solver,feop.op_nonlinear,rbop.op_nonlinear,s;name="non linear")
  return err_lin,err_nlin
end
