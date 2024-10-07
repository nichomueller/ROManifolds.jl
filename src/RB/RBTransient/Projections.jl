function RBSteady.projection(red::TransientAffineReduction,s::AbstractTransientSnapshots,args...)
  s1 = flatten_snapshots(select_snapshots(s,1,1))
  basis_space = projection(get_reduction_space(red),s1,args...)
  basis_time = PODBasis(I[1:num_times(s),1:1])
  TransientBasis(basis_space,basis_time)
end

function RBSteady.projection(red::TransientReduction,s::AbstractTransientSnapshots,args...)
  s1 = flatten_snapshots(s)
  basis_space = projection(get_reduction_space(red),s1,args...)
  proj_s1 = galerkin_projection(get_basis(basis_space),s1,args...)
  proj_s2 = change_mode(proj_s1,num_params(s))
  basis_time = projection(get_reduction_time(red),proj_s2)
  TransientBasis(basis_space,basis_time)
end

"""
"""
struct TransientBasis{A<:Projection,B<:Projection} <: Projection
  basis_space::A
  basis_time::B
end

get_basis_space(a::TransientBasis) = get_basis(a.basis_space)
get_basis_time(a::TransientBasis) = get_basis(a.basis_time)

RBSteady.get_basis(a::TransientBasis) = kron(get_basis_time(a),get_basis_space(a))
RBSteady.num_fe_dofs(a::TransientBasis) = num_fe_dofs(a.basis_space)*num_fe_dofs(a.basis_time)
RBSteady.num_reduced_dofs(a::TransientBasis) = num_reduced_dofs(a.basis_space)*num_reduced_dofs(a.basis_time)

function RBSteady.project(a::TransientBasis,X::AbstractMatrix)
  basis_space = get_basis(a.basis_space)
  basis_time = get_basis(a.basis_time)
  X̂ = basis_space'*X*basis_time
  return X̂
end

function RBSteady.inv_project(a::TransientBasis,X̂::AbstractMatrix)
  basis_space = get_basis(a.basis_space)
  basis_time = get_basis(a.basis_time)
  X = basis_space*X̂*basis_time'
  return X
end

for f in (:(RBSteady.project),:(RBSteady.inv_project))
  @eval begin
    function $f(a::TransientBasis,y::AbstractVector)
      ns = num_reduced_dofs(a.basis_space)
      nt = num_reduced_dofs(a.basis_time)
      Y = reshape(y,ns,nt)
      $f(a,Y)
    end
  end
end

function RBSteady.galerkin_projection(
  proj_left::TransientBasis,
  a::TransientBasis)

  proj_basis_space = galerkin_projection(get_basis_space(proj_left),get_basis_space(a))
  proj_basis_time = galerkin_projection(get_basis_time(proj_left),get_basis_time(a))
  proj_basis = kron(proj_basis_time,proj_basis_space)
  return ReducedProjection(proj_basis)
end

function RBSteady.galerkin_projection(
  proj_left::TransientBasis,
  a::TransientBasis,
  proj_right::TransientBasis)

  @notimplemented "In unsteady problems, we need to provide a combining function"
end

function RBSteady.galerkin_projection(
  proj_left::TransientBasis,
  a::TransientBasis,
  proj_right::TransientBasis,
  combine)

  proj_basis_space = galerkin_projection(
    get_basis_space(proj_left),
    get_basis_space(a),
    get_basis_space(proj_right))

  proj_basis_time = galerkin_projection(
    get_basis_time(proj_left),
    get_basis_time(a),
    get_basis_time(proj_right),
    combine)

  nleft = num_reduced_dofs(proj_left)
  ns = num_reduced_dofs(a.basis_space)
  nt = num_reduced_dofs(a.basis_time)
  n = num_reduced_dofs(a)
  nright = num_reduced_dofs(proj_right)

  proj_basis = zeros(nleft,n,nright)
  @inbounds for is = 1:ns, it = 1:nt
    ist = (it-1)*ns+is
    proj_basis[:,ist,:] = kron(proj_basis_time[:,it,:],proj_basis_space[:,is,:])
  end

  return ReducedProjection(proj_basis)
end

function RBSteady.empirical_interpolation(a::TransientBasis)
  indices_space,interp_space = empirical_interpolation(get_basis_space(a))
  indices_time,interp_time = empirical_interpolation(get_basis_time(a))
  interp = kron(interp_time,interp_space)
  return (indices_space,indices_time),interp
end

# tt interface

get_cores_space(a::TTSVDCores) = get_cores(a)[1:end-1]
get_core_time(a::TTSVDCores) = get_cores(a)[end]

function RBSteady.galerkin_projection(
  proj_left::TTSVDCores,
  a::TTSVDCores,
  proj_right::TTSVDCores,
  combine)

  # space
  pl_space = get_cores_space(proj_left)
  a_space = get_cores_space(a)
  pr_space = get_cores_space(proj_right)
  p_space = contraction.(pl_space,a_space,pr_space)

  # time
  pl_time = get_core_time(proj_left)
  a_time = get_core_time(a)
  pr_time = get_core_time(proj_right)
  p_time = contraction(pl_time,a_time,pr_time,combine)

  p = sequential_product(p_space...,p_time)
  proj_cores = dropdims(p;dims=(1,2,3))

  return ReducedProjection(proj_cores)
end

# multfield interface

function Arrays.return_cache(::typeof(projection),red::TransientReduction,s::AbstractSnapshots)
  cache_space = return_cache(projection,get_reduction_space(red),s)
  cache_time = return_cache(projection,get_reduction_time(red),s)
  return TransientBasis(cache_space,cache_time)
end

function RBSteady.enrich!(
  red::SupremizerReduction,
  a::BlockProjection{<:TransientBasis},
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix;
  kwargs...)

  @check a.touched[1] "Primal field not defined"
  a_primal,a_dual... = a.array
  a_primal_space = a_primal.basis_space
  a_primal_time = a_primal.basis_time
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(a_dual)
    if a.touched[i]
      dual_i_space = get_basis_space(a_dual[i])
      C_primal_dual_i = supr_matrix[Block(1,i+1)]
      supr_space_i = H_primal \ C_primal_dual_i * dual_i_space
      a_primal_space = union(a_primal_space,supr_space_i,X_primal)

      dual_i_time = get_basis_time(a_dual[i])
      a_primal_time = time_enrichment(red,a_primal_time,dual_i_time;kwargs...)
    end
  end
  a[1] = TransientBasis(a_primal_space,a_primal_time)
  return
end

"""
    time_enrichment(basis_time::ArrayBlock;kwargs...) -> Vector{<:Matrix}

Enriches the temporal basis with temporal supremizers computed from
the kernel of the temporal basis associated to the primal field projected onto
the column space of the temporal basis (bases) associated to the duel field(s)

"""
function time_enrichment(red::SupremizerReduction,a_primal::Projection,basis_dual)
  tol = RBSteady.get_supr_tol(red)
  time_red = get_reduction_time(get_reduction(red))
  basis_primal′ = time_enrichment(get_basis(a_primal),basis_dual,tol)
  projection(time_red,basis_primal′)
end

function time_enrichment(basis_primal,basis_dual,tol)
  basis_pd = basis_primal'*basis_dual

  function enrich!(basis_primal,basis_pd,v)
    vnew = copy(v)
    orth_complement!(vnew,basis_primal)
    vnew /= norm(vnew)
    hcat(basis_primal,vnew),vcat(basis_pd,vnew'*basis_dual)
  end

  i = 1
  while i ≤ size(basis_pd,2)
    proj = i == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd[:,i],basis_pd[:,1:i-1])
    dist = norm(basis_pd[:,i]-proj)
    if dist ≤ tol
      basis_primal,basis_pd = enrich!(basis_primal,basis_pd,basis_dual[:,i])
      i = 0
    else
      basis_pd[:,i] .-= proj
    end
    i += 1
  end

  return basis_primal
end
