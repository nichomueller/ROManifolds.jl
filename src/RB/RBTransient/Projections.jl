function RBSteady.projection(red::TransientAffineReduction,s::AbstractTransientSnapshots,args...)
  s1 = flatten_snapshots(select_snapshots(s,1,1))
  basis_space = projection(get_reduction_space(red),s1,args...)
  basis_time = PODBasis(I[1:num_times(s),1:1])
  TransientPODBasis(basis_space,basis_time)
end

function RBSteady.projection(red::TransientPODReduction,s::AbstractTransientSnapshots,args...)
  s1 = flatten_snapshots(s)
  basis_space = projection(get_reduction_space(red),s1,args...)
  proj_s1 = galerkin_projection(get_basis(basis_space),s1,args...)
  proj_s2 = change_mode(proj_s1,num_params(s))
  basis_time = projection(get_reduction_time(red),proj_s2)
  TransientPODBasis(basis_space,basis_time)
end

"""
"""
struct TransientPODBasis{A<:PODBasis,B<:PODBasis} <: Projection
  basis_space::A
  basis_time::B
end

get_basis_space(a::TransientPODBasis) = get_basis(a.basis_space)
get_basis_time(a::TransientPODBasis) = get_basis(a.basis_time)

RBSteady.get_basis(a::TransientPODBasis) = kron(get_basis_time(a),get_basis_space(a))
RBSteady.num_fe_dofs(a::TransientPODBasis) = num_fe_dofs(a.basis_space)*num_fe_dofs(a.basis_time)
RBSteady.num_reduced_dofs(a::TransientPODBasis) = num_reduced_dofs(a.basis_space)*num_reduced_dofs(a.basis_time)

function RBSteady.project(a::TransientPODBasis,X::AbstractMatrix)
  basis_space = get_basis(a.basis_space)
  basis_time = get_basis(a.basis_time)
  X̂ = basis_space'*X̂*basis_time
  return x̂
end

function RBSteady.inv_project(a::TransientPODBasis,X̂::AbstractMatrix)
  basis_space = get_basis(a.basis_space)
  basis_time = get_basis(a.basis_time)
  X = basis_space*X̂*basis_time'
  return X
end

for f in (:(RBSteady.project),:(RBSteady.inv_project))
  @eval begin
    function $f(a::TransientPODBasis,y::AbstractVector)
      ns = num_reduced_dofs(a.basis_space)
      nt = num_reduced_times(a.basis_time)
      Y = reshape(y,ns,nt)
      $f(a,Y)
    end
  end
end

function RBSteady.galerkin_projection(
  proj_left::TransientPODBasis,
  a::TransientPODBasis)

  proj_basis_space = galerkin_projection(get_basis_space(proj_left),get_basis_space(a))
  proj_basis_time = galerkin_projection(get_basis_time(proj_left),get_basis_time(a))
  proj_basis = kron(proj_basis_time,proj_basis_space)
  return ReducedProjection(proj_basis)
end

function RBSteady.galerkin_projection(
  proj_left::TransientPODBasis,
  a::TransientPODBasis,
  proj_right::TransientPODBasis)

  @notimplemented "In unsteady problems, we need to provide a combining function"
end

function RBSteady.galerkin_projection(
  proj_left::TransientPODBasis,
  a::TransientPODBasis,
  proj_right::TransientPODBasis,
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

function RBSteady.empirical_interpolation(a::TransientPODBasis)
  indices_space,interp_space = empirical_interpolation(get_basis_space(a))
  indices_time,interp_time = empirical_interpolation(get_basis_time(a))
  interp = kron(interp_time,interp_space)
  return (indices_time,indices_time),interp
end

# multfield interface

function RBSteady.enrich(
  a::BlockProjection{<:TransientPODBasis},
  norm_matrix::AbstractMatrix,
  supr_matrix::AbstractMatrix;
  kwargs...)

  @check length(findall(a.touched)) == length(a)
  a_primal,a_dual... = a.array
  a_primal_space = a_primal.basis_space
  a_primal_time = a_primal.basis_time
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(a_dual)
    a_dual_space = a_dual[i].basis_space
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_space_i = H_primal \ C_primal_dual_i * a_dual_space[i]
    a_primal_space = union(a_primal_space,supr_space_i,X_primal)

    a_dual_time = a_dual[i].basis_time
    a_primal_time = time_enrichment(a_primal_time,a_dual_time;kwargs...)
  end
  return BlockProjection([a_primal,a_dual...],a.touched)
end

"""
    time_enrichment(basis_time::ArrayBlock;kwargs...) -> Vector{<:Matrix}

Enriches the temporal basis with temporal supremizers computed from
the kernel of the temporal basis associated to the primal field projected onto
the column space of the temporal basis (bases) associated to the duel field(s)

"""
function time_enrichment(red,a_primal::Projection,a_dual::Projection)
  basis_primal′ = time_enrichment(red,get_basis(a_primal),get_basis(a_dual))
  projection(red,basis_primal′)
end

function time_enrichment(red,basis_primal,basis_dual;tol=1e-2)
  basis_pd = basis_primal'*basis_dual

  function enrich(basis_primal,basis_pd,v)
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
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,i])
      i = 0
    else
      basis_pd[:,i] .-= proj
    end
    i += 1
  end

  return basis_primal
end
