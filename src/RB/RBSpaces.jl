abstract type AbstractRBSpace{T} end

get_basis_space(rb::AbstractRBSpace) = rb.basis_space
get_basis_time(rb::AbstractRBSpace) = rb.basis_time

function Algebra.allocate_vector(rb::AbstractRBSpace{T}) where T
  zeros(T,num_rb_dofs(rb))
end

function Algebra.allocate_vector(rb::AbstractRBSpace{T}) where T
  zeros(T,num_rb_dofs(rb))
end

function save(info::RBInfo,rb::AbstractRBSpace)
  if info.save_structures
    path = joinpath(info.rb_path,"rb")
    save(path,rb)
  end
end

function load(info::RBInfo,T::Type{AbstractRBSpace})
  path = joinpath(info.rb_path,"rb")
  load(path,T)
end

struct RBSpace{T} <: AbstractRBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpace(
    basis_space::Matrix{T},
    basis_time::Matrix{T}) where T
    new{T}(basis_space,basis_time)
  end

  function RBSpace(
    basis_space_nnz::NnzMatrix{T},
    basis_time_nnz::NnzMatrix{T}) where T

    basis_space = recast(basis_space_nnz)
    basis_time = get_nonzero_val(basis_time_nnz)
    new{T}(basis_space,basis_time)
  end
end

function num_rb_dofs(rb::RBSpace)
  size(rb.basis_space,2)*size(rb.basis_time,2)
end

function get_reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Snapshots,
  args...)

  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(energy_norm,feop)
  basis_space,basis_time = compress(info,feop,snaps,norm_matrix,args...)
  RBSpace(basis_space,basis_time)
end

function recast(rb::RBSpace,x::PTArray{T}) where T
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  ns_rb = size(basis_space,2)
  nt_rb = size(basis_time,2)

  n = length(x)
  array = Vector{T}(undef,n)
  @inbounds for i = 1:n
    x_mat_i = reshape(x[i],nt_rb,ns_rb)
    x_i = basis_space*(basis_time*x_mat_i)'
    array[i] = copy(x_i)
  end

  PTArray(array)
end

struct BlockRBSpace{T} <: AbstractRBSpace{T}
  basis_space::Vector{Matrix{T}}
  basis_time::Vector{Matrix{T}}

  function BlockRBSpace(
    basis_space::Vector{Matrix{T}},
    basis_time::Vector{Matrix{T}}) where T
    new{T}(basis_space,basis_time)
  end

  function BlockRBSpace(
    basis_space_nnz::BlockNnzMatrix{T},
    basis_time_nnz::BlockNnzMatrix{T}) where T

    basis_space = recast(basis_space_nnz)
    basis_time = get_nonzero_val(basis_time_nnz)
    new{T}(basis_space,basis_time)
  end
end

get_nfields(rb::BlockRBSpace) = length(rb.basis_space)
Base.getindex(rb::BlockRBSpace,i...) = RBSpace(rb.basis_space[i...],rb.basis_time[i...])

function num_rb_dofs(rb::BlockRBSpace)
  nfields = get_nfields(rb)
  ndofs = 0
  @inbounds for i = 1:nfields
    ndofs += num_rb_dofs(rb[i])
  end
  ndofs
end

function get_reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...)

  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(energy_norm,feop)
  basis_space,basis_time = compress(info,feop,snaps,norm_matrix,args...)
  BlockRBSpace(basis_space,basis_time)
end

function compress(info::RBInfo,::PTFEOperator,snaps,args...)
  nzm = NnzArray(snaps)
  ϵ = info.ϵ
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  transposed = size(nzm,1) < size(nzm,2) ? TranposedPOD() : DefaultPOD()
  compress(nzm,steady,transposed,args...;ϵ)
end

function compress(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...;
  compute_supremizers=false,
  kwargs...)

  nzm = NnzArray(snaps)
  nfields = get_nfields(nzm)
  all_idx = index_pairs(nfields,1)
  rb = map(all_idx) do i
    feopi = filter_operator(feop,i)
    compress(info,feopi,nzm[i])
  end
  bases_space = map(get_basis_space,rb)
  bases_time = map(get_basis_time,rb)
  if compute_supremizers
    bases_space = add_space_supremizers(bases_space,feop,snaps,args...)
    bases_time = add_time_supremizers(bases_time;kwargs...)
  end
  BlockRBSpace(bases_space,bases_time)
end

function add_space_supremizers(
  bases_space::Vector{<:Matrix},
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  norm_matrix,
  args...)

  bs_primal,bs_dual... = bases_space
  n_dual_fields = length(bs_dual)
  all_idx = index_pairs(n_dual_fields,1)
  for idx in all_idx
    printstyled("Computing supremizers in space for dual field $idx\n";color=:blue)
    feop_i = filter_operator(feop,idx)
    supr_i = space_supremizers(bs_dual[idx],feop_i,snaps[idx],args...)
    orth_supr_i = gram_schmidt(supr_i,bs_primal,norm_matrix)
    bs_primal = hcat(bs_primal,orth_supr_i)
  end
  return bs_primal,bs_dual
end

function space_supremizers(
  basis_space::Matrix,
  feop::PTFEOperator,
  snaps::Snapshots,
  fesolver::PODESolver,
  args...)

  constraint_mat = collect_jacobians(fesolver,feop,snaps,args...)
  if length(constraint_mat) == 1
    return constraint_mat*basis_space
  else
    @assert length(constraint_mat) == length(snaps)
    return map(*,constraint_mat,snaps)
  end
end

function add_time_supremizers(bases_time::Vector{<:Matrix};ttol::Real)
  bt_primal,bt_dual... = bases_time
  n_dual_fields = length(bt_dual)
  all_idx = index_pairs(n_dual_fields,1)
  for idx in all_idx
    printstyled("Computing supremizers in time for dual field $idx\n";color=:blue)
    supr_i = add_time_supremizers(bt_primal,bt_dual[idx];ttol)
    append!(bt_primal,supr_i)
  end
  return bt_primal,btdual
end

function add_time_supremizers(basis_u::Matrix,basis_p::Matrix;ttol=1e-2)
  basis_up = basis_u'*basis_p

  function enrich(
    basis_u::AbstractMatrix,
    basis_up::AbstractMatrix,
    v::AbstractArray)

    vnew = orth_complement(v,basis_u)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),vcat(basis_up,vnew'*basis_p)
  end

  count = 0
  ntp_minus_ntu = size(basis_p,2) - size(basis_u,2)
  if ntp_minus_ntu > 0
    for ntp = 1:ntp_minus_ntu
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
    end
  end

  ntp = 1
  while ntp ≤ size(basis_up,2)
    proj = ntp == 1 ? zeros(size(basis_up[:,1])) : orth_projection(basis_up[:,ntp],basis_up[:,1:ntp-1])
    dist = norm(basis_up[:,1]-proj)
    if dist ≤ ttol
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
      ntp = 0
    else
      basis_up[:,ntp] -= proj
    end
    ntp += 1
  end

  printstyled("Added $count time supremizers\n";color=:blue)
  basis_u
end

function filter_rbspace(rb::BlockRBSpace,idx::Int)
  basis_time = get_basis_space(rb)[idx]
  basis_time = get_basis_time(rb)[idx]
  RBSpace(basis_space,basis_time)
end
