abstract type AbstractRBSpace{T} end

get_basis_space(rb::AbstractRBSpace) = rb.basis_space
get_basis_time(rb::AbstractRBSpace) = rb.basis_time

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

function get_reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Vector{<:PTArray},
  args...)

  basis_space,basis_time = compress(info,feop,snaps,args...)
  RBSpace(basis_space,basis_time)
end

function recast(rbspace::RBSpace,x::PTArray{T}) where T
  basis_space = get_basis_space(rbspace)
  basis_time = get_basis_time(rbspace)
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

function get_reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Vector{Vector{<:PTArray}},
  args...)

  basis_space,basis_time = compress(info,feop,snaps,args...)
  BlockRBSpace(basis_space,basis_time)
end

abstract type PODStyle end
struct DefaultPOD <: PODStyle end
struct SteadyPOD <: PODStyle end
struct TranposedPOD <: PODStyle end

function compress(info::RBInfo,snaps::PTArray)
  nzm = NnzArray(snaps)
  ϵ = info.ϵ
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  transposed = size(nzm,1) < size(nzm,2) ? TranposedPOD() : DefaultPOD()
  compress(nzm,steady,transposed;ϵ)
end

function compress(info::RBInfo,feop::PTFEOperator,snaps::Snapshots,args...)
  nzm = NnzArray(snaps)
  ϵ = info.ϵ
  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(energy_norm,feop)
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  transposed = size(nzm,1) < size(nzm,2) ? TranposedPOD() : DefaultPOD()
  compress(nzm,norm_matrix,steady,transposed;ϵ)
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
    bases_space = add_space_supremizers(bases_space,feop,snaps,args...;norm_matrix)
    bases_time = add_time_supremizers(bases_time;kwargs...)
  end
  BlockRBSpace(bases_space,bases_time)
end

function compress(
  nzm::NnzMatrix,
  norm_matrix,
  args...;
  kwargs...)

  basis_space = tpod(nzm,norm_matrix;kwargs...)
  compressed_nza = prod(basis_space,nzm)
  compressed_nza_t = change_mode(compressed_nza)
  basis_time = tpod(compressed_nza_t;kwargs...)
  basis_space,basis_time
end

function compress(
  nzm::NnzMatrix,
  norm_matrix,
  ::DefaultPOD,
  ::TranposedPOD;
  kwargs...)

  nza_t = change_mode(nzm)
  basis_time = tpod(nza_t;kwargs...)
  compressed_nza_t = prod(basis_time,nza_t)
  compressed_nza = change_mode(compressed_nza_t)
  basis_space = tpod(compressed_nza,norm_matrix;kwargs...)
  basis_space,basis_time
end

for T in (:DefaultPOD,:TranposedPOD)
  @eval begin
    function compress(
      nzm::NnzMatrix,
      norm_matrix,
      ::SteadyPOD,
      ::$T;
      kwargs...)

      basis_space = tpod(nzm,norm_matrix;kwargs...)
      basis_time = ones(eltype(nzm),1,1)
      basis_space,basis_time
    end
  end
end

function add_space_supremizers(
  bases_space::Vector{<:Matrix},
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...;
  kwargs...)

  bs_primal,bs_dual... = bases_space
  n_dual_fields = length(bs_dual)
  all_idx = index_pairs(n_dual_fields,1)
  for idx in all_idx
    printstyled("Computing supremizers in space for dual field $idx\n";color=:blue)
    feop_i = filter_operator(feop,idx)
    supr_i = space_supremizers(bs_dual[idx],feop_i,snaps[idx],args...)
    orth_supr_i = gram_schmidt(supr_i,bs_primal)
    bs_primal = hcat(bs_primal,orth_supr_i)
  end
  return bs_primal,bs_dual
end

function space_supremizers(
  basis_space::Matrix,
  feop::PTFEOperator,
  snaps::Vector{<:PTArray},
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

function filter_rbspace(rbspace::BlockRBSpace,idx::Int)
  basis_time = get_basis_space(rbspace)[idx]
  basis_time = get_basis_time(rbspace)[idx]
  RBSpace(basis_space,basis_time)
end
