abstract type AbstractRBSpace{T} end

get_basis_space(rb::AbstractRBSpace) = rb.basis_space
get_basis_time(rb::AbstractRBSpace) = rb.basis_time

struct RBSpace{T} <: AbstractRBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpace(
    basis_space_nnz::NnzMatrix{T},
    basis_time_nnz::NnzMatrix{T}) where T

    basis_space = recast(basis_space_nnz)
    basis_time = get_nonzero_val(basis_time_nnz)
    new{T}(basis_space,basis_time)
  end
end

struct BlockRBSpace{T} <: AbstractRBSpace{T}
  basis_space::Vector{Matrix{T}}
  basis_time::Vector{Matrix{T}}

  function BlockRBSpace(
    basis_space_nnz::BlockNnzMatrix{T},
    basis_time_nnz::BlockNnzMatrix{T}) where T

    basis_space = recast(basis_space_nnz)
    basis_time = get_nonzero_val(basis_time_nnz)
    new{T}(basis_space,basis_time)
  end
end

abstract type PODStyle end
struct DefaultPOD <: PODStyle end
struct SteadyPOD <: PODStyle end
struct TranposedPOD <: PODStyle end

function compress_snapshots(info::RBInfo,feop::PTFEOperator,nzm::NnzMatrix,args...)
  ϵ = info.ϵ
  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(energy_norm,feop)
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  transposed = size(nzm,1) < size(nzm,2) ? TranposedPOD() : DefaultPOD()
  basis_space,basis_time = compress_snapshots(nzm,norm_matrix,steady,transposed;ϵ)
  RBSpace(basis_space,basis_time)
end

function compress_snapshots(
  info::RBInfo,
  feop::PTFEOperator,
  nzm::BlockNnzMatrix,
  args...;
  compute_supremizers=false,
  kwargs...)

  nfields = length(nzm)
  all_idx = index_pairs(1:nfields,1)
  rb = map(all_idx) do i
    feopi = filter_operator(feop,i)
    nzmi = nzm[i]
    compress_snapshots(info,feopi,nzmi)
  end
  bases_space = map(get_basis_space,rb)
  bases_time = map(get_basis_time,rb)
  if compute_supremizers
    add_space_supremizers!(bases_space,feop,nzm;norm_matrix)
    add_time_supremizers!(bases_time;kwargs...)
  end
  BlockRBSpace(basis_space,basis_time)
end

function compress_snapshots(
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

function compress_snapshots(
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
    function compress_snapshots(
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

function add_space_supremizers!(
  bases_space::Vector{<:Matrix},
  feop::PTFEOperator,
  args...;
  kwargs...)

  bs_primal,bs_dual... = bases_space
  n_dual_fields = length(bs_dual)
  all_idx = index_pairs(1:n_dual_fields,1)
  for idx in all_idx
    printstyled("Computing supremizers in space for dual field $idx\n";color=:blue)
    feop_i = filter_operator(feop,idx)
    supr_i = space_supremizers(bs_dual[idx],feop_i,args...)
    orth_supr_i = gram_schmidt(supr_i,bs_primal)
    bs_primal = hcat(bs_primal,orth_supr_i)
  end
  return bs_primal,bs_dual
end

# function space_supremizers(
#   basis_space::AbstractMatrix,
#   snaps::BlockNnzMatrix,
#   feop::PTFEOperator,
#   fesolver::ODESolver,
#   params::Table)

#   collector = CollectJacobiansMap(fesolver,feop)
#   constraint_mat = lazy_map(collector,snaps,params)
#   if length(constraint_mat) == 1
#     return constraint_mat*basis_space # THIS IS WRONG
#   else
#     return map(*,constraint_mat,snaps) # THIS IS WRONG
#   end
# end

function add_time_supremizers!(bases_time::Vector{<:Matrix};ttol::Real)
  bt_primal,bt_dual... = bases_time
  n_dual_fields = length(bt_dual)
  all_idx = index_pairs(1:n_dual_fields,1)
  for idx in all_idx
    printstyled("Computing supremizers in time for dual field $idx\n";color=:blue)
    supr_i = add_time_supremizers(bt_primal,bt_dual[idx];ttol)
    append!(bt_primal,supr_i)
  end
  return bt_primal,btdual
end

function add_time_supremizers(bases_time::AbstractMatrix...;ttol=1e-2)
  basis_u,basis_p = bases_time
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
