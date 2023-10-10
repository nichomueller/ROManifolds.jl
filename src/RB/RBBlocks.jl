abstract type RBBlock{T,N} end

Base.getindex(b::RBBlock,i...) = b.blocks[i...]
Base.iterate(b::RBBlock,args...) = iterate(b.blocks,args...)
get_blocks(b) = b.blocks
get_nblocks(b) = length(b.blocks)

struct BlockSnapshots{T} <: RBBlock{T,1}
  blocks::Vector{Snapshots{T}}

  function BlockSnapshots(v::Vector{Vector{<:PTArray{T}}}) where T
    blocks = Snapshots.(v)
    new{T}(blocks)
  end
end

const AbstractSnapshots{T} = Union{Snapshots{T},BlockSnapshots{T}}

struct BlockNnzMatrix{T} <: RBBlock{T,1}
  blocks::Vector{NnzMatrix{T}}

  function BlockNnzMatrix(blocks::Vector{NnzMatrix{T}}) where T
    @check all([length(nzm) == length(blocks[1]) for nzm in blocks[2:end]])
    new{T}(blocks)
  end
end

function NnzArray(s::BlockSnapshots{T}) where T
  blocks = map(s.snaps) do val
    array = get_array(hcat(val...))
    NnzMatrix(array...)
  end
  BlockNnzMatrix(blocks)
end

struct BlockRBSpace{T} <: RBBlock{T,1}
  blocks::Vector{BlockRBSpace{T}}

  function BlockRBSpace(blocks::Vector{BlockRBSpace{T}}) where T
    new{T}(blocks)
  end

  function BlockRBSpace(bases_space::Vector{Matrix{T}},bases_time::Vector{Matrix{T}}) where T
    blocks = map(RBSpace,(bases_space,bases_time))
    BlockRBSpace(blocks)
  end
end

const AbstractRBSpace{T} = Union{RBSpace{T},BlockRBSpace{T}}

function num_rb_dofs(rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  ndofs = 0
  @inbounds for i = 1:nblocks
    ndofs += num_rb_dofs(rb[i])
  end
  ndofs
end

function reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...;
  compute_supremizers=false,
  kwargs...)

  energy_norm = info.energy_norm
  nblocks = get_nblocks(snaps)
  bases = map(1:nblocks) do n
    feopn = feop[n]
    norm_matrix = get_norm_matrix(feop,energy_norm[n])
    basis_space_nnz,basis_time = compress(info,feopn,snaps[n],norm_matrix,args...)
    basis_space = recast(basis_space_nnz)
    basis_space,basis_time
  end
  bases_space = first.(bases)
  bases_time = last.(bases)
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
    println("Computing supremizers in space for dual field $idx")
    feop_i = feop[idx]
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
    println("Computing supremizers in time for dual field $idx")
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

  println("Added $count time supremizers")
  basis_u
end

struct BlockRBAlgebraicContribution{T,N} <: BlockRBAlgebraicContribution{T,N}
  blocks::Array{RBAlgebraicContribution{T},N}

  function BlockRBAlgebraicContribution(
    blocks::Array{RBAlgebraicContribution{T},N}) where {T,N}

    new{T,N}(blocks)
  end
end

const AbstractRBAlgebraicContribution{T,N} = Union{RBAlgebraicContribution{T},BlockRBAlgebraicContribution{T,N}}

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::BlockRBSpace,
  snaps::BlockSnapshots,
  μ::Table)

  nblocks = get_nblocks(rbspace)
  nsnaps = info.nsnaps_system
  _μ = snapsθ[1:nsnaps]
  _snapsθ = map(1:nblocks) do row
    snapsθ = recenter(fesolver,snaps[row],μ)
    snapsθ[1:nsnaps]
  end
  rhs = collect_compress_rhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::BlockRBSpace,
  snaps::Vector{<:PTArray},
  μ::Table)

  times = get_times(fesolver)
  nblocks = get_nblocks(rbspace)
  @assert length(snaps) == nblocks
  blocks = map(index_pairs(nblocks,1)) do (row,col)
    feop_row_col = feop[row,col]
    snaps_row = snaps[row]
    rbspace_row = rbspace[row]
    ress,trian = collect_residuals_for_trian(fesolver,feop_row_col,snaps_row,μ,times)
    compress_component(info,feop_row,ress,trian,times,rbspace_row)
  end
  return BlockRBAlgebraicContribution(blocks)
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::BlockRBSpace{T},
  snaps::Vector{<:PTArray},
  μ::Table) where T

  times = get_times(fesolver)
  θ = fesolver.θ

  njacs = length(feop.jacs)
  ad_jacs = Vector{BlockRBAlgebraicContribution{T,2}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    blocks = map(index_pairs(nblocks,nblocks)) do (row,col)
      feop_row_col = feop[row,col]
      snaps_row = snaps[row]
      rbspace_row = rbspace[row]
      rbspace_col = rbspace[col]
      jacs,trian = collect_jacobians_for_trian(fesolver,feop_row_col,snaps_row,μ,times;i)
      compress_component(info,feop_row_col,jacs,trian,times,rbspace_row,rbspace_col;combine_projections)
    end
    ad_jacs[i] = BlockRBAlgebraicContribution(blocks)
  end
  return ad_jacs
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::BlockRBAlgebraicContribution{T,1},
  sols::Vector{<:PTArray},
  args...) where T

  nblocks = get_nblocks(rbres)
  blocks = map(index_pairs(nblocks,1)) do (row,col)
    feop_row_col = feop[row,col]
    sols_row = sols[row]
    rbres_row = rbres[row]
    collect_rhs_contributions!(cache,info,feop_row_col,fesolver,rbres_row,sols_row,args...)
  end
  vcat(blocks...)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjacs::Vector{BlockRBAlgebraicContribution{T,2}},
  sols::Vector{<:PTArray},
  args...) where T

  njacs = length(rbjacs)
  nblocks = get_nblocks(testitem(rbjacs))
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    blocks = map(index_pairs(nblocks,nblocks)) do (row,col)
      feop_row_col = feop[row,col]
      sols_row = sols[row]
      rb_jac_i_row_col = rb_jac_i[row,col]
      collect_lhs_contributions!(cache,info,feop_row_col,fesolver,rb_jac_i_row_col,sols_row,args...;i)
    end
    rb_jacs_contribs[i] = hvcat(blocks...)
  end
  return sum(rb_jacs_contribs)
end
