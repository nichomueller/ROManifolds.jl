abstract type RBSpace{T} end

struct SingleFieldRBSpace{T} <: RBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function SingleFieldRBSpace(
    basis_space_nnz::NnzArray{T,N,OT},
    basis_time_nnz::NnzArray{T,N,OT}) where {T,N,OT}

    basis_space = recast(basis_space_nnz)
    basis_time = get_nonzero_val(basis_time_nnz)
    new{T}(basis_space,basis_time)
  end
end

function compress_snapshots(
  info::RBInfo,
  snaps::SingleFieldSnapshots,
  feop::ParamTransientFEOperator,
  args...)

  ϵ = info.ϵ
  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(energy_norm,feop)

  basis_space,basis_time = tpod(snaps,norm_matrix;ϵ)
  SingleFieldRBSpace(basis_space,basis_time)
end

struct MultiFieldRBSpace{T} <: RBSpace{T}
  basis_space::BlockMatrix{T}
  basis_time::BlockMatrix{T}

  function MultiFieldRBSpace(
    basis_space_nnz::BlockNnzArray{T,N,OT},
    basis_time_nnz::BlockNnzArray{T,N,OT}) where {T,N,OT}

    basis_space = recast(basis_space_nnz)
    basis_time = get_nonzero_val(basis_time_nnz)
    new{T}(basis_space,basis_time)
  end
end

function compress_snapshots(
  info::RBInfo,
  snaps::MultiFieldSnapshots,
  feop::ParamTransientFEOperator;
  compute_supremizers=false,
  kwargs...)

  ϵ = info.ϵ
  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(energy_norm,feop)

  basis_space,basis_time = tpod(snaps,norm_matrix;ϵ)
  if compute_supremizers
    add_space_supremizers!(basis_space,snaps,feop,args...;X=norm_matrix)
    add_time_supremizers!(basis_time;kwargs...)
  end
  MultiFieldRBSpace(basis_space,basis_time)
end

get_basis_space(rb::RBSpace) = rb.basis_space

get_basis_time(rb::RBSpace) = rb.basis_time

function add_space_supremizers!(
  bases_space::BlockMatrix,
  feop::ParamTransientFEOperator,
  args...;
  kwargs...)

  bsprimal,bsdual... = bases_space.blocks
  for (i,bsd_i) in enumerate(bsdual)
    printstyled("Computing supremizers in space for dual field $i\n";color=:blue)
    filter = (1,i+1)
    filt_op = filter_operator(feop,filter)
    supr_i = space_supremizers(bsd_i,filt_op,args...)
    orth_supr_i = gram_schmidt(supr_i,bsprimal)
    append!(bsprimal,orth_supr_i) # THIS IS WRONG
  end
  return bsprimal,bsdual
end

function space_supremizers(
  bs::AbstractMatrix,
  snaps::MultiFieldSnapshots,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  params::Table)

  collector = CollectJacobiansMap(fesolver,feop)
  constraint_mat = lazy_map(collector,snaps,params)
  if length(constraint_mat) == 1
    return constraint_mat*bs # THIS IS WRONG
  else
    return map(*,constraint_mat,snaps) # THIS IS WRONG
  end
end

function add_time_supremizers!(bases_time::BlockMatrix;ttol::Real)
  btprimal,btdual... = bases_time.blocks
  for (i,btd_i) in enumerate(tbdual)
    printstyled("Computing supremizers in time for dual field $i\n";color=:blue)
    supr_i = add_time_supremizers(btprimal,btd_i;ttol)
    append!(btprimal,supr_i)
  end
  return btprimal,btdual
end

function add_time_supremizers(bases_time::AbstractMatrix...;ttol=1e-2)
  basis_u,basis_p = bases_time
  basis_up = basis_u'*basis_p

  function enrich(
    basis_u::AbstractMatrix{Float},
    basis_up::AbstractMatrix{Float},
    v::AbstractArray{Float})

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

function filter_rbspace(rbspace::RBSpace,filter::Int)
  if isa(rbspace,MultiFieldRBSpace)
    _bs,_bt = get_basis_space(rbspace),get_basis_time(rbspace)
    basis_space,basis_time = _bs.blocks[filter],_bt.blocks[filter]
    return SingleFieldRBSpace(basis_space,basis_time)
  else
    return rbspace
  end
end
