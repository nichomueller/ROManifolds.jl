abstract type RBSpace end

struct RBSpaceSteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
end

function RBSpaceSteady(
  snaps::Snapshots;
  ϵ=1e-4,style=ReducedPOD())

  id = get_id(snaps)
  basis_space = assemble_spatial_rb(snaps;ϵ,style)
  RBSpaceSteady(id,Matrix(basis_space))
end

function RBSpaceSteady(
  snaps::NTuple{2,Snapshots};
  ϵ=1e-4,style=ReducedPOD())

  all_snaps = vcat(snaps...)
  id = get_id(all_snaps)
  basis_space = assemble_spatial_rb(all_snaps;ϵ,style)
  RBSpaceSteady(id,Matrix(basis_space))
end

struct RBSpaceUnsteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
  basis_time::Matrix{Float}
end

function RBSpaceUnsteady(
  snaps::Snapshots;
  ϵ=1e-4,style=ReducedPOD())

  id = get_id(snaps)
  basis_space,basis_time = assemble_spatio_temporal_rb(snaps;ϵ,style)
  RBSpaceUnsteady(id,Matrix(basis_space),Matrix(basis_time))
end

function RBSpaceUnsteady(
  snaps::NTuple{2,Snapshots};
  ϵ=1e-4,style=ReducedPOD())

  all_snaps = vcat(snaps...)
  id = get_id(all_snaps)
  basis_space,basis_time = assemble_spatio_temporal_rb(all_snaps;ϵ,style)
  RBSpaceUnsteady(id,Matrix(basis_space),Matrix(basis_time))
end

function assemble_rb_space(
  info::RBInfo,
  snaps::NTuple{N,Snapshots},
  args...;
  kwargs...)::NTuple{N,RBSpace} where N

  if info.load_offline
    printstyled("Loading reduced bases\n";color=:blue)
    load(info,get_id(snaps))
  else
    printstyled("Assembling reduced bases\n";color=:blue)
    assemble_rb_space(info,snaps...,args...;kwargs...)
  end
end

function assemble_rb_space(
  info::RBInfoSteady,
  snaps_u::Snapshots;
  kwargs...)

  basis_space = assemble_spatial_rb(snaps_u;ϵ=info.ϵ,kwargs...)
  (RBSpaceSteady(get_id(snaps_u),basis_space),)
end

function assemble_rb_space(
  info::RBInfoUnsteady,
  snaps_u::Snapshots;
  kwargs...)

  basis_space,basis_time = assemble_spatio_temporal_rb(snaps_u;ϵ=info.ϵ,kwargs...)
  (RBSpaceUnsteady(get_id(snaps_u),basis_space,basis_time),)
end

function assemble_rb_space(
  info::RBInfoSteady,
  snaps_u::Snapshots,
  snaps_p::Snapshots,
  args...;
  kwargs...)

  def = isindef(info)

  bs_u = assemble_spatial_rb(snaps_u;ϵ=info.ϵ,kwargs...)
  bs_p = assemble_spatial_rb(snaps_p;ϵ=info.ϵ,kwargs...)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),args...)

  rbspace_u = RBSpaceSteady(get_id(snaps_u),bs_u_supr)
  rbspace_p = RBSpaceSteady(get_id(snaps_p),bs_p)

  (rbspace_u,rbspace_p)
end

function assemble_rb_space(
  info::RBInfoUnsteady,
  snaps_u::Snapshots,
  snaps_p::Snapshots,
  args...;
  kwargs...)

  def = isindef(info)
  opB,ttol... = args

  bs_u,bt_u = assemble_spatio_temporal_rb(snaps_u;ϵ=info.ϵ,kwargs...)
  bs_p,bt_p = assemble_spatio_temporal_rb(snaps_p;ϵ=info.ϵ,kwargs...)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),opB)
  bt_u_supr = add_time_supremizers(def,(bt_u,bt_p),ttol...)

  rbspace_u = RBSpaceUnsteady(get_id(snaps_u),bs_u_supr,bt_u_supr)
  rbspace_p = RBSpaceUnsteady(get_id(snaps_p),bs_p,bt_p)

  (rbspace_u,rbspace_p)
end

function assemble_spatial_rb(snap::Snapshots;kwargs...)
  POD(snap;kwargs...)
end

function assemble_spatio_temporal_rb(snap::Snapshots;kwargs...)
  Nr,Nc = size(get_snap(snap))
  assemble_spatio_temporal_rb(Val{Nr > Nc}(),snap;kwargs...)
end

function assemble_spatio_temporal_rb(::Val{false},snap::Snapshots;kwargs...)
  snap_space = get_snap(snap)
  ns = get_nsnap(snap)
  basis_space = POD(snap_space;kwargs...)
  red_snap_time = mode2_unfolding(basis_space'*snap_space,ns)
  basis_time = POD(red_snap_time;kwargs...)

  basis_space,basis_time
end

function assemble_spatio_temporal_rb(::Val{true},snap::Snapshots;kwargs...)
  snap_space = get_snap(snap)
  ns = get_nsnap(snap)
  snap_time = mode2_unfolding(snap_space,ns)
  basis_time = POD(snap_time;kwargs...)
  red_snap_space = mode2_unfolding(basis_time'*snap_time,ns)
  basis_space = POD(red_snap_space;kwargs...)
  basis_space,basis_time
end

function add_space_supremizers(
  ::Val{false},
  basis::NTuple{2,AbstractMatrix{Float}},
  args...)

  first(basis)
end

function add_space_supremizers(
  ::Val{true},
  basis::NTuple{2,AbstractMatrix{Float}},
  opB::ParamBilinOperator)

  basis_u, = basis
  supr = assemble_space_supremizers(basis,opB)
  hcat(basis_u,supr)
end

function assemble_space_supremizers(
  basis::NTuple{2,AbstractMatrix{Float}},
  opB::ParamBilinOperator)

  printstyled("Computing supremizers in space\n";color=:blue)
  basis_u,basis_p = basis
  constraint_mat = assemble_constraint_matrix(opB,basis_p)
  gram_schmidt(constraint_mat,basis_u)
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,Ttr},
  basis_p::AbstractMatrix{Float}) where Ttr

  @assert opB.id == :B
  B = assemble_affine_quantity(opB)
  B'*basis_p
end

function assemble_constraint_matrix(
  ::ParamBilinOperator,
  ::AbstractMatrix{Float},
  ::Snapshots)

  error("Implement this")
end

function add_time_supremizers(
  ::Val{false},
  basis::NTuple{2,AbstractMatrix{Float}},
  args...)

  first(basis)
end

function add_time_supremizers(
  ::Val{true},
  basis::NTuple{2,AbstractMatrix{Float}},
  ttol=1e-2)

  printstyled("Checking if supremizers in time need to be added\n";color=:blue)

  basis_u,basis_p = basis
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

get_id(rb::RBSpace) = rb.id

get_basis_space(rb::RBSpace) = rb.basis_space

get_basis_time(rb::RBSpaceUnsteady) = rb.basis_time

get_Ns(rb::RBSpace) = size(rb.basis_space,1)

get_ns(rb::RBSpace) = size(rb.basis_space,2)

get_Nt(rb::RBSpaceUnsteady) = size(rb.basis_time,1)

get_nt(rb::RBSpaceUnsteady) = size(rb.basis_time,2)

function save(info::RBInfo,rb::RBSpace)
  id = get_id(rb)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  save(path_id,rb)
end

function save(path::String,rb::RBSpaceSteady)
  bs = get_basis_space(rb)
  save(joinpath(path,"basis_space"),bs)
end

function save(path::String,rb::RBSpaceUnsteady)
  bs = get_basis_space(rb)
  bt = get_basis_time(rb)
  save(joinpath(path,"basis_space"),bs)
  save(joinpath(path,"basis_time"),bt)
end

function load(info::RBInfoSteady,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  basis_space = load(joinpath(path_id,"basis_space"))
  RBSpaceSteady(id,basis_space)
end

function load(info::RBInfoUnsteady,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  basis_space = load(joinpath(path_id,"basis_space"))
  basis_time = load(joinpath(path_id,"basis_time"))
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function load(info::RBInfo,ids::NTuple{N,Symbol}) where N
  load_id(id::Symbol) = load(info,id)
  load_id.(ids)
end

function rb_space_projection(rbrow::RBSpace,mat::AbstractArray)
  brow = get_basis_space(rbrow)
  @assert size(brow,1) == size(mat,1) "Cannot project array"
  Matrix(brow'*mat)
end

function rb_space_projection(rbrow::RBSpace,rbcol::RBSpace,mat::AbstractMatrix)
  brow,bcol = get_basis_space(rbrow),get_basis_space(rbcol)
  @assert size(brow,1) == size(mat,1) && size(bcol,1) == size(mat,2) "Cannot project matrix"
  brow'*mat*bcol
end

function rb_time_projection(rbrow::RBSpaceUnsteady,mat::AbstractArray)
  brow = get_basis_time(rbrow)
  @assert size(brow,1) == size(mat,1) "Cannot project array"

  nrow = size(brow,2)
  Q = size(mat,2)
  proj = allocate_matrix(Matrix{Float},nrow,Q)

  @inbounds for q = 1:Q, it = 1:nrow
    proj[it,q] = sum(brow[:,it].*mat[:,q])
  end

  proj
end

function rb_time_projection(
  rbrow::RBSpaceUnsteady,
  rbcol::RBSpaceUnsteady,
  mat::AbstractMatrix;
  idx_forwards=1:size(mat,1),
  idx_backwards=1:size(mat,1))

  brow = get_basis_time(rbrow)
  bcol = get_basis_time(rbcol)
  @assert size(brow,1) == size(bcol,1) == size(mat,1) "Cannot project matrix"

  nrow = size(brow,2)
  ncol = size(bcol,2)
  Q = size(mat,2)
  proj = allocate_matrix(Matrix{Float},nrow*ncol,Q)

  @inbounds for q = 1:Q, jt = 1:ncol, it = 1:nrow
    proj[(jt-1)*nrow+it,q] = sum(brow[idx_forwards,it].*bcol[idx_backwards,jt].*mat[idx_forwards,q])
  end

  proj
end

function rb_spacetime_projection(rbrow::RBSpaceUnsteady,mat::AbstractMatrix)
  proj_space = rb_space_projection(rbrow,mat)
  proj_space_time = rb_time_projection(rbrow,proj_space')
  reshape(proj_space_time,:)
end

function rb_spacetime_projection(
  rbrow::RBSpaceUnsteady,
  rbcol::RBSpaceUnsteady,
  mats::Vector{SparseMatrixCSC{Float,Int}};
  idx_forwards=1:size(mat,1),
  idx_backwards=1:size(mat,1))

  nsrow,ntrow = get_ns(rbrow),get_nt(rbrow)
  nscol,ntcol = get_ns(rbcol),get_nt(rbcol)
  Nt = length(mats)

  proj_space = allocate_matrix(Matrix{Float},nsrow*nscol,Nt)
  @inbounds for n = 1:Nt
    proj_space[:,n] = rb_space_projection(rbrow,rbcol,mats[n])[:]
  end

  proj_space_time = rb_time_projection(rbrow,rbcol,proj_space';
    idx_forwards,idx_backwards)

  proj_spacetime = allocate_matrix(Matrix{Float},nsrow*ntrow,nscol*ntcol)
  @inbounds for is = 1:nscol, js = 1:nsrow
    proj_ij = proj_space_time[:,(is-1)*nsrow+js]
    copyto!(view(proj_spacetime,1+(js-1)*ntrow:js*ntrow,1+(is-1)*ntcol:is*ntcol),
            proj_ij)
  end

  proj_spacetime
end
