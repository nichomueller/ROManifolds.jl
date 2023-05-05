abstract type RBSpace end

struct RBSpaceSteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
end

function RBSpaceSteady(snaps::Snapshots;ϵ=1e-5,style=ReducedPOD())
  id = get_id(snaps)
  basis_space = rb_space(snaps;ϵ,style)
  RBSpaceSteady(id,Matrix(basis_space))
end

struct RBSpaceUnsteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
  basis_time::Matrix{Float}
end

function RBSpaceUnsteady(snaps::Snapshots;ϵ=1e-5,style=ReducedPOD())
  id = get_id(snaps)
  basis_space = rb_space(snaps;ϵ,style)
  basis_time = rb_time(snaps,basis_space;ϵ,style)
  RBSpaceUnsteady(id,Matrix(basis_space),Matrix(basis_time))
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
  proj = zeros(nrow,Q)

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
  proj = zeros(nrow*ncol,Q)

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

  proj_space = zeros(nsrow*nscol,Nt)
  @inbounds for n = 1:Nt
    proj_space[:,n] = rb_space_projection(rbrow,rbcol,mats[n])[:]
  end

  proj_space_time = rb_time_projection(rbrow,rbcol,proj_space';
    idx_forwards,idx_backwards)

  proj_spacetime = zeros(nsrow*ntrow,nscol*ntcol)
  @inbounds for it = 1:ntcol, jt = 1:ntrow
    proj_ij = proj_space_time[(jt-1)*ntrow+it,:]
    copyto!(view(proj_spacetime,1+(js-1)*ntrow:js*ntrow,1+(is-1)*ntcol:is*ntcol),
            proj_ij)
  end

  proj_spacetime
end
