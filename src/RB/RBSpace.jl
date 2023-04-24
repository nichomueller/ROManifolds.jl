abstract type RBSpace end

struct RBSpaceSteady <: RBSpace
  id::Symbol
  basis_space::EMatrix{Float}
end

function RBSpaceSteady(snaps::Snapshots;ϵ=1e-5,style=ReducedPOD())
  id = get_id(snaps)
  basis_space = rb_space(snaps;ϵ,style)
  RBSpaceSteady(id,basis_space)
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
  RBSpaceUnsteady(id,basis_space,basis_time)
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
  bs = Matrix(get_basis_space(rb))
  save(joinpath(path,"basis_space"),bs)
end

function save(path::String,rb::RBSpaceUnsteady)
  bs = Matrix(get_basis_space(rb))
  bt = Matrix(get_basis_time(rb))
  save(joinpath(path,"basis_space"),bs)
  save(joinpath(path,"basis_time"),bt)
end

function load(info::RBInfoSteady,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  basis_space = load(EMatrix{Float},joinpath(path_id,"basis_space"))
  RBSpaceSteady(id,basis_space)
end

function load(info::RBInfoUnsteady,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  basis_space = load(EMatrix{Float},joinpath(path_id,"basis_space"))
  basis_time = load(EMatrix{Float},joinpath(path_id,"basis_time"))
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
  Matrix((brow'*mat*bcol)[:])
end

function rb_time_projection(rbrow::RBSpaceUnsteady,mat::AbstractArray)
  brow = get_basis_time(rbrow)
  @assert size(brow,1) == size(mat,1) "Cannot project array"

  nrow = size(brow,2)
  Q = size(mat,2)
  proj = Matrix{Float}(undef,nrow,Q)

  for q = 1:Q
    for it = 1:nrow
      proj[it,q] = sum(brow[:,it].*mat[:,q])
    end
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
  proj = [Matrix{Float}(undef,nrow,ncol) for _ = 1:Q]

  for q = 1:Q
    for jt = 1:ncol
      for it = 1:nrow
        proj[q][it,jt] = sum(brow[idx_forwards,it].*bcol[idx_backwards,jt].*mat[idx_forwards,q])
      end
    end
  end

  proj
end

function rb_spacetime_projection(rbrow::RBSpaceUnsteady,mat::AbstractMatrix)
  proj_space = [rb_space_projection(rbrow,mat[:,i]) for i=axes(mat,2)]
  proj_spacetime_block = rb_time_projection(rbrow,Matrix(proj_space)')
  ns,nt = get_ns(rbrow),get_nt(rbrow)

  proj_spacetime = zeros(ns*nt)
  for i = 1:ns
    proj_spacetime[1+(i-1)*nt:i*nt] = proj_spacetime_block[i]
  end

  proj_spacetime
end

function rb_spacetime_projection(
  rbrow::RBSpaceUnsteady,
  rbcol::RBSpaceUnsteady,
  mat::Block;
  idx_forwards=1:size(mat,1),
  idx_backwards=1:size(mat,1))

  proj_space = [rb_space_projection(rbrow,rbcol,mat[i])[:] for i=eachindex(mat)]
  proj_spacetime_block = rb_time_projection(rbrow,rbcol,Matrix(proj_space)';
    idx_forwards=idx_forwards,idx_backwards=idx_backwards)
  nsrow,ntrow = get_ns(rbrow),get_nt(rbrow)
  nscol,ntcol = get_ns(rbcol),get_nt(rbcol)

  proj_spacetime = zeros(nsrow*ntrow,nscol*ntcol)
  for i = 1:nscol
    for j = 1:nsrow
      proj_spacetime[1+(j-1)*ntrow:j*ntrow,1+(i-1)*ntcol:i*ntcol] =
        proj_spacetime_block[(i-1)*nsrow+j]
    end
  end

  proj_spacetime
end
