abstract type RBSpace end

struct RBSpaceSteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
end

function RBSpaceSteady(
  id::NTuple{N,Symbol},
  basis_space::NTuple{N,Matrix{Float}}) where N

  RBSpaceSteady.(id,basis_space)
end

function RBSpaceSteady(
  snaps::Snapshots;ismdeim=Val(false),ϵ=1e-5)

  id = get_id(snaps)
  basis_space = POD(snaps,ismdeim;ϵ=ϵ)
  RBSpaceSteady(id,basis_space)
end

function RBSpaceSteady(
  snaps::NTuple{N,Snapshots};ismdeim=Val(false),ϵ=1e-5) where N
  Broadcasting(s->RBSpaceSteady(s;ismdeim=ismdeim,ϵ=ϵ))(snaps)
end

struct RBSpaceUnsteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
  basis_time::Matrix{Float}
end

function RBSpaceUnsteady(
  id::NTuple{N,Symbol},
  basis_space::NTuple{N,Matrix{Float}},
  basis_time::NTuple{N,Matrix{Float}}) where N

  RBSpaceUnsteady.(id,basis_space,basis_time)
end

function RBSpaceUnsteady(
  snaps::Snapshots;ismdeim=Val(false),ϵ=1e-5)

  id = get_id(snaps)
  snaps2 = mode2_unfolding(snaps)
  basis_space = POD(snaps,ismdeim;ϵ)
  basis_time = POD(snaps2,ismdeim;ϵ)
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function RBSpaceUnsteady(
  snaps::NTuple{N,Snapshots};ismdeim=Val(false),ϵ=1e-5) where N
  Broadcasting(s->RBSpaceUnsteady(s;ismdeim=ismdeim,ϵ=ϵ))(snaps)
end

function RBSpace(
  id::Symbol,
  basis_space::Matrix{Float})

  RBSpaceSteady(id,basis_space)
end

function RBSpace(
  id::Symbol,
  basis_space::Matrix{Float},
  basis_time::Matrix{Float})

  RBSpaceUnsteady(id,basis_space,basis_time)
end

get_id(rb::RBSpace) = rb.id
get_id(rb::NTuple{N,RBSpace}) where N = get_id.(rb)
get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_space(rb::NTuple{N,RBSpace}) where N = get_basis_space.(rb)
get_basis_time(rb::RBSpaceUnsteady) = rb.basis_time
get_basis_time(rb::NTuple{N,RBSpaceUnsteady}) where N = get_basis_time.(rb)
get_basis_spacetime(rb::RBSpaceUnsteady) = kron(rb.basis_time,rb.basis_space)
get_Ns(rb::RBSpace) = size(rb.basis_space,1)
get_ns(rb::RBSpace) = size(rb.basis_space,2)
get_Nt(rb::RBSpaceUnsteady) = size(rb.basis_time,1)
get_nt(rb::RBSpaceUnsteady) = size(rb.basis_time,2)
get_dims(rb::RBSpaceSteady) = get_Ns(rb),get_ns(rb)
get_dims(rb::RBSpaceUnsteady) = get_Ns(rb),get_ns(rb),get_Nt(rb),get_nt(rb)

function save(info::RBInfo,rb::RBSpace)
  id = get_id(rb)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  if info.save_offline
    save(path_id,rb)
  end
end

function save(path::String,rb::RBSpaceSteady)
  save(joinpath(path,"basis_space"),rb.basis_space)
end

function save(path::String,rb::RBSpaceUnsteady)
  save(joinpath(path,"basis_space"),rb.basis_space)
  save(joinpath(path,"basis_time"),rb.basis_time)
end

function load_rb(info::RBInfoSteady,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  basis_space = load(joinpath(path_id,"basis_space"))
  RBSpaceSteady(id,basis_space)
end

function load_rb(info::RBInfoUnsteady,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  basis_space = load(joinpath(path_id,"basis_space"))
  basis_time = load(joinpath(path_id,"basis_time"))
  RBSpaceUnsteady(id,basis_space,basis_time)
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

  btp_fun(it,q) = sum(brow[:,it].*mat[:,q])
  btp_fun(q) = Matrix(Broadcasting(it -> btp_fun(it,q))(1:nrow))
  btp_fun.(1:Q)
end

function rb_time_projection(
  rbrow::RBSpaceUnsteady,
  rbcol::RBSpaceUnsteady,
  mat::AbstractMatrix,
  idx_forwards=1:size(mat,1),
  idx_backwards=1:size(mat,1))

  brow = get_basis_time(rbrow)
  bcol = get_basis_time(rbcol)
  @assert size(brow,1) == size(bcol,1) == size(mat,1) "Cannot project matrix"

  nrow = size(brow,2)
  ncol = size(bcol,2)
  Q = size(mat,2)

  time_proj_fun(it,jt,q) =
    sum(brow[idx_forwards,it].*bcol[idx_backwards,jt].*mat[idx_forwards,q])
  time_proj_fun(jt,q) = Broadcasting(it -> time_proj_fun(it,jt,q))(1:nrow)
  time_proj_fun(q) = Broadcasting(jt -> time_proj_fun(jt,q))(1:ncol)
  Matrix.(time_proj_fun.(1:Q))
end
