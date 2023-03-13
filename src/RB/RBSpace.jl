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
  snaps::Snapshots;ϵ=1e-5)

  id = get_id(snaps)
  basis_space = POD(snaps;ϵ)
  RBSpaceSteady(id,basis_space)
end

function RBSpaceSteady(
  snaps::NTuple{N,Snapshots};ϵ=1e-5) where N
  Broadcasting(s->RBSpaceSteady(s;ϵ))(snaps)
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
  snaps::Snapshots;ϵ=1e-5)

  id = get_id(snaps)
  s,ns = get_snap(snaps),get_nsnap(snaps)
  basis_space = POD(s;ϵ)
  s2 = mode2_unfolding(basis_space'*s,ns)
  basis_time = POD(s2;ϵ)
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function RBSpaceUnsteady(
  snaps::NTuple{N,Snapshots};ϵ=1e-5) where N
  Broadcasting(s->RBSpaceUnsteady(s;ϵ))(snaps)
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
get_Ns(rb::RBSpace) = size(rb.basis_space,1)
get_ns(rb::RBSpace) = size(rb.basis_space,2)
get_Nt(rb::RBSpaceUnsteady) = size(rb.basis_time,1)
get_nt(rb::RBSpaceUnsteady) = size(rb.basis_time,2)

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
  mat::AbstractMatrix;
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

struct RBFEFunction{T<:CellField} <: FEFunction
  cell_field::T
  cell_dof_values::AbstractArray{<:AbstractVector{<:Number}}
  free_values::AbstractVector{<:Number}
  dirichlet_values::AbstractVector{<:Number}
  fe_space::SingleFieldFESpace

  function RBFEFunction(
    cell_field::T,
    cell_dof_values::AbstractArray{<:AbstractVector{<:Number}},
    free_values::AbstractVector{<:Number},
    dirichlet_values::AbstractVector{<:Number},
    fe_space::SingleFieldFESpace)

    new{T}(cell_field,cell_dof_values,free_values,dirichlet_values,fe_space)
  end
end

function Gridap.FEFunction(
  op::RBVariable,
  vec::AbstractVector)

  test = get_test(op)
  dir_vals = get_dirichlet_dof_values(test)
  cell_vals = scatter_free_and_dirichlet_values(test,vec,dir_vals)
  cell_field = CellField(test,cell_vals)
  RBFEFunction(cell_field,cell_vals,vec,dir_vals,test)
end

function Gridap.FEFunction(
  op::LagrangianQuadFESpace,
  mat::AbstractMatrix)

  n -> FEFunction(op,mat[:,n])
end
