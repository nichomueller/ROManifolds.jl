function RB.get_norm_matrix(
  info::RBInfo,
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    RB.get_norm_matrix(info,trial,test)
  end
end

struct DistributedTransientSnapshots{T}
  snaps::T
end

DistributedSnapshots(snaps) = DistributedTransientSnapshots(snaps)

GridapDistributed.local_views(s::DistributedTransientSnapshots) = local_views(s.snaps)

function RB.Snapshots(
  values::PVector{P},
  args...) where {P<:AbstractParamContainer}

  index_partition = values.index_partition
  snaps = map(local_views(values)) do values
    Snapshots(values,args...)
  end
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.Snapshots(
  values::AbstractVector{<:PVector{P}},
  args...) where {P<:AbstractParamContainer}

  index_partition = first(values).index_partition
  parts = map(part_id,index_partition)
  snaps = map(parts) do part
    vals_part = Vector{P}(undef,length(values))
    for (k,v) in enumerate(values)
      map(local_views(v),index_partition) do val,ip
        if part_id(ip) == part
          vals_part[k] = val
        end
      end
    end
    Snapshots(vals_part,args...)
  end
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.Snapshots(a::DistributedArrayContribution,args...)
  b = Contribution(IdDict{DistributedTriangulation,DistributedTransientSnapshots}())
  for (trian,values) in a.dict
    b[trian] = Snapshots(values,args...)
  end
  b
end

function RB.get_realization(s::DistributedTransientSnapshots)
  s1 = PartitionedArrays.getany(local_views(s.snaps))
  get_realization(s1)
end

function RB.get_values(s::DistributedTransientSnapshots)
  snaps = map(local_views(s)) do s
    get_values(s)
  end
  index_partition = s.snaps.index_partition
  PVector(snaps,index_partition)
end

function RB.select_snapshots(s::DistributedTransientSnapshots,args...;kwargs...)
  snaps = map(local_views(s)) do s
    select_snapshots(s,args...;kwargs...)
  end
  index_partition = s.snaps.index_partition
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.reverse_snapshots(s::DistributedTransientSnapshots)
  snaps = map(local_views(s)) do s
    reverse_snapshots(s)
  end
  index_partition = s.snaps.index_partition
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

const DistributedTransientNnzSnapshots = DistributedTransientSnapshots{T} where {
  T<:Union{<:PSparseMatrix,AbstractVector{<:PSparseMatrix}}}

function RB.Snapshots(
  values::PSparseMatrix{P},
  args...) where {P<:AbstractParamContainer}

  row_partition = values.row_partition
  col_partition = values.col_partition
  snaps = map(local_views(values)) do values
    Snapshots(values,args...)
  end
  psnaps = PSparseMatrix(snaps,row_partition,col_partition)
  DistributedSnapshots(psnaps)
end

function RB.get_values(s::DistributedTransientNnzSnapshots)
  snaps = map(local_views(s)) do s
    get_values(s)
  end
  row_partition = s.snaps.row_partition
  col_partition = s.snaps.col_partition
  PSparseMatrix(snaps,row_partition,col_partition)
end

function RB.select_snapshots(s::DistributedTransientNnzSnapshots,args...;kwargs...)
  snaps = map(local_views(s)) do s
    select_snapshots(s,args...;kwargs...)
  end
  row_partition = s.snaps.row_partition
  col_partition = s.snaps.col_partition
  psnaps = PSparseMatrix(snaps,row_partition,col_partition)
  DistributedSnapshots(psnaps)
end

function RB.reverse_snapshots(s::DistributedTransientNnzSnapshots)
  snaps = map(local_views(s)) do s
    reverse_snapshots(s)
  end
  row_partition = s.snaps.row_partition
  col_partition = s.snaps.col_partition
  psnaps = PSparseMatrix(snaps,row_partition,col_partition)
  DistributedSnapshots(psnaps)
end

const DistributedRBSpace = RBSpace{S,BS,BT} where {S<:DistributedFESpace,BS,BT}

function RB.reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  dtrial = _to_distributed_fe_space(trial)
  test = get_test(feop)
  norm_matrix = RB.get_norm_matrix(info,feop)
  basis_space,basis_time = map(
    local_views(dtrial),
    local_views(test),
    local_views(s),
    local_views(norm_matrix)
    ) do trial,test,s,norm_matrix

    soff = select_snapshots(s,RB.offline_params(info))
    reduced_basis(soff,norm_matrix;ϵ=RB.get_tol(info))
  end |> tuple_of_arrays

  index_partition = s.snaps.index_partition
  p_basis_space = PMatrix(basis_space,index_partition)
  reduced_trial = RBSpace(trial,p_basis_space,basis_time)
  reduced_test = RBSpace(test,p_basis_space,basis_time)
  return reduced_trial,reduced_test
end

function RB.reduced_basis(s::DistributedTransientSnapshots,args...;kwargs...)
  basis_space,basis_time = map(local_views(s)) do s
    reduced_basis(s,args...;kwargs...)
  end |> tuple_of_arrays
  index_partition = s.snaps.index_partition
  p_basis_space = PMatrix(basis_space,index_partition)
  return p_basis_space,basis_time
end

function GridapDistributed._find_vector_type(
  spaces::AbstractVector{<:RBSpace},gids)
  T = get_vector_type(PartitionedArrays.getany(spaces))
  if isa(gids,PRange)
    vector_type = typeof(PVector{T}(undef,partition(gids)))
  else
    vector_type = typeof(BlockPVector{T}(undef,gids))
  end
  return vector_type
end

function GridapDistributed.local_views(r::DistributedRBSpace)
  map(
    local_views(r.space),
    local_views(get_basis_space(r)),
    local_views(get_basis_time(r))
    ) do space,basis_space,basis_time

    RBSpace(space,basis_space,basis_time)
  end
end

function RB.compress(r::DistributedRBSpace,s::DistributedTransientSnapshots)
  map(local_views(r),local_views(s)) do r,s
    compress(r,s)
  end
end

function RB.compress(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  s::DistributedTransientSnapshots;
  kwargs...)

  map(local_views(trial),local_views(test),local_views(s)) do trial,test,s
    compress(trial,test,s;kwargs...)
  end
end

function RB.recast(r::DistributedRBSpace,red_x::AbstractVector)
  vector_partition = map(local_views(r),local_views(red_x)) do r,red_x
    recast(r,red_x)
  end
  index_partition = partition(r.space.gids)
  PVector(vector_partition,index_partition)
end

function RB.compress_basis_space(A::PMatrix,test::RBSpace)
  basis_test = get_basis_space(test)
  map(eachcol(A)) do a
    basis_test'*a
  end
end

function RB.compress_basis_space(A::PMatrix,trial::RBSpace,test::RBSpace)
  basis_test = get_basis_space(test)
  basis_trial = get_basis_space(trial)
  map(get_values(A)) do A
    basis_test'*A*basis_trial
  end
end

function RB.combine_basis_time(
  trial::DistributedRBSpace,
  test::DistributedRBSpace;
  kwargs...)

  map(local_views(trial),local_views(test)) do trial,test
    RB.combine_basis_time(trial,test;kwargs...)
  end
end

function RB.mdeim(
  info::RBInfo,
  fs::DistributedFESpace,
  trian::DistributedTriangulation,
  basis_space::AbstractMatrix,
  basis_time::AbstractMatrix)

  lu_interp,red_trian,integration_domain = map(
    local_views(fs),local_views(trian),local_views(basis_space),local_views(basis_time)
    ) do fs,trian,basis_space,basis_time
    mdeim(info,fs,trian,basis_space,basis_time)
  end |> tuple_of_arrays
  d_red_trian = DistributedTriangulation(red_trian)
  return lu_interp,d_red_trian,integration_domain
end

function RB.reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  c::Contribution{DistributedTriangulation})

  info = RB.get_info(solver)
  a = distributed_array_contribution()
  for (trian,values) in c.dict
    RB.reduced_vector_form!(a,info,op,values,trian)
  end
  return a
end

function RB.reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  c::Contribution{DistributedTriangulation};
  kwargs...)

  info = RB.get_info(solver)
  a = distributed_array_contribution()
  for (trian,values) in c.dict
    RB.reduced_matrix_form!(a,info,op,values,trian;kwargs...)
  end
  return a
end

# post process

get_ind_part_filename(info::RBInfo) = info.dir * "/index_partition.jld"

function get_dir_part(dir::AbstractString,part::Integer)
  dir_part = joinpath(dir,"part_$(part)")
  FEM.create_dir(dir_part)
  dir_part
end

function get_part_filename(filename::AbstractString,part::Integer)
  _filename,extension = splitext(filename)
  dir,varname = splitdir(_filename)
  dir_part = get_dir_part(dir,part)
  joinpath(dir_part,varname*extension)
end

function DrWatson.save(info::RBInfo,s::DistributedTransientSnapshots)
  i_filename = get_ind_part_filename(info)
  s_filename = RB.get_snapshots_filename(info)
  index_partition = s.snaps.index_partition
  map(local_views(s),local_views(index_partition)) do s,index_partition
    part = part_id(index_partition)
    i_part_filename = get_part_filename(i_filename,part)
    s_part_filename = get_part_filename(s_filename,part)
    serialize(i_part_filename,index_partition)
    serialize(s_part_filename,s)
  end
end

function load_distributed_snapshots(distribute,info::RBInfo)
  i_filename = get_ind_part_filename(info)
  s_filename = RB.get_snapshots_filename(info)
  i_parts,s_parts = map(readdir(info.dir;join=true)) do dir
    part = parse(Int,dir[end])
    i_part_filename = get_part_filename(i_filename,part)
    s_part_filename = get_part_filename(s_filename,part)
    deserialize(i_part_filename),deserialize(s_part_filename)
  end |> tuple_of_arrays
  index_partition = distribute(i_parts)
  snaps_partition = distribute(s_parts)
  psnaps = PVector(snaps_partition,index_partition)
  DistributedSnapshots(psnaps)
end
