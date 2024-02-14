function FEM.get_L2_norm_matrix(
  info::RBInfo,
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    get_L2_norm_matrix(info,trial,test)
  end
end

function FEM.get_H1_norm_matrix(
  info::RBInfo,
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    get_H1_norm_matrix(info,trial,test)
  end
end

struct DistributedTransientSnapshots{T<:PVector{<:AbstractTransientSnapshots}}
  snaps::T
end

function DistributedSnapshots(snaps::PVector{<:AbstractTransientSnapshots})
  DistributedTransientSnapshots(snaps)
end

GridapDistributed.local_views(s::DistributedTransientSnapshots) = local_views(s.snaps)

function RB.Snapshots(
  values::AbstractVector{<:PVector{V}},
  args...) where V

  index_partition = first(values).index_partition
  parts = map(part_id,index_partition)
  snaps = map(parts) do part
    vals_part = Vector{V}(undef,length(values))
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

struct DistributedRBSpace{T<:AbstractVector{<:RBSpace}}
  spaces::T
end

GridapDistributed.local_views(a::DistributedRBSpace) = a.spaces

function RB.reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  ϵ = get_tol(info)
  norm_matrix = get_norm_matrix(info,feop)
  reduced_trial,reduced_test = map(
    local_views(trial),
    local_views(test),
    local_views(s),
    local_views(norm_matrix)
    ) do trial,test,s,norm_matrix

    soff = select_snapshots(s,offline_params(info))
    basis_space,basis_time = reduced_basis(soff,norm_matrix;ϵ)
    reduced_trial = TrialRBSpace(trial,basis_space,basis_time)
    reduced_test = TestRBSpace(test,basis_space,basis_time)
    reduced_trial,reduced_test
  end |> tuple_of_arrays

  dtrial = DistributedRBSpace(reduced_trial)
  dtest = DistributedRBSpace(reduced_test)
  return dtrial,dtest
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

function RB.compress(r::DistributedRBSpace,xmat::ParamVector{<:AbstractMatrix})
  partition = xmat.index_partition
  vector_partition = map(local_views(r),local_views(xmat)) do r,xmat
    compress(r,xmat)
  end
  PVector(vector_partition,partition)
end

function RB.compress(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  xmat::ParamVector{<:AbstractMatrix};
  kwargs...)

  partition = xmat.index_partition
  vector_partition = map(local_views(trial),local_views(test),local_views(xmat)
    ) do trial,test,xmat
    compress(trial,test,xmat;kwargs...)
  end
  PVector(vector_partition,partition)
end

function RB.recast(r::RBSpace,red_x::ParamVector{<:AbstractMatrix})
  partition = red_x.index_partition
  vector_partition = map(red_x) do red_x
    recast(r,red_x)
  end
  PVector(vector_partition,partition)
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

# function get_dir_parts(dir::AbstractString)
#   _path, = splitext(dir)
#   path, = splitdir(_path)
#   parts = Int[]
#   for p in readdir(path,join=true)
#     ppath, = splitext(p)
#     if ppath[end-1] == '_'
#       try
#         i = parse(Int,ppath[end])
#         push!(parts,i)
#       catch
#       end
#     end
#   end
#   sort!(parts)
#   map(parts) do part
#     get_dir_part(dir,part)
#   end
# end

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
  PVector(snaps_partition,index_partition)
end
