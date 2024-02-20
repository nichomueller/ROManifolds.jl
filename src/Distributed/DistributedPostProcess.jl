get_ind_part_filename(info::RBInfo) = info.dir * "/index_partition.jld"
get_gids_filename(info::RBInfo) = info.dir * "/gids.jld"

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
  row_partition = s.snaps.row_partition
  map(local_views(s),local_views(row_partition)) do s,row_partition
    part = part_id(row_partition)
    i_part_filename = get_part_filename(i_filename,part)
    s_part_filename = get_part_filename(s_filename,part)
    serialize(i_part_filename,row_partition)
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
  row_partition = distribute(i_parts)
  snaps_partition = distribute(s_parts)
  psnaps = PMatrix(snaps_partition,row_partition)
  DistributedSnapshots(psnaps)
end

get_fields_filename(info::RBInfo) = info.dir * "/fields.jld"
get_free_values_filename(info::RBInfo) = info.dir * "/free_values.jld"

function DrWatson.save(info::RBInfo,uh::DistributedCellField)
  i_filename = get_ind_part_filename(info)
  fields_filename = get_fields_filename(info)
  free_values_filename = get_free_values_filename(info)
  index_partition = uh.metadata.free_values.index_partition
  map(uh.fields,local_views(uh.metadata.free_values),index_partition) do fields,free_values,index_partition
    part = part_id(index_partition)
    i_part_filename = get_part_filename(i_filename,part)
    fields_part_filename = get_part_filename(fields_filename,part)
    free_values_part_filename = get_part_filename(free_values_filename,part)
    serialize(i_part_filename,index_partition)
    serialize(fields_part_filename,fields)
    serialize(free_values_part_filename,free_values)
  end
end

function load_distributed_cell_field(distribute,info::RBInfo)
  i_filename = get_ind_part_filename(info)
  fields_filename = get_fields_filename(info)
  free_values_filename = get_free_values_filename(info)
  i_parts,fields_parts,free_values_parts = map(readdir(info.dir;join=true)) do dir
    part = parse(Int,dir[end])
    i_part_filename = get_part_filename(i_filename,part)
    fields_part_filename = get_part_filename(fields_filename,part)
    free_values_part_filename = get_part_filename(free_values_filename,part)
    deserialize(i_part_filename),deserialize(fields_part_filename),deserialize(free_values_part_filename)
  end |> tuple_of_arrays
  fields = distribute(fields_parts)
  index_partition = distribute(i_parts)
  free_values_partition = distribute(free_values_parts)
  free_values = PVector(free_values_partition,index_partition)
  DistributedCellField(fields,free_values)
end
