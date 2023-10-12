function get_parent_dir(dir::String;nparent=1)
  dir = dir[1:findall(x->x=='/',dir)[end]-1]
  for _ = 1:nparent-1
    dir = dir[1:findall(x->x=='/',dir)[end]-1]
  end
  dir
end

"""Get a full list of subdirectories at a given root directory"""
function get_all_subdirectories(path::String)
  filter(isdir,readdir(path,join=true))
end

"""Get a full list of subdirectories at a given root directory"""
function get_all_subfiles(path::String)
  filter(isfile,readdir(correct_path(path),join=true))
end

"""Create a directory at the given path"""
function create_dir!(path::String)
  if !isdir(path)
    create_dir!(get_parent_dir(path))
    mkdir(path)
  end
  return
end

correct_path(path::String) = path*".txt"

function save(path::String,obj)
  serialize(correct_path(path),obj)
  return nothing
end

function load(path::String,::Type{T})::T where T
  obj = deserialize(correct_path(path))
  @assert typeof(obj) <: T
  return obj
end

function num_active_files(path::String)
  subd = get_all_subfiles(path)
  subd_no_extension = map(x->split(x,'.')[end-1],subd)
  suffix = map(x->last(split(x,'_')),subd_no_extension)
  count = 0
  for s in suffix
    sint = tryparse(Int,s)
    count = isnothing(sint) ? count : max(count,sint)
  end
  count
end

function num_active_dirs(path::String)
  subd = get_all_subdirectories(path)
  suffix = map(x->last(split(x,'_')),subd)
  count = 0
  for s in suffix
    sint = tryparse(Int,s)
    count = isnothing(sint) ? count : max(count,sint)
  end
  count
end
