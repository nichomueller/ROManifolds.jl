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

"""Create a directory at the given path"""
function create_dir!(path::String)
  if !isdir(path)
    create_dir!(get_parent_dir(path))
    mkdir(path)
  end
  return
end

correct_path(path::String) = path*".txt"

save(path::String,obj) = serialize(correct_path(path),obj)

save(path::String,objs::Tuple) = Broadcasting(obj->save(path,obj))(expand(objs))

function load(::Type{T},path::String)::T where T
  obj = deserialize(correct_path(path))
  if typeof(obj) == T
    obj
  else
    convert(T,load(path))
  end
end

load(path::String) = load(Matrix{Float},path)

myisfile(path::String) = isfile(correct_path(path))
