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

correct_path(path::String) = path*".csv"

save(path::String,s) = writedlm(correct_path(path),s, ','; header=false)

load(path::String) = readdlm(correct_path(path), ',')

myisfile(path::String) = isfile(correct_path(path))

function generate_dcube_model(
  d::Int,
  npart::Int,
  path::String)

  @assert d ≤ 3 "Select d-dimensional domain, where d ≤ 3"
  if d == 1
    domain = (0,1)
    partition = (npart)
  elseif d == 2
    domain = (0,1,0,1)
    partition = (npart,npart)
  else
    domain = (0,1,0,1,0,1)
    partition = (npart,npart,npart)
  end
  model = CartesianDiscreteModel(domain,partition)
  to_json_file(model,path)
  return
end
