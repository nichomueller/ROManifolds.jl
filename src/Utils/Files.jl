function get_parent_dir(dir::String)
  dir[1:findall(x->x=='/',dir)[end]-1]
end

"""Create a directory at the given path"""
function create_dir!(path::String)
  if !isdir(path)
    create_dir!(get_parent_dir(path))
    mkdir(path)
  end
  return
end

"""Get a full list of subdirectories at a given root directory"""
function get_all_subdirectories(path::String)
  filter(isdir,readdir(path,join=true))
end

correct_path(path::String) = path*".csv"
save(path::String,s) = writedlm(correct_path(path),s, ','; header=false)
load(path::String) = readdlm(correct_path(path), ',')
myisfile(path::String) = isfile(correct_path(path))

function save_structures_in_list(
  list_structures::Tuple,
  list_names::NTuple{D},
  path::String) where D

  @assert length(list_structures) == D "Wrong length"

  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in
    enumerate(list_structures) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i1,i2) in enumerate(l_idx)
      save_CSV(l_val[i1],joinpath(path, list_names[i2]*".csv"))
    end
  end

  return

end

function load_structures_in_list(
  list_names::Tuple{Vararg{String, D}},
  list_types::Tuple,
  path::String) where D

  @assert length(list_types) == D "Wrong length"

  ret_tuple = ()

  for (idx, name) in enumerate(list_names)
    ret_tuple = (ret_tuple...,
      load_CSV(list_types[idx],joinpath(path,name*".csv")))
  end

  ret_tuple

end

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
    domain = (0, 1, 0, 1, 0, 1)
    partition = (npart,npart,npart)
  end
  model = CartesianDiscreteModel(domain,partition)
  to_json_file(model,path)
  return
end
