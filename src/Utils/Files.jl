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

#= function save_CSV(var, path::String)
  writedlm(correct_path(path), var, ','; header=false)
end

function save_CSV(var::Vector{Matrix{T}}, path::String) where T
  vec_string = ["$(var[i])" for i = eachindex(var)]
  mod_vec_string = Broadcasting(str -> replace(str, ";;" => ""))(vec_string)
  save_CSV(mod_vec_string, path)
end

function save_CSV(var::Vector{<:AbstractArray{T}}, path::Vector{String}) where T
  @assert length(var) == length(path) "length(var) not equals to length(path)"
  Broadcasting(save_CSV)(var, path)
  return
end

function save_CSV(var::SparseMatrixCSC, path::String)
  writedlm(correct_path(path), findnz(var), ','; header=false)
end

function save_CSV(var::SparseVector, path::String)
  writedlm(correct_path(path), findnz(var), ','; header=false)
end

function load_CSV(::Array{T,D}, path::String) where {T,D}
  var = readdlm(correct_path(path), ',', T)
  var_ret = D == 1 ? var[:,1] : var
  var_ret::Array{T,D}
end

function load_CSV(::Vector{Vector{T}}, path::String) where T
  mat_any = Matrix{T}(readdlm(correct_path(path), ',')')
  matrix_to_vecblocks(mat_any)::Vector{Vector{T}}
end

function load_CSV(::Vector{Matrix{T}}, path::String) where T
  mat_any = readdlm(correct_path(path), ',')
  function to_matrix(i::Int)
    vec_substring = split(chop(mat_any[i]; head=1, tail=1), "; ")
    vecvec_substring = Broadcasting(y -> split(y, " "))(vec_substring)
    vecvec = Broadcasting(y->parse.(T, y))(vecvec_substring)
    Matrix{T}(blocks_to_matrix(vecvec)')
  end
  Broadcasting(to_matrix)(eachindex(mat_any))::Vector{Matrix{T}}
end

function load_CSV(::Vector{Matrix{T}}, path::Vector{String}) where T
  Broadcasting(p -> load_CSV(Matrix{T}(undef,0,0), p))(path)::Vector{Matrix{T}}
end

function load_CSV(::SparseMatrixCSC{T, Int}, path::String) where T
  ijv = readdlm(correct_path(path), ',')
  sparse(Int.(ijv[1,:]), Int.(ijv[2,:]), T.(ijv[3,:]))::SparseMatrixCSC{T, Int}
end

function load_CSV(::SparseVector{T, Int}, path::String) where T
  iv = readdlm(correct_path(path), ',')
  sparse(Int.(iv[1,:]), T.(iv[2,:]))::SparseVector{T, Int}
end =#

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
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(path, list_names[i₂]*".csv"))
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
      load_CSV(list_types[idx], joinpath(path, name*".csv")))
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
