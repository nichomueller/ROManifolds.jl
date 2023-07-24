function fem_path(tpath::String)
  create_dir!(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function rom_path(tpath::String,ϵ::Float)
  rbpath = joinpath(tpath,"$ϵ")
  create_dir!(rbpath)
  rbpath
end

function rom_path(
  tpath::String,ϵ::Float;
  st_mdeim=false,fun_mdeim=false)

  @assert isdir(tpath) "Provide valid path for the current test"

  keyword = if !st_mdeim && !fun_mdeim
    "standard"
  else
    st = st_mdeim ? "st" : ""
    fun = fun_mdeim ? "fun" : ""
    st*fun
  end

  outermost_path = joinpath(tpath,"rom")
  outer_path = joinpath(outermost_path,keyword)
  rom_path = joinpath(outer_path,"$ϵ")
  create_dir!(rom_path)
  rom_path
end

function mesh_path(tpath::String,mesh::String)
  joinpath(get_parent_dir(tpath;nparent=3),"meshes/$mesh")
end

function set_labels!(model,bnd_info)
  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)
  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels,tags[i],bnds[i])
    end
  end
end

function get_discrete_model(
  mshpath::String,
  bnd_info::Dict)

  if !ispath(mshpath)
    mshpath_msh_format = mshpath[1:findall(x->x=='.',mshpath)[end]-1]*".msh"
    model_msh_format = GmshDiscreteModel(mshpath_msh_format)
    to_json_file(model_msh_format,mshpath)
  end
  model = DiscreteModelFromFile(mshpath)
  set_labels!(model,bnd_info)
  model
end
