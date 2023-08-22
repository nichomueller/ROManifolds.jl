struct RBInfo
  ϵ::Float
  nsnaps_state::Int
  nsnaps_system::Int
  fe_path::String
  rb_path::String
  energy_norm::Bool
  load_structures::Bool
  save_structures::Bool
  postprocess::Bool
end

function RBInfo(test_path::String;ϵ=1e-4,nsnaps_state=80,nsnaps_system=20,
  energy_norm=false,st_mdeim=false,fun_mdeim=false,
  load_structures=false,save_structures=true,postprocess=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim,fun_mdeim)
  RBInfo(ϵ,nsnaps_state,nsnaps_system,fe_path,rb_path,
    energy_norm,load_structures,save_structures,postprocess)
end

function get_fe_path(tpath::String)
  create_dir!(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function get_rb_path(
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

  outermost_path = joinpath(tpath,"rb")
  outer_path = joinpath(outermost_path,keyword)
  rb_path = joinpath(outer_path,"$ϵ")
  create_dir!(rb_path)
  rb_path
end

for (fsave,fload) in zip((:save,:save_test),(:load,:load_test))
  @eval begin
    function $fsave(info::RBInfo,objs::Tuple)
      map(obj->$fsave(info,obj),expand(objs))
    end

    function $fload(types::Tuple,info::RBInfo)
      map(type->$fload(type,info),expand(types))
    end
  end
end

function save(info::RBInfo,snaps::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps")
    convert!(Matrix{Float},snaps)
    save(path,snaps)
  end
end

function save(info::RBInfo,params::Table)
  if info.save_structures
    path = joinpath(info.fe_path,"params")
    save(path,params)
  end
end

function load(T::Type{Snapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps")
  s = load(T,path)
  s
end

function load(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params")
  load(T,path)
end

function save_test(info::RBInfo,snaps::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps_test")
    save(path,snaps)
  end
end

function save_test(info::RBInfo,params::Table)
  if info.save_structures
    path = joinpath(info.fe_path,"params_test")
    save(path,params)
  end
end

function load_test(T::Type{Snapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps_test")
  load(T,path)
end

function load_test(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params_test")
  load(T,path)
end
