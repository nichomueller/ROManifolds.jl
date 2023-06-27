struct RBInfo
  ϵ::Float
  nsnaps::Int
  nsnaps_mdeim::Int
  fe_path::String
  offline_path::String
  online_path::String
  energy_norm::Bool
  load_offline::Bool
  save_offline::Bool
  save_online::Bool
  postprocess::Bool
end

function RBInfo(test_path::String;ϵ=1e-4,nsnaps=80,nsnaps_mdeim=20,
  energy_norm=false,st_mdeim=false,fun_mdeim=false,
  load_offline=false,save_offline=true,save_online=true,postprocess=false)

  fe_path = fem_path(test_path)
  offline_path,online_path = rom_off_on_paths(test_path,ϵ;st_mdeim,fun_mdeim)
  RBInfo(ϵ,nsnaps,nsnaps_mdeim,fe_path,offline_path,online_path,
    energy_norm,load_offline,save_offline,save_online,postprocess)
end

function save(info::RBInfo,objs::Tuple)
  map(obj->save(info,obj),expand(objs))
end

function load(types::Tuple,info::RBInfo)
  map(type->load(type,info),expand(types))
end

function save(info::RBInfo,snaps::Snapshots)
  if info.save_offline
    path = joinpath(info.fe_path,"fe_snaps")
    convert!(Matrix{Float},snaps)
    save(path,snaps)
  end
end

function save(info::RBInfo,params::Table)
  if info.save_offline
    path = joinpath(info.fe_path,"params")
    save(path,params)
  end
end

function load(T::Type{Snapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fe_snaps")
  s = load(T,path)
  convert!(EMatrix{Float},s)
end

function load(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params")
  load(T,path)[idx]
end
