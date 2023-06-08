struct RBInfo
  ϵ::Float
  nsnaps::Int
  mdeim_nsnaps::Int
  fe_path::String
  offline_path::String
  online_path::String
  energy_norm::Bool
  load_offline::Bool
  save_offline::Bool
  save_online::Bool
  postprocess::Bool
end

function RBInfo(tpath::String;ϵ=1e-4,nsnaps=80,mdeim_nsnaps=20,
  energy_norm=false,st_mdeim=false,fun_mdeim=false,
  load_offline=false,save_offline=true,save_online=true,postprocess=false)

  fe_path = fem_path(test_path)
  offline_path,online_path = rom_off_on_paths(tpath,ϵ;st_mdeim,fun_mdeim)
  RBInfo(ϵ,nsnaps,mdeim_nsnaps,fe_path,offline_path,online_path,
    energy_norm,load_offline,save_offline,save_online,postprocess)
end

function save(info::RBInfo,ref::Symbol,objs::Tuple)
  map(obj->save(info,ref,obj))(expand(objs))
end
