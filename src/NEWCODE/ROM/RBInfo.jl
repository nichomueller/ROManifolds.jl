struct RBInfo
  ϵ::Float
  nsnaps_state::Int
  nsnaps_system::Int
  fe_path::String
  rb_path::String
  energy_norm::Bool
  load_offline::Bool
  save_offline::Bool
  save_online::Bool
  postprocess::Bool
end

function RBInfo(test_path::String;ϵ=1e-4,nsnaps_state=80,nsnaps_system=20,
  energy_norm=false,st_mdeim=false,fun_mdeim=false,
  load_offline=false,save_offline=true,save_online=true,postprocess=false)

  fe_path = fem_path(test_path)
  rb_path = rom_path(test_path,ϵ;st_mdeim,fun_mdeim)
  RBInfo(ϵ,nsnaps_state,nsnaps_system,fe_path,rb_path,
    energy_norm,load_offline,save_offline,save_online,postprocess)
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

function save(info::RBInfo,snaps::GenericSnapshots)
  if info.save_offline
    path = joinpath(info.fe_path,"fesnaps")
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

function load(T::Type{GenericSnapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps")
  s = load(T,path)
  convert!(EMatrix{Float},s)
  s
end

function load(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params")
  load(T,path)
end

function save_test(info::RBInfo,snaps::GenericSnapshots)
  if info.save_offline
    path = joinpath(info.fe_path,"fesnaps_test")
    save(path,snaps)
  end
end

function save_test(info::RBInfo,params::Table)
  if info.save_offline
    path = joinpath(info.fe_path,"params_test")
    save(path,params)
  end
end

function load_test(T::Type{GenericSnapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps_test")
  load(T,path)
end

function load_test(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params_test")
  load(T,path)
end
