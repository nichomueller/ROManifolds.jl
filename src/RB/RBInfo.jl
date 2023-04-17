abstract type RBInfo end

struct RBInfoSteady <: RBInfo
  ptype::ProblemType
  ϵ::Float
  nsnap::Int
  online_snaps::UnitRange{Int64}
  mdeim_nsnap::Int
  offline_path::String
  online_path::String
  use_energy_norm::Bool
  load_offline::Bool
  save_offline::Bool
  save_online::Bool
  fun_mdeim::Bool
  adaptivity::Bool
  postprocess::Bool
end

function RBInfoSteady(
  ptype::ProblemType,
  tpath::String,
  mesh::String;
  ϵ=1e-5,nsnap=80,online_snaps=95:100,mdeim_snap=20,use_energy_norm=false,
  load_offline=true,save_offline=true,save_online=true,
  fun_mdeim=false,adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,tpath,ϵ;fun_mdeim=fun_mdeim)
  RBInfoSteady(ptype,ϵ,nsnap,online_snaps,mdeim_snap,offline_path,online_path,
    use_energy_norm,load_offline,save_offline,save_online,fun_mdeim,adaptivity,postprocess)
end

mutable struct RBInfoUnsteady <: RBInfo
  ptype::ProblemType
  ϵ::Float
  nsnap::Int
  online_snaps::UnitRange{Int64}
  mdeim_nsnap::Int
  offline_path::String
  online_path::String
  time_red_method::String
  use_energy_norm::Bool
  load_offline::Bool
  save_offline::Bool
  save_online::Bool
  st_mdeim::Bool
  fun_mdeim::Bool
  adaptivity::Bool
  postprocess::Bool
end

function RBInfoUnsteady(
  ptype::ProblemType,
  tpath::String,
  mesh::String;
  ϵ=1e-5,nsnap=80,online_snaps=95:100,mdeim_snap=20,time_red_method="ST-HOSVD",
  use_energy_norm=false,load_offline=true,
  save_offline=true,save_online=true,st_mdeim=false,fun_mdeim=false,
  adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,tpath,ϵ;
    st_mdeim=st_mdeim,fun_mdeim=fun_mdeim)
  RBInfoUnsteady(ptype,ϵ,nsnap,online_snaps,mdeim_snap,offline_path,
    online_path,time_red_method,use_energy_norm,load_offline,
    save_offline,save_online,st_mdeim,fun_mdeim,
    adaptivity,postprocess)
end

issteady(info::RBInfo) = issteady(info.ptype)

isindef(info::RBInfo) = isindef(info.ptype)

ispdomain(info::RBInfo) = ispdomain(info.ptype)

function save(info::RBInfo,args::Tuple)
  if info.save_offline
    Broadcasting(arg->save(info,arg))(expand(args))
  end
end
