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
  online_rhs::Bool
  load_offline::Bool
  save_offline::Bool
  save_online::Bool
  fun_mdeim::Bool
  adaptivity::Bool
  postprocess::Bool
end

function RBInfoSteady(
  ptype::ProblemType,
  mesh="cube5x5x5.json",
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes";
  ϵ=1e-5,nsnap=80,online_snaps=95:100,mdeim_snap=20,use_energy_norm=false,
  online_rhs=false,load_offline=true,save_offline=true,save_online=true,
  fun_mdeim=false,adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,root)
  RBInfoSteady(ptype,ϵ,nsnap,online_snaps,mdeim_snap,offline_path,online_path,
    use_energy_norm,online_rhs,load_offline,
    save_offline,save_online,fun_mdeim,adaptivity,postprocess)
end

struct TimeInfo
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_dt(ti::TimeInfo) = ti.dt
get_Nt(ti::TimeInfo) = Int((ti.tF-ti.t0)/ti.dt)
get_θ(ti::TimeInfo) = ti.θ
get_timesθ(ti::TimeInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ

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
  online_rhs::Bool
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
  mesh="cube5x5x5.json",
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes";
  ϵ=1e-5,nsnap=80,online_snaps=95:100,mdeim_snap=20,time_red_method="ST-HOSVD",
  use_energy_norm=false,online_rhs=false,load_offline=true,
  save_offline=true,save_online=true,st_mdeim=true,fun_mdeim=false,
  adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,root)
  RBInfoUnsteady(ptype,ϵ,nsnap,online_snaps,mdeim_snap,offline_path,
    online_path,time_red_method,use_energy_norm,online_rhs,load_offline,
    save_offline,save_online,st_mdeim,fun_mdeim,
    adaptivity,postprocess)
end

issteady(info::RBInfo) = issteady(info.ptype)
isindef(info::RBInfo) = isindef(info.ptype)
ispdomain(info::RBInfo) = ispdomain(info.ptype)
