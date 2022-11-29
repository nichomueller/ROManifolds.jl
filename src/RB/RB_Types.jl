abstract type RBSpace{T} end

struct RBSpaceSteady{T} <: RBSpace{T}
  snaps::Snapshots{T}
  basis_space::Matrix{T}

  function RBSpaceSteady(
    snaps::Snapshots{T},
    basis_space::Matrix{T}) where T

    new{T}(snaps,basis_space)
  end
end

struct RBSpaceUnsteady{T} <: RBSpace{T}
  snaps::Snapshots{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpaceUnsteady(
    snaps::Snapshots{T},
    basis_space::Matrix{T},
    basis_time::Matrix{T}) where T

    new{T}(snaps,basis_space,basis_time)
  end
end

function allocate_rbspace_steady(id::Symbol,::Type{T}) where T
  esnap = allocate_snapshot(id,T)
  emat = allocate_matrix(T)
  RBSpaceSteady(esnap,emat)
end

function allocate_rbspace_unsteady(id::Symbol,::Type{T}) where T
  esnap = allocate_snapshot(id,T)
  emat = allocate_matrix(T)
  RBSpaceUnsteady(esnap,emat,emat)
end

get_snaps(rb::RBSpace) = rb.snaps
get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_time(rb::RBSpaceUnsteady) = rb.basis_time
get_basis_spacetime(rb::RBSpaceUnsteady) = kron(rb.basis_space,rb.basis_time)
correct_path(rb::RBSpace,path::String) = correct_path(rb.snaps,path)

function save(rb::RBSpaceSteady,info::RBInfoSteady)
  off_path = info.offline_path
  save(rb.basis_space,correct_path(rb,joinpath(off_path,"basis_space_")))
end

function save(rb::RBSpaceUnsteady,info::RBInfoUnsteady)
  off_path = info.offline_path
  save(rb.basis_space,correct_path(rb,joinpath(off_path,"basis_space_")))
  save(rb.basis_time,correct_path(rb,joinpath(off_path,"basis_time_")))
end

function load_rb!(rb::RBSpaceSteady,path::String)
  load_snap!(rb.snaps,path)
  rb.basis_space = load(correct_path(rb,joinpath(path,"basis_space_")))
  rb
end

function load_rb!(rb::RBSpaceUnsteady,path::String)
  load_snap!(rb.snaps,path)
  rb.basis_space = load(correct_path(rb,joinpath(path,"basis_space_")))
  rb.basis_time = load(correct_path(rb,joinpath(path,"basis_time_")))
  rb
end

function load_rb(info::RBSpaceSteady,id::Symbol)
  rb = allocate_rbspace_steady(id,T)
  off_path = info.offline_path
  load_rb!(rb,off_path)
end

function load_rb(info::RBSpaceUnsteady,id::Symbol)
  rb = allocate_rbspace_unsteady(id,T)
  off_path = info.offline_path
  load_rb!(rb,off_path)
end

function rb_spatial_subspace(rb::RBSpaceUnsteady{T}) where T
  RBSpaceSteady{T}(rb.snap,rb.basis_space)
end

function allocate_rb_solution(rb::RBSpaceSteady{T}) where T
  ns = get_ns(rb)
  Vector{T}(undef,ns)
end

function allocate_rb_solution(rb::RBSpaceUnsteady{T}) where T
  ns,nt = get_ns(rb),get_nt(rb)
  Vector{T}(undef,ns*nt)
end

get_Ns(rb::RBSpace) = size(rb.basis_space,1)
get_ns(rb::RBSpace) = size(rb.basis_space,2)
get_Nt(rb::RBSpaceUnsteady) = size(rb.basis_time,1)
get_nt(rb::RBSpaceUnsteady) = size(rb.basis_time,2)
get_dims(rb::RBSpaceSteady) = get_Ns(rb),get_ns(rb)
get_dims(rb::RBSpaceUnsteady) = get_Ns(rb),get_ns(rb),get_Nt(rb),get_nt(rb)

abstract type RBInfo end

struct RBInfoSteady <: RBInfo
  ptype::ProblemType
  ϵ::Float
  nsnap::Int
  mdeim_nsnap::Int
  offline_path::String
  online_path::String
  use_energy_norm::Bool
  online_rhs::Bool
  get_offline_structures::Bool
  save_offline::Bool
  save_online::Bool
  adaptivity::Bool
  postprocess::Bool
end

mutable struct RBInfoUnsteady <: RBInfo
  ptype::ProblemType
  time_info::TimeInfo
  ϵ::Float
  nsnap::Int
  mdeim_nsnap::Int
  offline_path::String
  online_path::String
  time_red_method::String
  use_energy_norm::Bool
  online_rhs::Bool
  get_offline_structures::Bool
  save_offline::Bool
  save_online::Bool
  st_mdeim::Bool
  fun_mdeim::Bool
  adaptivity::Bool
  postprocess::Bool
end

function RBInfo(
  ptype::ProblemType,
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes";
  ϵ=1e-5,nsnap=80,mdeim_snap=20,use_energy_norm=false,online_rhs=false,
  get_offline_structures=true,save_offline=true,save_online=true,
  adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,root)
  RBInfoSteady(ptype,offline_path,online_path,ϵ,nsnap,mdeim_snap,
    use_energy_norm,online_rhs,get_offline_structures,
    save_offline,save_online,adaptivity,postprocess)
end

function RBInfo(
  ptype::ProblemType,
  time_info::TimeInfo,
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes";
  ϵ=1e-5,nsnap=80,mdeim_snap=20,time_red_method="ST-HOSVD",
  use_energy_norm=false,online_rhs=false,get_offline_structures=true,
  save_offline=true,save_online=true,st_mdeim=true,fun_mdeim=false,
  adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,root)
  RBInfoUnsteady(ptype,time_info,offline_path,online_path,ϵ,nsnap,mdeim_snap,
    time_red_method,use_energy_norm,online_rhs,get_offline_structures,
    save_offline,save_online,st_mdeim,fun_mdeim,
    adaptivity,postprocess)
end

issteady(info::RBInfo) = issteady(info.ptype)
isindef(info::RBInfo) = isindef(info.ptype)
ispdomain(info::RBInfo) = ispdomain(info.ptype)
save(rb::RBSpace,info::RBInfo) = if info.save_offline save(rb,info.offline_path) end

mutable struct RBResults
  offline_time::Float
  online_time::Float
  err::Float
  pointwise_err::Matrix{Float}

  function RBResults()
    new(0.,0.,0.,Matrix{Float}(undef,0,0))
  end
end

time_dict(r::RBResults) = Dict("offline_time"=>r.offline_time,"online_time"=>r.online_time)
err_dict(r::RBResults) = Dict("err"=>r.err,"pointwise_err"=>r.pointwise_err)

function save(rbinfo::RBInfo,r::RBResults)
  on_path = rbinfo.online_path
  save(time_dict(r),joinpath(on_path,"times"))
  save(err_dict(r),joinpath(on_path,"errors"))
end
