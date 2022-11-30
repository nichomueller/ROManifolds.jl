abstract type RBSpace{T} end

struct RBSpaceSteady{T} <: RBSpace{T}
  id::Symbol
  basis_space::Matrix{T}

  function RBSpaceSteady(
    snaps::Snapshots{T};ϵ=1e-5) where T

    id = get_id(snaps)
    basis_space = POD(snaps,ϵ)
    new{T}(id,basis_space)
  end
end

struct RBSpaceUnsteady{T} <: RBSpace{T}
  id::Symbol
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpaceUnsteady(
    snaps::Snapshots{T};ϵ=1e-5) where T

    id = get_id(snaps)
    snaps2 = mode2(snaps)
    basis_space = POD(snaps,ϵ)
    basis_time = POD(snaps2,ϵ)
    new{T}(id,basis_space,basis_time)
  end
end

allocate_rbspace(id::Symbol,::Type{T}) where T = RBSpaceSteady(allocate_snapshot(id,T))
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
  rb = allocate_rbspace(id,T)
  off_path = info.offline_path
  load_rb!(rb,off_path)
end

function load_rb(info::RBSpaceUnsteady,id::Symbol)
  rb = allocate_rbspace(id,T)
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

abstract type RBVarOperator{Top,TT,Tsp} end

mutable struct RBLinOperator{Top,Tsp} <: RBVarOperator{Top,nothing,Tsp}
  feop::ParamLinOperator{Top}
  rbspace_row::Tsp

  function RBVarOperator(
    feop::ParamLinOperator{Top},
    rbspace_row::Tsp) where {Top,Tsp}

    new{Top,Tsp}(feop,rbspace_row)
  end
end

mutable struct RBBilinOperator{Top,TT,Tsp} <: RBVarOperator{Top,TT,Tsp}
  feop::ParamBilinOperator{Top,TT}
  rbspace_row::Tsp
  rbspace_col::Tsp

  function RBVarOperator(
    feop::ParamBilinOperator{Top,TT},
    rbspace_row::Tsp,
    rbspace_col::Tsp) where {Top,TT,Tsp}

    new{Top,TT,Tsp}(feop,rbspace_row,rbspace_col)
  end
end

get_background_feop(rbop::RBVarOperator) = rbop.feop
get_rbspace_row(rbop::RBVarOperator) = rbop.rbspace_row
get_rbspace_col(rbop::RBBilinOperator) = rbop.rbspace_col
get_basis_space_row(rbop::RBVarOperator) = get_basis_space(get_rbspace_row(rbop))
get_basis_space_col(rbop::RBVarOperator) = get_basis_space(get_rbspace_col(rbop))
get_basis_time_row(rbop::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT} =
  get_basis_time(get_rbspace_row(rbop))
get_basis_time_col(rbop::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT} =
  get_basis_time(get_rbspace_col(rbop))

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVarOperator,
  trian::Triangulation)
  get_cell_dof_ids(get_background_feop(rbop),trian)
end

Gridap.FESpaces.assemble_vector(op::RBLinOperator) = assemble_vector(op.feop)
Gridap.FESpaces.assemble_matrix(op::RBBilinOperator) = assemble_matrix(op.feop)

realization(op::RBVarOperator) = realization(get_pspace(op))

Gridap.Algebra.allocate_vector(op::RBLinOperator) = assemble_vector(op.feop)
Gridap.Algebra.allocate_matrix(op::RBBilinOperator) = assemble_matrix(op.feop)
allocate_structure(op::RBLinOperator) = allocate_vector(op)
allocate_structure(op::RBBilinOperator) = allocate_matrix(op)

function get_findnz_mapping(op::RBLinOperator)
  v = assemble_structure(op)
  collect(eachindex(v))
end

"Small, full vector -> large, sparse vector"
function get_findnz_mapping(op::RBBilinOperator)
  M = assemble_structure(op)
  first(findnz(M[:]))
end

"Viceversa"
function get_inverse_findnz_mapping(op::RBVarOperator)
  findnz_map = get_findnz_mapping(op)
  inv_map(i::Int) = findall(x -> x == i,findnz_map)[1]
  inv_map
end

function rb_projection(op::RBLinOperator{Affine,Tsp}) where Tsp
  id = get_id(op)
  println("Vector $id is affine: computing Φᵀ$id")

  feop = get_background_feop(op)
  vec = assemble_affine_vector(feop)
  rbspace_row = get_rbspace_row(op)

  rbspace_row'*vec
end

function rb_projection(op::RBBilinOperator{Affine,TT,Tsp}) where {TT,Tsp}
  id = get_id(op)
  println("Matrix $id is affine: computing Φᵀ$id Φ")

  feop = get_background_feop(op)
  mat = assemble_affine_matrix(feop)
  rbspace_row = get_rbspace_row(op)
  rbspace_col = get_rbspace_col(op)

  rbspace_row'*mat*rbspace_col
end

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
