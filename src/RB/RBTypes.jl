get_nsnap(v::AbstractVector) = length(v)
get_nsnap(m::AbstractMatrix) = size(m)[2]

mutable struct Snapshots{T}
  id::Symbol
  snap::AbstractArray{T}
  nsnap::Int
end

function Snapshots(id::Symbol,snap::AbstractArray{T}) where T
  nsnap = get_nsnap(snap)
  Snapshots{T}(id,snap,nsnap)
end

function Snapshots(id::Symbol,blocks::Vector{<:AbstractArray{T}}) where T
  snap = Matrix(blocks)
  nsnap = get_nsnap(blocks)
  Snapshots{T}(id,snap,nsnap)
end

Snapshots(s::Snapshots,idx) = Snapshots(s.id,getindex(s.snap,idx))

allocate_snapshot(id::Symbol,::Type{T}) where T = Snapshots(id,allocate_vblock(T))

get_id(s::Snapshots) = s.id
get_snap(s::Snapshots) = s.snap
get_nsnap(s::Snapshots) = s.nsnap

save(path::String,s::Snapshots) = save(joinpath(path,"$(s.id)"),s.snap)

function load_snap(path::String,id::Symbol)
  s = load(joinpath(path,"$(id)"))
  Snapshots(id,s)
end

get_Nt(s::Snapshots) = get_Nt(get_snap(s),get_nsnap(s))
mode2_unfolding(s::Snapshots) = mode2_unfolding(get_snap(s),get_nsnap(s))
POD(s::Snapshots,args...) = POD(s.snap,args...)
POD(s::Vector{Snapshots},args...) = Broadcasting(si->POD(si,args...))(s)

abstract type RBSpace{T} end

struct RBSpaceSteady{T} <: RBSpace{T}
  id::Symbol
  basis_space::AbstractArray{T}
end

function RBSpaceSteady(
  id::Symbol,
  basis_space::NTuple{N,AbstractArray{T}}) where {N,T}

  rbspace = ()
  for bs = basis_space
    rbspace = (rbspace...,RBSpaceSteady{T}(id,bs))
  end
  rbspace
end

function RBSpaceSteady(
  snaps::Snapshots{T};ϵ=1e-5) where T

  id = get_id(snaps)
  basis_space = POD(snaps,ϵ)
  RBSpaceSteady{T}(id,basis_space)
end

function RBSpaceSteady(
  snaps::NTuple{N,Snapshots{T}};ϵ=1e-5) where {N,T}
  Broadcasting(s->RBSpaceSteady(s;ϵ=ϵ))(snaps)
end

struct RBSpaceUnsteady{T} <: RBSpace{T}
  id::Symbol
  basis_space::AbstractArray{T}
  basis_time::AbstractArray{T}
end

function RBSpaceUnsteady(
  id::Symbol,
  basis_space::NTuple{N,AbstractArray{T}},
  basis_time::NTuple{N,AbstractArray{T}}) where {N,T}

  rbspace = ()
  for bst = zip(basis_space,basis_time)
    rbspace = (rbspace...,RBSpaceUnsteady{T}(id,bst...))
  end
  rbspace
end

function RBSpaceUnsteady(
  snaps::Snapshots{T};ϵ=1e-5) where T

  id = get_id(snaps)
  snaps2 = mode2(snaps)
  basis_space = POD(snaps,ϵ)
  basis_time = POD(snaps2,ϵ)
  RBSpaceUnsteady{T}(id,basis_space,basis_time)
end

function RBSpaceUnsteady(
  snaps::NTuple{N,Snapshots{T}};ϵ=1e-5) where {N,T}
  Broadcasting(s->RBSpaceUnsteady(s;ϵ=ϵ))(snaps)
end

function RBSpace(
  id::Symbol,
  basis_space::Matrix{T}) where T

  RBSpaceSteady{T}(id,basis_space)
end

function RBSpace(
  id::Symbol,
  basis_space::Matrix{T},
  basis_time::Matrix{T}) where T

  RBSpaceUnsteady{T}(id,basis_space,basis_time)
end

allocate_rbspace(id::Symbol,::Type{T}) where T = RBSpaceSteady(allocate_snapshot(id,T))
allocate_rbspace(rb::RBSpaceSteady{T}) where T = RBSpaceSteady{T}(rb.id,rb.basis_space)
allocate_rbspace(rb::RBSpaceUnsteady{T}) where T =
  RBSpaceUnsteady{T}(rb.id,rb.basis_space,rb.basis_time)
get_id(rb::RBSpace) = rb.id
get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_time(rb::RBSpaceUnsteady) = rb.basis_time
get_basis_spacetime(rb::RBSpaceUnsteady) = kron(rb.basis_space,rb.basis_time)

function save(path::String,rb::RBSpaceSteady,id::Symbol)
  save(path*"basis_space_$id",rb.basis_space)
end

function save(path::String,rb::RBSpaceUnsteady,id::Symbol)
  save(path*"basis_space_$id",rb.basis_space)
  save(path*"basis_time_$id",rb.basis_time)
end

function load!(rb::RBSpaceSteady,path::String,id::Symbol)
  rb.basis_space = load(path*"basis_space_$id")
  rb
end

function load!(rb::RBSpaceUnsteady,path::String,id::Symbol)
  rb.basis_space = load(path*"basis_space_$id")
  rb.basis_time = load(path*"basis_time_$id")
  rb
end

function load_rb(path::String,id::Symbol,T=Float)
  rb = allocate_rbspace(id,T)
  load!(rb,path,T)
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
end

mutable struct RBBilinOperator{Top,TT,Tsp} <: RBVarOperator{Top,TT,Tsp}
  feop::ParamBilinOperator{Top,TT}
  rbspace_row::Tsp
  rbspace_col::Tsp
end

function RBVarOperator(
  feop::ParamLinOperator{Top},
  rbspace_row::Tsp) where {Top,Tsp}

  RBLinOperator{Top,Tsp}(feop,rbspace_row)
end

function RBVarOperator(
  feop::ParamBilinOperator{Top,TT},
  rbspace_row::Tsp,
  rbspace_col::Tsp) where {Top,TT,Tsp}

  RBBilinOperator{Top,TT,Tsp}(feop,rbspace_row,rbspace_col)
end

get_background_feop(rbop::RBVarOperator) = rbop.feop
get_rbspace_row(rbop::RBVarOperator) = rbop.rbspace_row
get_rbspace_col(rbop::RBBilinOperator) = rbop.rbspace_col
get_id(rbop::RBBilinOperator) = get_id(get_background_feop(rbop))
get_basis_space_row(rbop::RBVarOperator) = get_basis_space(get_rbspace_row(rbop))
get_basis_space_col(rbop::RBVarOperator) = get_basis_space(get_rbspace_col(rbop))
get_basis_time_row(rbop::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT} =
  get_basis_time(get_rbspace_row(rbop))
get_basis_time_col(rbop::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT} =
  get_basis_time(get_rbspace_col(rbop))
Gridap.FESpaces.get_test(op::RBVarOperator) = get_test(op.feop)
Gridap.FESpaces.get_trial(op::RBBilinOperator) = get_trial(op.feop)

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVarOperator,
  trian::Triangulation)
  get_cell_dof_ids(get_background_feop(rbop),trian)
end

Gridap.FESpaces.assemble_vector(op::RBLinOperator) = assemble_vector(op.feop)
Gridap.FESpaces.assemble_matrix(op::RBBilinOperator) = assemble_matrix(op.feop)
assemble_lifting(op::RBBilinOperator) = assemble_lifting(op.feop)
assemble_matrix_and_lifting(op::RBBilinOperator) = assemble_matrix_and_lifting(op.feop)

get_pspace(op::RBVarOperator) = get_pspace(op.feop)
realization(op::RBVarOperator) = realization(get_pspace(op))

Gridap.Algebra.allocate_vector(op::RBLinOperator) = assemble_vector(op.feop)
Gridap.Algebra.allocate_matrix(op::RBBilinOperator) = assemble_matrix(op.feop)
allocate_structure(op::RBLinOperator) = allocate_vector(op)
allocate_structure(op::RBBilinOperator) = allocate_matrix(op)

function assemble_affine_vector(
  op::RBLinOperator{Affine,<:RBSpaceSteady})
  assemble_vector(op)(realization(op))
end

function assemble_affine_vector(
  op::RBLinOperator{Affine,<:RBSpaceUnsteady})
  assemble_vector(op)(realization(op),first(get_timesθ(op)))
end

function assemble_affine_matrix(
  op::RBBilinOperator{Affine,TT,<:RBSpaceSteady}) where TT
  assemble_matrix(op)(realization(op))
end

function assemble_affine_matrix(
  op::RBBilinOperator{Affine,TT,<:RBSpaceUnsteady}) where TT
  assemble_matrix(op)(realization(op),first(get_timesθ(op)))
end

function assemble_affine_matrix_and_lifting(
  op::RBBilinOperator{Affine,TT,<:RBSpaceSteady}) where TT
  assemble_matrix_and_lifting(op)(realization(op))
end

function assemble_affine_matrix_and_lifting(
  op::RBBilinOperator{Affine,TT,<:RBSpaceUnsteady}) where TT
  assemble_matrix_and_lifting(op)(realization(op),first(get_timesθ(op)))
end

function save(path::String,mat_and_lift::NTuple{2,AbstractArray})
  mat,lift = mat_and_lift
  save(path,mat)
  save(path*"_lift",lift)
end

function get_findnz_mapping(op::RBLinOperator)
  v = assemble_vector(op)(realization(op))
  collect(eachindex(v))
end

"Small, full vector -> large, sparse vector"
function get_findnz_mapping(op::RBBilinOperator)
  M = assemble_matrix(op)(realization(op))
  first(findnz(M[:]))
end

"Viceversa"
function get_inverse_findnz_mapping(op::RBVarOperator)
  findnz_map = get_findnz_mapping(op)
  inv_map(i::Int) = findall(x -> x == i,findnz_map)[1]
  inv_map
end

function unfold_spacetime(
  op::RBVarOperator{Top,TT,<:RBSpaceUnsteady},
  vals::AbstractVector{Tv}) where {Top,TT,Tv}

  Ns = get_Ns(op)
  Nt = get_Nt(op)
  @assert size(vals,1) == Ns*Nt "Wrong space-time dimensions"

  space_vals = Matrix{Tv}(reshape(vals,Ns,Nt))
  time_vals = Matrix{Tv}(reshape(vals,Nt,Ns))
  space_vals,time_vals
end

function unfold_spacetime(
  op::RBVarOperator{Top,TT,<:RBSpaceUnsteady},
  vals::AbstractMatrix{Tv}) where {Top,TT,Tv}

  unfold_vec(k::Int) = unfold_spacetime(op,vals[:,k])
  vals = Broadcasting(unfold_vec)(axis(vals,2))
  Matrix(first.(vals)),Matrix(last.(vals))
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
  println("Matrix $id and its lifting are affine: computing Φᵀ$(id)Φ and Φᵀlift_$id")

  feop = get_background_feop(op)
  mat,lift = assemble_affine_matrix_and_lifting(feop)
  rbspace_row = get_rbspace_row(op)
  rbspace_col = get_rbspace_col(op)

  rbspace_row'*mat*rbspace_col,rbspace_row'*lift
end

struct TimeInfo
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_dt(ti::TimeInfo) = ti.dt
get_θ(ti::TimeInfo) = ti.θ
get_timesθ(ti::TimeInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ

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

mutable struct RBInfoUnsteady <: RBInfo
  ptype::ProblemType
  time_info::TimeInfo
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
  time_info::TimeInfo,
  mesh="cube5x5x5.json",
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes";
  ϵ=1e-5,nsnap=80,online_snaps=95:100,mdeim_snap=20,time_red_method="ST-HOSVD",
  use_energy_norm=false,online_rhs=false,load_offline=true,
  save_offline=true,save_online=true,st_mdeim=true,fun_mdeim=false,
  adaptivity=false,postprocess=false)

  offline_path,online_path = rom_off_on_paths(ptype,mesh,root)
  RBInfoUnsteady(ptype,time_info,ϵ,nsnap,online_snaps,mdeim_snap,offline_path,
    online_path,time_red_method,use_energy_norm,online_rhs,load_offline,
    save_offline,save_online,st_mdeim,fun_mdeim,
    adaptivity,postprocess)
end

issteady(info::RBInfo) = issteady(info.ptype)
isindef(info::RBInfo) = isindef(info.ptype)
ispdomain(info::RBInfo) = ispdomain(info.ptype)
save(info::RBInfo,args...) = if info.save_offline save(info.offline_path,args...) end
load(info::RBInfo,args...) = if info.load_offline load(info.offline_path,args...) end
load_snap(info::RBInfo,args...) = if info.load_offline load_snap(info.offline_path,args...) end
load_rb(info::RBInfo,args...) = if info.load_offline load_rb(info.offline_path,args...) end
load_mdeim(info::RBInfo,args...) = if info.load_offline load_mdeim(info.offline_path,args...) end
