abstract type RBSpace end

struct RBSpaceSteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
end

function RBSpaceSteady(
  id::Symbol,
  basis_space::NTuple{N,Matrix{Float}}) where N

  rbspace = ()
  for bs = basis_space
    rbspace = (rbspace...,RBSpaceSteady(id,bs))
  end
  rbspace
end

function RBSpaceSteady(
  snaps::Snapshots;ismdeim=Val(false),ϵ=1e-5)

  id = get_id(snaps)
  basis_space = POD(snaps,ismdeim;ϵ=ϵ)
  RBSpaceSteady(id,basis_space)
end

function RBSpaceSteady(
  snaps::NTuple{N,Snapshots};ismdeim=Val(false),ϵ=1e-5) where N
  Broadcasting(s->RBSpaceSteady(s;ismdeim=ismdeim,ϵ=ϵ))(snaps)
end

struct RBSpaceUnsteady <: RBSpace
  id::Symbol
  basis_space::Matrix{Float}
  basis_time::Matrix{Float}
end

function RBSpaceUnsteady(
  id::Symbol,
  basis_space::NTuple{N,Matrix{Float}},
  basis_time::NTuple{N,Matrix{Float}}) where N

  rbspace = ()
  for bst = zip(basis_space,basis_time)
    rbspace = (rbspace...,RBSpaceUnsteady(id,bst...))
  end
  rbspace
end

function RBSpaceUnsteady(
  snaps::Snapshots;ismdeim=Val(false),ϵ=1e-5)

  id = get_id(snaps)
  snaps2 = mode2(snaps)
  basis_space = POD(snaps,ismdeim;ϵ)
  basis_time = POD(snaps2,ismdeim;ϵ)
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function RBSpaceUnsteady(
  snaps::NTuple{N,Snapshots};ismdeim=Val(false),ϵ=1e-5) where N
  Broadcasting(s->RBSpaceUnsteady(s;ismdeim=ismdeim,ϵ=ϵ))(snaps)
end

function RBSpace(
  id::Symbol,
  basis_space::Matrix{Float})

  RBSpaceSteady(id,basis_space)
end

function RBSpace(
  id::Symbol,
  basis_space::Matrix{Float},
  basis_time::Matrix{Float})

  RBSpaceUnsteady(id,basis_space,basis_time)
end

get_id(rb::RBSpace) = rb.id
get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_time(rb::RBSpaceUnsteady) = rb.basis_time
get_basis_spacetime(rb::RBSpaceUnsteady) = kron(rb.basis_space,rb.basis_time)

save(info::RBInfo,rb::RBSpace) = if info.save_offline save(info.offline_path,rb) end

function save(path::String,rb::RBSpaceSteady)
  id = get_id(rb)
  save(joinpath(path,"basis_space_$id"),rb.basis_space)
end

function save(path::String,rb::RBSpaceUnsteady)
  id = get_id(rb)
  save(joinpath(path,"basis_space_$id"),rb.basis_space)
  save(joinpath(path,"basis_time_$id"),rb.basis_time)
end

function load_rb(info::RBInfoSteady,id::Symbol)
  path = info.offline_path
  basis_space = load(joinpath(path,"basis_space_$id"))
  RBSpaceSteady(id,basis_space)
end

function load_rb(info::RBInfoUnsteady,id::Symbol)
  path = info.offline_path
  basis_space = load(joinpath(path,"basis_space_$id"))
  basis_time = load(joinpath(path,"basis_time_$id"))
  RBSpaceUnsteady(id,basis_space,basis_time)
end

get_Ns(rb::RBSpace) = size(rb.basis_space,1)
get_ns(rb::RBSpace) = size(rb.basis_space,2)
get_Nt(rb::RBSpaceUnsteady) = size(rb.basis_time,1)
get_nt(rb::RBSpaceUnsteady) = size(rb.basis_time,2)
get_dims(rb::RBSpaceSteady) = get_Ns(rb),get_ns(rb)
get_dims(rb::RBSpaceUnsteady) = get_Ns(rb),get_ns(rb),get_Nt(rb),get_nt(rb)
