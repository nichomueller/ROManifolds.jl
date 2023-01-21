abstract type RBStructure{Top,Ttr} end

struct RBAffineLinStructure <: RBStructure{Affine,nothing}
  op::RBLinVariable{Affine}
  basis::Matrix{Float}
end

struct RBLinStructure{Top} <: RBStructure{Top,nothing}
  op::RBLinVariable{Top}
  mdeim::MDEIM
end

struct RBAffineBilinStructure{Ttr} <: RBStructure{Affine,Ttr}
  op::RBBilinVariable{Affine,Ttr}
  basis::Matrix{Float}
end

struct RBBilinStructure{Top,Ttr} <: RBStructure{Top,Ttr}
  op::RBBilinVariable{Top,Ttr}
  mdeim::MDEIM
end

struct RBLiftStructure{Top,Ttr} <: RBStructure{Top,Ttr}
  op::RBLiftVariable{Top,Ttr}
  mdeim::MDEIM
end

function RBStructure(
  op::RBLinVariable{Affine},
  basis::Matrix{Float})

  RBAffineLinStructure(op,basis)
end

function RBStructure(
  op::RBLinVariable{Top},
  mdeim::MDEIM) where Top

  RBLinStructure{Top}(op,mdeim)
end

function RBStructure(
  op::RBBilinVariable{Affine,Ttr},
  basis::Matrix{Float}) where Ttr

  RBAffineBilinStructure{Ttr}(op,basis)
end

function RBStructure(
  op::RBBilinVariable{Top,Ttr},
  mdeim::MDEIM) where {Top,Ttr}

  RBBilinStructure{Top,Ttr}(op,mdeim)
end

function RBStructure(
  op::RBLiftVariable{Top,Ttr},
  mdeim::MDEIM) where {Top,Ttr}

  RBLiftStructure{Top,Ttr}(op,mdeim)
end

function RBStructure(info::RBInfo,args...)
  if info.load_offline
    load(info,args...)
  else
    assemble_rb_structure(info,args...)
  end
end

function assemble_rb_structure(info::RBInfo,tt::TimeTracker,op::RBVariable,args...)
  tt.offline_time.assembly_time += @elapsed begin
    rb_variable = assemble_rb_structure(info,op,args...)
  end
  save(info,rb_variable,get_id(op))
  rb_variable
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBLinVariable{Affine},
  args...)

  id = get_id(op)
  println("Linear operator $id is affine: computing Φᵀ$id")

  basis = rb_space_projection(op)
  RBStructure(op,basis)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBLinVariable,
  args...)

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Linear operator $id is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim = MDEIM(info,op,args...)
  RBStructure(op,mdeim)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinVariable{Affine,<:TrialFESpace},
  args...)

  id = get_id(op)
  println("Bilinear operator $id is affine and has no lifting: computing Φᵀ$(id)Φ")

  basis = rb_space_projection(op)
  RBStructure(op,basis)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  args...) where Ttr

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  basis = rb_space_projection(op)
  op_lift = RBLiftVariable(op)
  mdeim_lift = MDEIM(info,op_lift,args...)
  RBStructure(op,basis),RBStructure(op_lift,mdeim_lift)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinVariable{Top,<:TrialFESpace},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is non-affine and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim = MDEIM(info,op,args...)
  RBStructure(op,mdeim)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinVariable,
  args...)

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id and its lifting are non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim,mdeim_lift = MDEIM(info,op,args...)
  op_lift = RBLiftVariable(op)
  RBStructure(op,mdeim),RBStructure(op_lift,mdeim_lift)
end

get_op(rbs::RBStructure) = rbs.op

function get_offline_quantity(
  rbs::Union{RBAffineLinStructure,RBAffineBilinStructure})

  rbs.basis
end

function get_offline_quantity(rbs::RBStructure)
  rbs.mdeim
end

function save(info::RBInfo,b,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  if info.save_offline
    save(path_id,b)
  end
end

function save(path::String,rbv::NTuple{2,RBStructure})
  rbvar,rbvar_lift = rbv
  save(path,rbvar)
  path_lift = path*"_lift"
  create_dir!(path_lift)
  save(path_lift,rbvar_lift)
end

function save(
  path::String,
  rbv::Union{RBAffineLinStructure,RBAffineBilinStructure})

  basis = get_offline_quantity(rbv)
  save(joinpath(path,"basis_space"),basis)
end

function save(path::String,rbv::RBStructure)
  mdeim = get_offline_quantity(rbv)
  save(path,mdeim)
end

function load(
  info::RBInfo,
  tt::TimeTracker,
  op::RBVariable,
  args...)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")
  if ispath(path_id)
    _,meas,field = args
    load(info,op,getproperty(meas,field))
  else
    println("Failed to load variable $(id): running offline assembler instead")
    assemble_rb_structure(info,tt,op,args...)
  end

end

function load(
  info::RBInfo,
  op::RBLinVariable,
  meas::Measure)

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  mdeim = load(path_id,op,meas)
  RBStructure(op,mdeim)
end

function load(
  info::RBInfo,
  op::RBLinVariable{Affine},
  ::Measure)

  id = get_id(op)
  println("Loading projected affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  basis = load(joinpath(path_id,"basis_space"))
  RBStructure(op,basis)
end

function load(
  info::RBInfo,
  op::RBBilinVariable,
  meas::Measure)

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id and its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  mdeim = load(path_id,op,meas)
  op_lift = RBLiftVariable(op)
  mdeim_lift = load(path_id_lift,op,meas)
  RBStructure(op,mdeim),RBStructure(op_lift,mdeim_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Top,<:TrialFESpace},
  meas::Measure) where Top

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  mdeim = load(path_id,op,meas)
  RBStructure(op,mdeim)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  meas::Measure) where Ttr

  id = get_id(op)
  println("Loading projected affine variable $id and MDEIM structures for its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  basis = load(joinpath(path_id,"basis_space"))
  op_lift = RBLiftVariable(op)
  mdeim_lift = load(path_id_lift,op,meas)
  RBStructure(op,basis),RBStructure(op_lift,mdeim_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,<:TrialFESpace},
  ::Measure)

  id = get_id(op)
  println("Loading projected affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  basis = load(joinpath(path_id,"basis_space"))
  RBStructure(op,basis)
end
