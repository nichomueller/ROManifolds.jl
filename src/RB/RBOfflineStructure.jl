abstract type RBOfflineStructure{Top,Ttr} end

struct RBAffineStructure{Ttr} <: RBOfflineStructure{Affine,Ttr}
  op::RBVariable{Affine,Ttr}
  off_structure::Matrix{Float}
end

struct RBNonaffineStructure{Ttr} <: RBOfflineStructure{Nonaffine,Ttr}
  op::RBVariable{Nonaffine,Ttr}
  off_structure::MDEIM
end

struct RBNonlinearStructure{Ttr} <: RBOfflineStructure{Nonlinear,Ttr}
  op::RBVariable{Nonlinear,Ttr}
  off_structure::MDEIM
end

function RBOfflineStructure(
  op::RBVariable{Affine,Ttr},
  off_structure::Matrix{Float}) where Ttr

  RBAffineStructure{Ttr}(op,off_structure)
end

function RBOfflineStructure(
  op::RBVariable{Nonaffine,Ttr},
  off_structure::MDEIM) where Ttr

  RBNonaffineStructure{Top,Ttr}(op,off_structure)
end

function RBOfflineStructure(
  op::RBVariable{Nonlinear,Ttr},
  off_structure::MDEIM) where Ttr

  RBNonlinearStructure{Ttr}(op,off_structure)
end

function RBOfflineStructure(info::RBInfo,args...)
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

  os = rb_space_projection(op)
  RBOfflineStructure(op,os)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBLinVariable,
  args...)

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Linear operator $id is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  os = MDEIM(info,op,args...)
  RBOfflineStructure(op,os)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinVariable{Affine,<:TrialFESpace},
  args...)

  id = get_id(op)
  println("Bilinear operator $id is affine and has no lifting: computing Φᵀ$(id)Φ")

  os = rb_space_projection(op)
  RBOfflineStructure(op,os)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  args...) where Ttr

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  os = rb_space_projection(op)
  op_lift = RBLiftVariable(op)
  os_lift = MDEIM(info,op_lift,args...)
  RBOfflineStructure(op,os),RBOfflineStructure(op_lift,os_lift)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinVariable{Top,<:TrialFESpace},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is non-affine and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  os = MDEIM(info,op,args...)
  RBOfflineStructure(op,os)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinVariable,
  args...)

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id and its lifting are non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  os,os_lift = MDEIM(info,op,args...)
  op_lift = RBLiftVariable(op)
  RBOfflineStructure(op,os),RBOfflineStructure(op_lift,os_lift)
end

get_op(rbs::RBOfflineStructure) = rbs.op

get_offline_structure(rbs::RBOfflineStructure) = rbs.off_structure

function save(info::RBInfo,b,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  if info.save_offline
    save(path_id,b)
  end
end

function save(path::String,rbv::NTuple{2,RBOfflineStructure})
  rbvar,rbvar_lift = rbv
  save(path,rbvar)
  path_lift = path*"_lift"
  create_dir!(path_lift)
  save(path_lift,rbvar_lift)
end

function save(path::String,rbv::RBAffineStructure)
  os = get_offline_structure(rbv)
  save(joinpath(path,"basis_space"),os)
end

function save(path::String,rbv::RBOfflineStructure)
  os = get_offline_structure(rbv)
  save(path,os)
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

  os = load(path_id,op,meas)
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBLinVariable{Affine},
  args...)

  id = get_id(op)
  println("Loading projected affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable,
  meas::Measure)

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id and its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  os = load(path_id,op,meas)
  op_lift = RBLiftVariable(op)
  os_lift = load(path_id_lift,op,meas)
  RBOfflineStructure(op,os),RBOfflineStructure(op_lift,os_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Top,<:TrialFESpace},
  meas::Measure) where Top

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  meas::Measure) where Ttr

  id = get_id(op)
  println("Loading projected affine variable $id and MDEIM structures for its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  os = load(joinpath(path_id,"basis_space"))
  op_lift = RBLiftVariable(op)
  os_lift = load(path_id_lift,op,meas)
  RBOfflineStructure(op,os),RBOfflineStructure(op_lift,os_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,<:TrialFESpace},
  args...)

  id = get_id(op)
  println("Loading projected affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Nonlinear,<:TrialFESpace},
  args...)

  id = get_id(op)
  println("Loading projected nonlinear variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Nonlinear,Ttr},
  args...) where Ttr

  id = get_id(op)
  println("Loading projected nonlinear variable $id and its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  os = load(joinpath(path_id,"basis_space"))
  os_lift = load(joinpath(path_id_lift,"basis_space"))
  RBOfflineStructure(op,os),RBOfflineStructure(op_lift,os_lift)
end

function eval_off_structure(rbs::RBOfflineStructure,args...)
  get_offline_structure(rbs)
end

function eval_off_structure(
  rbs::RBNonlinearStructure,
  rbspaceθ::RBSpaceUnsteady)

  op = get_op(rbs)
  mdeim = get_offline_structure(rbs)
  bs = get_basis_space(mdeim)
  basis_block = blocks(bs,size(bs,2))

  btθ = get_basis_time(rbspaceθ)
  rbrow = get_rbspace_row(op)
  btbtbt = rb_time_projection(rbrow,rbspaceθ,btθ)

  kron(btbtbt,basis_block)
end

function eval_off_structure(
  rbs::RBNonlinearStructure,
  rbspaceθ::RBSpaceUnsteady)

  op = get_op(rbs)
  mdeim = get_offline_structure(rbs)
  bs = get_basis_space(mdeim)
  basis_block = blocks(bs,size(bs,2))

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt
  btθ = get_basis_time(rbspaceθ)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  btbtbt = rb_time_projection(rbrow,rbcol,btθ,idx,idx)
  btbtbt_shift = rb_time_projection(rbrow,rbcol,btθ,idx_forwards,idx_backwards)

  bst = kron(btbtbt,basis_block)
  bst_shift = kron(btbtbt_shift,basis_block)

  bst,bst_shift
end
