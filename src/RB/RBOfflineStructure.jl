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

  RBNonaffineStructure{Ttr}(op,off_structure)
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
  op::RBLinVariable{Top},
  meas::Measure) where Top

  id = get_id(op)
  println("Loading MDEIM structures for $Top variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBLinVariable{Affine},
  ::Measure)

  id = get_id(op)
  println("Loading projected Affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Top,Ttr},
  meas::Measure) where {Top,Ttr}

  id = get_id(op)
  println("Loading MDEIM structures for $Top variable $id and its lifting")
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
  println("Loading MDEIM structures for $Top variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBOfflineStructure(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  meas::Measure) where Ttr

  id = get_id(op)
  println("Loading projected Affine variable $id and MDEIM structures for its lifting")
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
  ::Measure)

  id = get_id(op)
  println("Loading projected Affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBOfflineStructure(op,os)
end

function eval_off_structure(rbs::Tuple)
  eval_off_structure.(rbs)
end

function eval_off_structure(rbs::Tuple,bsθ::Tuple)
  @assert length(rbs) == length(bsθ) "Wrong length"
  eval_off_structure.(rbs,bsθ)
end

function eval_off_structure(rbs::RBOfflineStructure,args...)
  op = get_op(rbs)
  id = get_id(op)
  println("Evaluating the RB offline quantity for $id")

  eval_off_structure(Val(issteady(op)),rbs,args...)
end

function eval_off_structure(::Val{true},rbs::RBAffineStructure)
  get_offline_structure(rbs)
end

function eval_off_structure(::Val{true},rbs::RBNonaffineStructure)
  mdeim = get_offline_structure(rbs)
  get_basis_space(mdeim)
end

function eval_off_structure(::Val{true},rbs::RBNonlinearStructure)
  mdeim = get_offline_structure(rbs)
  bs = get_basis_space(mdeim)
  blocks(bs,size(bs,2))
end

function eval_off_structure(::Val{false},rbs::RBAffineStructure)
  op = get_op(rbs)
  ns_row = get_ns(get_rbspace_row(op))

  os = get_offline_structure(rbs)
  blocks(os,ns_row)
end

function eval_off_structure(::Val{false},rbs::RBNonaffineStructure)
  op = get_op(rbs)
  ns_row = get_ns(get_rbspace_row(op))

  mdeim = get_offline_structure(rbs)
  os = get_basis_space(mdeim)
  blocks(os,ns_row)
end

function eval_off_structure(
  val::Val{false},
  rbs::RBNonlinearStructure,
  rbspaceθ::RBSpaceUnsteady)

  op = get_op(rbs)
  eval_off_structure(val,Val(islinear(op)),rbs,rbspaceθ)
end

function eval_off_structure(
  ::Val{false},
  ::Val{true},
  rbs::RBNonlinearStructure,
  rbspaceθ::RBSpaceUnsteady)

  op = get_op(rbs)
  mdeim = get_offline_structure(rbs)
  bs = get_basis_space(mdeim)
  basis_block = blocks(bs,size(bs,2))

  Nt = get_Nt(op)
  idx = 1:Nt
  btθ = get_basis_time(get_rbspace_row(op))
  rbrow = get_rbspace_row(op)
  rbcol = rbspaceθ
  btbtbt = rb_time_projection(rbrow,rbcol,btθ,idx,idx)

  kron(btbtbt,basis_block)
end

function eval_off_structure(
  ::Val{false},
  ::Val{false},
  rbs::RBNonlinearStructure,
  rbspaceθ::RBSpaceUnsteady)

  op = get_op(rbs)
  mdeim = get_offline_structure(rbs)
  bs = get_basis_space(mdeim)
  ns = size(bs,2)
  basis_block = blocks(bs,ns)
  basis_blockT = Matrix.([bb' for bb = basis_block])

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt
  btθ = get_basis_time(rbspaceθ)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  btbtbt = rb_time_projection(rbrow,rbcol,btθ,idx,idx)
  btbtbt_shift = rb_time_projection(rbrow,rbcol,btθ,idx_forwards,idx_backwards)
  nt = length(btbtbt)

  bst = [kron(basis_blockT[space_idx(k,ns)],btbtbt[time_idx(k,ns)]) for k=1:ns*nt]
  bst_shift = [kron(basis_blockT[space_idx(k,ns)],btbtbt_shift[time_idx(k,ns)]) for k=1:ns*nt]

  bst,bst_shift
end
