struct RBAffineDecomposition{Top,Ttr,Tad}
  op::RBVariable{Top,Ttr}
  affine_decomposition::Tad
end

function RBAffineDecomposition(
  op::RBVariable{Top,Ttr},
  affine_decomposition::Tad) where {Top,Ttr,Tad}

  RBAffineDecomposition{Top,Ttr,Tad}(op,affine_decomposition)
end

function RBAffineDecomposition(info::RBInfo,args...)
  if info.load_offline
    load(info,args...)
  else
    assemble_affine_decomposition(info,args...)
  end
end

function assemble_affine_decomposition(info::RBInfo,tt::TimeTracker,op::RBVariable,args...)
  tt.offline_time.assembly_time += @elapsed begin
    ad = assemble_affine_decomposition(info,op,args...)
  end
  save(info,ad,get_id(op))
  ad
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBLinVariable{Affine},
  args...)

  id = get_id(op)
  println("Linear operator $id is Affine: computing Φᵀ$id")

  ad = rb_space_projection(op)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBLinVariable{Top},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Linear operator $id is $Top:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  ad = MDEIM(info,op,args...)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBBilinVariable{Affine,<:TrialFESpace},
  args...)

  id = get_id(op)
  println("Bilinear operator $id is Affine and has no lifting: computing Φᵀ$(id)Φ")

  ad = rb_space_projection(op)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  args...) where Ttr

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is Affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  ad = rb_space_projection(op)
  op_lift = RBLiftVariable(op)
  ad_lift = MDEIM(info,op_lift,args...)
  RBAffineDecomposition(op,ad),RBAffineDecomposition(op_lift,ad_lift)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Top,<:TrialFESpace},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is $Top and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  ad = MDEIM(info,op,args...)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Top,Ttr},
  args...) where {Top,Ttr}

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id and its lifting are $Top:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  ad,ad_lift = MDEIM(info,op,args...)
  op_lift = RBLiftVariable(op)
  RBAffineDecomposition(op,ad),RBAffineDecomposition(op_lift,ad_lift)
end

get_op(rbs::RBAffineDecomposition) = rbs.op

get_affine_decomposition(rbs::RBAffineDecomposition) = rbs.affine_decomposition

function save(info::RBInfo,b,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  if info.save_offline
    save(path_id,b)
  end
end

function save(path::String,rbv::NTuple{2,RBAffineDecomposition})
  rbvar,rbvar_lift = rbv
  save(path,rbvar)
  path_lift = path*"_lift"
  create_dir!(path_lift)
  save(path_lift,rbvar_lift)
end

function save(
  path::String,
  rbv::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  os = get_affine_decomposition(rbv)
  save(joinpath(path,"basis_space"),os)
end

function save(path::String,rbv::RBAffineDecomposition)
  os = get_affine_decomposition(rbv)
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
    assemble_affine_decomposition(info,tt,op,args...)
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
  RBAffineDecomposition(op,os)
end

function load(
  info::RBInfo,
  op::RBLinVariable{Affine},
  ::Measure)

  id = get_id(op)
  println("Loading projected Affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,os)
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
  RBAffineDecomposition(op,os),RBAffineDecomposition(op_lift,os_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Top,<:TrialFESpace},
  meas::Measure) where Top

  id = get_id(op)
  println("Loading MDEIM structures for $Top variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBAffineDecomposition(op,os)
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
  RBAffineDecomposition(op,os),RBAffineDecomposition(op_lift,os_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,<:TrialFESpace},
  ::Measure)

  id = get_id(op)
  println("Loading projected Affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,os)
end

function eval_affine_decomposition(rbs::Tuple)
  eval_affine_decomposition.(expand(rbs))
end

function eval_affine_decomposition(rbs::RBAffineDecomposition)
  op = get_op(rbs)
  id = get_id(op)
  println("Evaluating the RB offline quantity for $id")

  eval_affine_decomposition(Val(issteady(op)),rbs)
end

function eval_affine_decomposition(
  ::Val{true},
  rbs::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  get_affine_decomposition(rbs)
end

function eval_affine_decomposition(::Val{true},rbs::RBAffineDecomposition)
  mdeim = get_affine_decomposition(rbs)
  get_basis_space(mdeim)
end

function eval_affine_decomposition(
  ::Val{false},
  rbs::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  op = get_op(rbs)
  ns_row = get_ns(get_rbspace_row(op))

  os = get_affine_decomposition(rbs)
  blocks(os,ns_row)
end

function eval_affine_decomposition(::Val{false},rbs::RBAffineDecomposition)
  op = get_op(rbs)
  ns_row = get_ns(get_rbspace_row(op))

  mdeim = get_affine_decomposition(rbs)
  os = get_basis_space(mdeim)
  blocks(os,ns_row)
end
