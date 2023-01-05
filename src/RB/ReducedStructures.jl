function rb_structure(info::RBInfo,args...)
  if info.load_offline
    load_rb_structure(info,args...)
  else
    assemble_rb_structure(info,args...)
  end
end

function assemble_rb_structure(info::RBInfo,tt::TimeTracker,op::RBVarOperator,args...)
  tt.offline_time += @elapsed begin
    rb_variable = assemble_rb_structure(info,op,args...)
  end
  save_rb_structure(info,rb_variable,get_id(op))
  rb_variable
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBLinOperator,
  args...)

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Linear operator $id is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBLinOperator{Affine},
  args...)

  id = get_id(op)
  println("Linear operator $id is affine: computing Φᵀ$id")

  rb_space_projection(op)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinOperator,
  args...)

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id and its lifting are non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinOperator{Top,<:TrialFESpace},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is non-affine and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinOperator{Affine,Ttr},
  args...) where Ttr

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  op_lift = RBLiftingOperator(op)
  rb_space_projection(op),mdeim_offline(info,op_lift,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinOperator{Affine,<:TrialFESpace},
  args...)

  id = get_id(op)
  println("Bilinear operator $id is affine and has no lifting: computing Φᵀ$(id)Φ")

  rb_space_projection(op)
end

function save_rb_structure(info::RBInfo,b,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  if info.save_offline
    save_rb_structure(path_id,b)
  end
end

function save_rb_structure(path::String,rbv::Tuple)
  rbvar,rbvar_lift = rbv
  save_rb_structure(path,rbvar)
  path_lift = path*"_lift"
  create_dir!(path_lift)
  save_rb_structure(path_lift,rbvar_lift)
end

function save_rb_structure(path::String,basis::Matrix{Float})
  save(joinpath(path,"basis_space"),basis)
end

function save_rb_structure(path::String,mdeim::MDEIMSteady)
  save(joinpath(path,"basis_space"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  red_lu = get_red_lu_factors(mdeim)
  save(joinpath(path,"LU"),red_lu.factors)
  save(joinpath(path,"p"),red_lu.ipiv)
end

function save_rb_structure(path::String,mdeim::MDEIMUnsteady)
  save(joinpath(path,"basis_space"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  save(joinpath(path,"basis_time"),get_basis_time(mdeim))
  save(joinpath(path,"idx_time"),get_idx_time(mdeim))
  red_lu = get_red_lu_factors(mdeim)
  save(joinpath(path,"LU"),red_lu.factors)
  save(joinpath(path,"p"),red_lu.ipiv)
end

function load_rb_structure(
  info::RBInfo,
  tt::TimeTracker,
  op::RBVarOperator,
  args...)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")
  if ispath(path_id)
    _,meas,field = args
    load_rb_structure(info,op,getproperty(meas,field))
  else
    println("Failed to load variable $(id): running offline assembler instead")
    assemble_rb_structure(info,tt,op,args...)
  end

end

function load_rb_structure(
  info::RBInfo,
  op::RBLinOperator,
  meas::Measure)

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  load_mdeim(path_id,op,meas)
end

function load_rb_structure(
  info::RBInfo,
  op::RBLinOperator{Affine},
  ::Measure)

  id = get_id(op)
  println("Loading projected affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  load(joinpath(path_id,"basis_space"))
end

function load_rb_structure(
  info::RBInfo,
  op::RBBilinOperator,
  meas::Measure)

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id and its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  load_mdeim(path_id,op,meas),load_mdeim(path_id_lift,op,meas)
end

function load_rb_structure(
  info::RBInfo,
  op::RBBilinOperator{Top,<:TrialFESpace},
  meas::Measure) where Top

  id = get_id(op)
  println("Loading MDEIM structures for non-affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  load_mdeim(path_id,op,meas)
end

function load_rb_structure(
  info::RBInfo,
  op::RBBilinOperator{Affine,Ttr},
  meas::Measure) where Ttr

  id = get_id(op)
  println("Loading projected affine variable $id and MDEIM structures for its lifting")
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  load(joinpath(path_id,"basis_space")),load_mdeim(path_id_lift,op,meas)
end

function load_rb_structure(
  ::RBInfo,
  op::RBBilinOperator{Affine,<:TrialFESpace},
  ::Measure)

  id = get_id(op)
  println("Loading projected affine variable $id")
  path_id = joinpath(info.offline_path,"$id")

  load(joinpath(path_id,"basis_space"))
end
