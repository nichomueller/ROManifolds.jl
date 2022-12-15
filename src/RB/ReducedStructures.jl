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

  rb_projection(op)
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
  op::RBBilinOperator{Top,UnconstrainedFESpace},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is non-affine and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBBilinOperator{Affine,TT},
  args...) where TT

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  println("Bilinear operator $id is affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")

  op_lift = RBLiftingOperator(op)
  rb_projection(op),mdeim_offline(info,op_lift,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinOperator{Affine,UnconstrainedFESpace},
  args...)

  id = get_id(op)
  println("Bilinear operator $id is affine and has no lifting: computing Φᵀ$(id)Φ")

  rb_projection(op)
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
  idx_lu = get_idx_lu_factors(mdeim)
  save(joinpath(path,"LU"),idx_lu.factors)
  save(joinpath(path,"p"),idx_lu.ipiv)
end

function save_rb_structure(path::String,mdeim::MDEIMUnsteady)
  save(joinpath(path,"basis_space"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  save(joinpath(path,"basis_time"),get_basis_time(mdeim))
  save(joinpath(path,"idx_time"),get_idx_time(mdeim))
  idx_lu = get_idx_lu_factors(mdeim)
  save(joinpath(path,"LU"),idx_lu.factors)
  save(joinpath(path,"p"),idx_lu.ipiv)
end

function load_rb_structure(
  info::RBInfo,
  op::RBVarOperator,
  meas::Measure)

  id = get_id(op)
  println("Importing reduced $id")
  path_id = joinpath(info.offline_path,"$id")
  load_mdeim(path_id,op,meas)
end

function load_rb_structure(
  info::RBInfo,
  op::Union{RBLinOperator{Affine},RBBilinOperator{Affine,TT}},
  args...) where TT

  id = get_id(op)
  println("Importing reduced $id")
  path_id = joinpath(info.offline_path,"$id")
  load(joinpath(path_id,"basis_space")),load(joinpath(path_id*"_lift","basis_space"))
end

function load_rb_structure(
  info::RBInfo,
  op::Union{RBLinOperator{Affine},RBBilinOperator{Affine,<:UnconstrainedFESpace}},
  args...)

  id = get_id(op)
  println("Importing reduced $id")
  path_id = joinpath(info.offline_path,"$id")
  load(joinpath(path_id,"basis_space"))
end
