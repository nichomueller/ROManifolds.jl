function assemble_rb_structure(info::RBInfo,tt::TimeTracker,op::RBVarOperator,args...)
  tt.offline_time += @elapsed begin
    rb_variable = assemble_rb_structure(info,op,args...)
  end
  save(info,rb_variable,get_id(op))
  rb_variable
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBLinOperator,
  args...)

  println("Linear operator $id is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")
  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBLinOperator{Affine},
  args...)

  println("Linear operator $id is affine: computing Φᵀ$id")
  rb_projection(op)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinOperator,
  args...)

  println("Bilinear operator $id and its lifting are non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")
  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinOperator{Top,UnconstrainedFESpace},
  args...) where Top

  println("Bilinear operator $id is non-affine and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")
  mdeim_offline(info,op,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinOperator{Affine,TT},
  args...) where TT

  println("Bilinear operator $id is affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots")
  op_lift = RBLiftingOperator(op)
  rb_projection(op),mdeim_offline(info,op_lift,args...)
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBBilinOperator{Affine,UnconstrainedFESpace},
  args...)

  println("Bilinear operator $id is affine and has no lifting: computing Φᵀ$(id)Φ")
  rb_projection(op)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBVarOperator,
  args...)

  print_case(info,op)
  mdeim_offline(info,op,args...)
end

save(info::RBInfo,b,id::Symbol) =
  if info.save_offline save(info.offline_path,b,id) end

function save(path::String,basis::Matrix{Float},id::Symbol)
  save(joinpath(path,"basis_space_"*"$id"),basis)
end

function save(path::String,b::NTuple{2,Matrix{Float}},id::Symbol)
  basis,basis_lift = b
  save(path,basis,id)
  save(path,basis_lift,id*:_lift)
end

function save(path::String,mdeim::MDEIMSteady,id::Symbol)
  save(joinpath(path,"basis_space_"*"$id"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space_"*"$id"),get_idx_space(mdeim))
  save_idx_lu_factors(path,mdeim,id)
end

function save(path::String,mdeim::MDEIMUnsteady,id::Symbol)
  save(joinpath(path,"basis_space_"*"$id"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space_"*"$id"),get_idx_space(mdeim))
  save(joinpath(path,"basis_time_"*"$id"),get_basis_time(mdeim))
  save(joinpath(path,"idx_time_"*"$id"),get_idx_time(mdeim))
  save_idx_lu_factors(path,mdeim,id)
end

function save(path::String,m::NTuple{2,<:MDEIM},id::Symbol)
  mdeim,mdeim_lift = m
  save(path,mdeim,id)
  save(path,mdeim_lift,id*:_lift)
end

function load_rb_structure(
  info::RBInfo,
  op::RBVarOperator,
  meas::Measure)

  id = get_id(op)
  println("Importing reduced $id")
  load_mdeim(info,op,meas)
end

function load_rb_structure(
  info::RBInfo,
  op::Union{RBLinOperator{Affine},RBBilinOperator{Affine,TT}}
  args...) where TT

  id = get_id(op)
  println("Importing reduced $id")
  load_rb(info,id),load_rb(info,id*:_lift)
end

function load_rb_structure(
  info::RBInfo,
  op::Union{RBLinOperator{Affine},RBBilinOperator{Affine,<:UnconstrainedFESpace}}
  args...)

  id = get_id(op)
  println("Importing reduced $id")
  load_rb(info,id)
end
