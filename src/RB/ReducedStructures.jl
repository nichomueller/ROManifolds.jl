function assemble_rb_structures(info::RBInfo,tt::TimeTracker,op::RBVarOperator,args...)
  tt.offline_time += @elapsed begin
    rb_variable = assemble_rb_structures(info,op,args...)
  end
  save_rb_structure(info,get_id(op),rb_variable)
  blocks(id_rb)
end

function assemble_rb_structures(
  ::RBInfo,
  op::RBVarOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}

  rb_projection(op)
end

function assemble_rb_structures(
  info::RBInfo,
  op::RBVarOperator,
  μ::Snapshots,
  args...)

  id = get_id(op)
  println("Matrix $id is non-affine: running the MDEIM offline phase on $(info.mdeim_nsnap) snapshots")

  mdeim_offline(info,op,μ,args...)
end

#= function reduced_measure!(
  ::RBVarOperator,
  ::AbstractArray,
  meas::ProblemMeasures,
  args...)

  meas
end =#

function save_rb_structure(
  info::RBInfo,
  id::Symbol,
  val)

  if info.save_offline
    off_path = info.offline_path
    save(joinpath(off_path,"$(id)_rb"),val)
  end
end

load(::ParamVarOperator{Affine,TT,Tsp},path::String) where {TT,Tsp} = load(path)
load(::ParamVarOperator{Top,TT,Tsp},path::String) where {Top,TT,Tsp} = load_mdeim(path)

function load_rb_structure(
  info::RBInfo,
  op::ParamLinOperator{Affine,Tsp},
  args...) where Tsp

  off_path = info.offline_path
  path = correct_path(joinpath(off_path,"$(id)_rb"))

  if isfile(path)
    id = get_id(op)
    println("Importing reduced $id")
    id_rb = load(op,path)
  else
    basis,other_args = args
    rbop = RBVarOperator(op,basis)
    id_rb = assemble_rb_structures(info,rbop,other_args...)
  end
  blocks(id_rb)
end

function load_rb_structure(
  info::RBInfo,
  op::ParamBilinOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}

  off_path = info.offline_path
  path = correct_path(joinpath(off_path,"$(id)_rb"))

  if isfile(path)
    id = get_id(op)
    println("Importing reduced $id")
    id_rb = load(op,path)
  else
    basis_row,basis_col,other_args = args
    rbop = RBVarOperator(op,basis_row,basis_col)
    id_rb = assemble_rb_structures(info,rbop,other_args...)
  end
  blocks(id_rb)
end
