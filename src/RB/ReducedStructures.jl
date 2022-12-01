function assemble_rb_structures(info::RBInfo,res::RBResults,op::ParamVarOperator,args...)
  res.offline_time += @elapsed begin
    assemble_rb_structures(info,op,args...)
  end
end

function assemble_rb_structures(
  info::RBInfo,
  op::RBVarOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}

  rb_structure = rb_projection(op)
  save_rb_structure(info,get_id(op),rb_structure)
  rb_structure
end

function assemble_rb_structures(
  info::RBInfo,
  op::RBVarOperator,
  args...)

  id = get_id(op)
  println("Matrix $id is non-affine: running the MDEIM offline phase on $(info.mdeim_nsnap) snapshots")

  mdeim = MDEIM(info,op,args...)
  save_rb_structure(info,get_id(op),mdeim)
  mdeim
end

function reduced_measure!(
  ::RBVarOperator,
  ::AbstractArray,
  meas::ProblemMeasures,
  args...)

  meas
end

function reduced_measure!(
  op::RBVarOperator,
  mdeim::MDEIM,
  meas::ProblemMeasures,
  field=:dΩ)

  m = getproperty(meas,field)
  red_meas = get_reduced_measure(op,mdeim,m)
  setproperty!(meas,field,red_meas)
end

function save_rb_structure(
  info::RBInfo,
  id::Symbol,
  val)

  if info.save_offline
    off_path = info.offline_path
    save(joinpath(off_path,"$(id)_rb"),val)
  end
end

function load_rb_structure(
  info::RBInfo,
  op::ParamVarOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}

  off_path = info.offline_path
  path = correct_path(joinpath(off_path,"$(id)_rb"))

  if isfile(path)
    id = get_id(op)
    println("Importing reduced $id")
    load(path)
  else
    assemble_rb_structures(info,op,args...)
  end
end
