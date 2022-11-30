function get_rb_structure(
  info::RBInfo,
  res::RBResults,
  op::ParamVarOperator{Affine,TT}) where TT

  id = get_id(op)
  println("Importing reduced $id")

  res.offline_time += @elapsed begin

  end
end

function assemble_rb_structures(info::RBInfo,res::RBResults,op::ParamVarOperator,args...)
  res.offline_time += @elapsed begin
    assemble_rb_structures(info,op,args...)
  end
end

function assemble_rb_structures(
  ::RBInfo,
  op::RBLinOperator{Affine,Tsp},
  args...) where Tsp
  compute_rb_projection(op)
end

function assemble_rb_structures(
  ::RBInfo,
  op::RBBilinOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}
  compute_rb_projection(op)
end

function assemble_rb_structures(
  info::RBInfo,
  op::RBVarOperator,
  args...)

  id = get_id(op)
  println("Matrix $id is non-affine: running the MDEIM offline phase on $(info.mdeim_nsnap) snapshots")

  MDEIM_offline(info,op,args...)
  assemble_MDEIM_Matₙ(Var, get_Φₛ(RBVars, var)...)
end
