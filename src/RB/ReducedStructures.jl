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
  rb_projection(op)
end

function assemble_rb_structures(
  ::RBInfo,
  op::RBBilinOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}
  rb_projection(op)
end

function assemble_rb_structures(
  info::RBInfo,
  op::RBVarOperator,
  args...)

  id = get_id(op)
  println("Matrix $id is non-affine: running the MDEIM offline phase on $(info.mdeim_nsnap) snapshots")

  mdeim = MDEIM(info,op,args...)
  project_mdeim_space(mdeim,op)
end

function project_mdeim_space!(mdeim::MDEIM,op::RBLinOperator)
  basis_space = get_basis_space(mdeim)
  findnz_map = get_findnz_mapping(op)
  full_basis_space = fill_rows_with_zeros(basis_space,findnz_map)
  rbspace_row = get_rbspace_row(op)

  mdeim.rbspace.basis_space = rbspace_row'*full_basis_space
  mdeim
end

function project_mdeim_space!(mdeim::MDEIM,op::RBBilinOperator)
  basis_space = get_basis_space(mdeim)
  findnz_map = get_findnz_mapping(op)
  full_basis_space = fill_rows_with_zeros(basis_space,findnz_map)

  rbspace_row = get_rbspace_row(op)
  rbspace_col = get_rbspace_col(op)

  mdeim.rbspace.basis_space = rbspace_row'*full_basis_space*rbspace_col
  mdeim
end
