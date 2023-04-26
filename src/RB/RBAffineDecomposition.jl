struct RBAffineDecomposition{Top,Ttr,Tad}
  op::RBVariable{Top,Ttr}
  affine_decomposition::Tad
end

function RBAffineDecomposition(
  op::RBVariable{Top,Ttr},
  affine_decomposition::Tad) where {Top,Ttr,Tad}

  RBAffineDecomposition{Top,Ttr,Tad}(op,affine_decomposition)
end

function RBAffineDecomposition(
  info::RBInfo,
  args...)

  if info.load_offline
    load(info,args...)
  else
    assemble_affine_decomposition(info,args...)
  end
end

get_op(ad::RBAffineDecomposition) = ad.op

get_affine_decomposition(ad::RBAffineDecomposition) = ad.affine_decomposition

get_id(ad::RBAffineDecomposition) = get_id(get_op(ad))

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBVariable{Affine,Ttr},
  args...) where Ttr

  id = get_id(op)
  printstyled("Operator $id is Affine: computing its RB Galerkin projection \n";color=:blue)

  ad = rb_space_projection(op)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  args...) where {Top,Ttr}

  id,nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Operator $id is $Top: running the MDEIM offline phase on $nsnap snapshots \n";
    color=:blue)

  ad = MDEIM(info,op,args...)
  RBAffineDecomposition(op,ad)
end

function eval_affine_decomposition(ad::RBAffineDecomposition)
  op = get_op(ad)
  eval_affine_decomposition(op,ad)
end

function eval_affine_decomposition(
  ::RBSteadyVariable,
  ad::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  get_affine_decomposition(ad)
end

function eval_affine_decomposition(
  ::RBSteadyVariable,
  ad::RBAffineDecomposition)

  mdeim = get_affine_decomposition(ad)
  get_basis_space(mdeim)
end

function eval_affine_decomposition(
  op::RBUnsteadyVariable,
  ad::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  ns_row = get_ns(get_rbspace_row(op))
  array3D(get_affine_decomposition(ad),ns_row)
end

function eval_affine_decomposition(
  op::RBUnsteadyVariable,
  ad::RBAffineDecomposition)

  ns_row = get_ns(get_rbspace_row(op))
  mdeim = get_affine_decomposition(ad)
  array3D(get_basis_space(mdeim),ns_row)
end

function save(info::RBInfo,ad::RBAffineDecomposition)
  id = get_id(ad)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  save(path_id,ad)
end

function save(
  path::String,
  ad::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  ad = get_affine_decomposition(ad)
  save(joinpath(path,"basis_space"),ad)
end

function save(path::String,ad::RBAffineDecomposition)
  ad = get_affine_decomposition(ad)
  save(path,ad)
end

function load(
  info::RBInfo,
  op::RBVariable,
  args...)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")
  if ispath(path_id)
    load(info,op,args...)
  else
    printstyled("Failed to load variable $(id): running offline assembler instead\n";
      color=:blue)
    assemble_affine_decomposition(info,op,args...)
  end

end

function load(
  info::RBInfo,
  op::RBVariable{Affine,Ttr},
  ::ProblemMeasures,
  ::Symbol,
  args...) where Ttr

  id = get_id(op)
  printstyled("Loading projected Affine variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  ad = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,ad)
end

function load(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  measures::ProblemMeasures,
  field::Symbol,
  args...) where {Top,Ttr}

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id \n";color=:blue)

  ad = load(info,op,getproperty(measures,field))
  RBAffineDecomposition(op,ad)
end
