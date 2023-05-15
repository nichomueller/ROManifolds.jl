struct RBAffineDecomposition{Top,Ttr,Tad}
  op::RBVariable{Top,Ttr}
  rb_aff_dec::Tad

  function RBAffineDecomposition(
    op::RBVariable{Top,Ttr},
    rb_aff_dec::Tad) where {Top,Ttr,Tad}

    new{Top,Ttr,Tad}(op,rb_aff_dec)
  end
end

get_op(adrb::RBAffineDecomposition) = adrb.op

get_rb_aff_dec(adrb::RBAffineDecomposition) = adrb.rb_aff_dec

get_id(adrb::RBAffineDecomposition) = get_id(get_op(adrb))

function RBAffineDecomposition(
  info::RBInfo,
  args...)

  if info.load_offline
    load(info,args...)
  else
    reduce_affine_decomposition(info,args...)
  end
end

function reduce_affine_decomposition(
  ::RBInfo,
  op::RBVariable{Affine,Ttr},
  ad::AffineDecomposition,
  args...) where Ttr

  mat = get_snap(get_aff_dec(ad))
  adrb = rb_space_projection(op,mat)
  RBAffineDecomposition(op,adrb)
end

function reduce_affine_decomposition(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  ad::AffineDecomposition,
  args...) where {Top,Ttr}

  adrb = MDEIM(info,op,ad,args...)
  RBAffineDecomposition(op,adrb)
end

function eval_affine_decomposition(
  adrb::RBAffineDecomposition{Affine,Ttr,Tad}) where {Ttr,Tad}

  get_rb_aff_dec(adrb)
end

function eval_affine_decomposition(
  adrb::RBAffineDecomposition)

  mdeim = get_rb_aff_dec(adrb)
  get_basis_space(mdeim)
end

function save(info::RBInfo,adrb::RBAffineDecomposition)
  id = get_id(adrb)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  save(path_id,adrb)
end

function save(
  path::String,
  adrb::RBAffineDecomposition{Affine,Ttr,Tad}) where {Ttr,Tad}

  adrb = get_rb_aff_dec(adrb)
  save(joinpath(path,"basis_space"),adrb)
end

function save(path::String,adrb::RBAffineDecomposition)
  adrb = get_rb_aff_dec(adrb)
  save(path,adrb)
end

function load(
  info::RBInfo,
  op::RBVariable,
  ad::AffineDecomposition,
  args...)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")
  if ispath(path_id)
    load(info,op,args...)
  else
    printstyled("Failed to load variable $(id): running offline assembler instead\n";
      color=:blue)
    reduce_affine_decomposition(info,op,ad,args...)
  end

end

function load(
  info::RBInfo,
  op::RBVariable{Affine,Ttr},
  ::ProblemMeasures,
  ::Symbol) where Ttr

  id = get_id(op)
  printstyled("Loading projected Affine variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  adrb = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,adrb)
end

function load(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  measures::ProblemMeasures,
  field::Symbol) where {Top,Ttr}

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id \n";color=:blue)

  adrb = load(info,op,getproperty(measures,field))
  RBAffineDecomposition(op,adrb)
end
