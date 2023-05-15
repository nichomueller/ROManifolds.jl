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
    assemble_affine_decomposition(info,args...)
  end
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBVariable{Affine,Ttr},
  args...) where Ttr

  id = get_id(op)
  printstyled("Operator $id is Affine: computing its RB Galerkin projection \n";color=:blue)

  adrb = rb_space_projection(op)
  RBAffineDecomposition(op,adrb)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  args...) where {Top,Ttr}

  id,nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Operator $id is $Top: running the MDEIM offline phase on $nsnap snapshots \n";
    color=:blue)

  adrb = MDEIM(info,op,args...)
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
  μ::Vector{Param},
  meas::Measure)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")
  if ispath(path_id)
    load(info,op,meas)
  else
    printstyled("Failed to load variable $(id): running offline assembler instead\n";
      color=:blue)
    reduce_affine_decomposition(info,op,μ,meas)
  end

end

function load(
  info::RBInfo,
  op::RBVariable{Affine,Ttr},
  ::Measure) where Ttr

  id = get_id(op)
  printstyled("Loading projected Affine variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  adrb = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,adrb)
end

function load(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  meas::Measure) where {Top,Ttr}

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  adrb = load(info,path_id,op,meas)
  RBAffineDecomposition(op,adrb)
end
