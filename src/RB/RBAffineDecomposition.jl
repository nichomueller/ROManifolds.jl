struct RBAffineDecomposition{Top,Ttr,Tad}
  op::RBVariable{Top,Ttr}
  affine_decomposition::Tad
end

function RBAffineDecomposition(
  op::RBVariable{Top,Ttr},
  affine_decomposition::Tad) where {Top,Ttr,Tad}

  RBAffineDecomposition{Top,Ttr,Tad}(op,affine_decomposition)
end

function RBAffineDecomposition(info::RBInfo,args...;kwargs...)
  if info.load_offline
    load(info,args...;kwargs...)
  else
    assemble_affine_decomposition(info,args...;kwargs...)
  end
end

function assemble_affine_decomposition(
  info::RBInfo,
  tt::TimeTracker,
  op::RBVariable,
  args...;
  lift=true)

  tt.offline_time.assembly_time += @elapsed begin
    ad = assemble_affine_decomposition(info,op,Val(lift),args...)
  end
  save(info,ad,get_id(op))
  ad
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBLinVariable{Affine},
  args...)

  id = get_id(op)
  printstyled("Linear operator $id is Affine: computing Φᵀ$id \n";color=:blue)

  ad = rb_space_projection(op)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBLinVariable{Top},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Linear operator $id is $Top:
    running the MDEIM offline phase on $mdeim_nsnap snapshots \n";color=:blue)

  ad = MDEIM(info,op,args...)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBBilinVariable{Affine,<:SingleFieldFESpace},
  args...)

  id = get_id(op)
  printstyled("Bilinear operator $id is Affine and has no lifting:
    computing Φᵀ$(id)Φ\n";color=:blue)

  ad = rb_space_projection(op)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Top,<:SingleFieldFESpace},
  args...) where Top

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Bilinear operator $id is $Top and has no lifting:
    running the MDEIM offline phase on $mdeim_nsnap snapshots\n";color=:blue)

  ad = MDEIM(info,op,args...)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  ::Val{true},
  args...) where Ttr

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Bilinear operator $id is Affine but its lifting is non-affine:
    running the MDEIM offline phase on $mdeim_nsnap snapshots\n";color=:blue)

  ad = rb_space_projection(op)
  op_lift = RBLiftVariable(op)
  ad_lift = MDEIM(info,op_lift,args...)
  RBAffineDecomposition(op,ad),RBAffineDecomposition(op_lift,ad_lift)
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  ::Val{false},
  args...) where Ttr

  id = get_id(op)
  printstyled("Bilinear operator $id is Affine, not building its lifting\n";
    color=:blue)

  ad = rb_space_projection(op)
  RBAffineDecomposition(op,ad)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Top,Ttr},
  ::Val{true},
  args...) where {Top,Ttr}

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Bilinear operator $id and its lifting are $Top:
    running the MDEIM offline phase on $mdeim_nsnap snapshots\n";color=:blue)

  ad = MDEIM(info,op,args...)
  op_lift = RBLiftVariable(op)
  ad_lift = MDEIM(info,op_lift,args...)
  RBAffineDecomposition(op,ad),RBAffineDecomposition(op_lift,ad_lift)
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBBilinVariable{Top,Ttr},
  ::Val{false},
  args...) where {Top,Ttr}

  id,mdeim_nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Bilinear operator $id is $Top: running the MDEIM offline phase
    on $mdeim_nsnap snapshots; not building its lifting\n";color=:blue)

  ad = MDEIM(info,op,args...)
  RBAffineDecomposition(op,ad)
end

get_op(rbs::RBAffineDecomposition) = rbs.op

get_affine_decomposition(rbs::RBAffineDecomposition) = rbs.affine_decomposition

function save(info::RBInfo,ad,id::Symbol)
  path_id = joinpath(info.offline_path,"$id")
  create_dir!(path_id)
  if info.save_offline
    save(path_id,ad)
  end
end

function save(path::String,rbv::NTuple{2,RBAffineDecomposition})
  rbvar,rbvar_lift = rbv
  save(path,rbvar)
  path_lift = path*"_lift"
  create_dir!(path_lift)
  save(path_lift,rbvar_lift)
end

function save(
  path::String,
  ad::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  os = get_affine_decomposition(ad)
  save(joinpath(path,"basis_space"),os)
end

function save(path::String,ad::RBAffineDecomposition)
  os = get_affine_decomposition(ad)
  save(path,os)
end

function load(
  info::RBInfo,
  tt::TimeTracker,
  op::RBVariable,
  args...;
  lift=true)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")
  if ispath(path_id)
    _,meas,field = args
    load(info,op,getproperty(meas,field),Val(lift))
  else
    printstyled("Failed to load variable $(id):
      running offline assembler instead\n";color=:blue)
    assemble_affine_decomposition(info,tt,op,args...;lift=lift)
  end

end

function load(
  info::RBInfo,
  op::RBLinVariable{Top},
  meas::Measure,
  args...) where Top

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBAffineDecomposition(op,os)
end

function load(
  info::RBInfo,
  op::RBLinVariable{Affine},
  ::Measure,
  args...)

  id = get_id(op)
  printstyled("Loading projected Affine variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Top,<:SingleFieldFESpace},
  meas::Measure,
  args...) where Top

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBAffineDecomposition(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,<:SingleFieldFESpace},
  ::Measure,
  args...)

  id = get_id(op)
  printstyled("Loading projected Affine variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,os)
end


function load(
  info::RBInfo,
  op::RBBilinVariable{Top,Ttr},
  meas::Measure,
  ::Val{true}) where {Top,Ttr}

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id and its lifting \n";
    color=:blue)
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  os = load(path_id,op,meas)
  op_lift = RBLiftVariable(op)
  os_lift = load(path_id_lift,op,meas)
  RBAffineDecomposition(op,os),RBAffineDecomposition(op_lift,os_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Top,Ttr},
  meas::Measure,
  ::Val{false}) where {Top,Ttr}

  id = get_id(op)
  printstyled("Loading MDEIM structures for $Top variable $id\n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  os = load(path_id,op,meas)
  RBAffineDecomposition(op,os)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  meas::Measure,
  ::Val{true}) where Ttr

  id = get_id(op)
  printstyled("Loading projected Affine variable $id and MDEIM structures
    for its lifting\n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")
  path_id_lift = joinpath(info.offline_path,"$(id)_lift")

  os = load(joinpath(path_id,"basis_space"))
  op_lift = RBLiftVariable(op)
  os_lift = load(path_id_lift,op,meas)
  RBAffineDecomposition(op,os),RBAffineDecomposition(op_lift,os_lift)
end

function load(
  info::RBInfo,
  op::RBBilinVariable{Affine,Ttr},
  ::Measure,
  ::Val{false}) where Ttr

  id = get_id(op)
  printstyled("Loading projected Affine variable $id \n";color=:blue)
  path_id = joinpath(info.offline_path,"$id")

  os = load(joinpath(path_id,"basis_space"))
  RBAffineDecomposition(op,os)
end

function eval_affine_decomposition(rbs::Tuple)
  eval_affine_decomposition.(expand(rbs))
end

function eval_affine_decomposition(rbs::RBAffineDecomposition)
  op = get_op(rbs)
  id = get_id(op)
  printstyled("Evaluating the RB offline quantity for $id \n";color=:blue)

  eval_affine_decomposition(Val(issteady(op)),rbs)
end

function eval_affine_decomposition(
  ::Val{true},
  rbs::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  get_affine_decomposition(rbs)
end

function eval_affine_decomposition(::Val{true},rbs::RBAffineDecomposition)
  mdeim = get_affine_decomposition(rbs)
  get_basis_space(mdeim)
end

function eval_affine_decomposition(
  ::Val{false},
  rbs::RBAffineDecomposition{Affine,Ttr,Matrix{Float}}) where Ttr

  op = get_op(rbs)
  ns_row = get_ns(get_rbspace_row(op))

  os = get_affine_decomposition(rbs)
  blocks(os,ns_row)
end

function eval_affine_decomposition(::Val{false},rbs::RBAffineDecomposition)
  op = get_op(rbs)
  ns_row = get_ns(get_rbspace_row(op))

  mdeim = get_affine_decomposition(rbs)
  os = get_basis_space(mdeim)
  blocks(os,ns_row)
end
