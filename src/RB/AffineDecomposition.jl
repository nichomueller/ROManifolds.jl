struct AffineDecomposition
  aff_dec::Snapshots
  findnz_idx::Vector{Int}
end

get_aff_dec(ad::AffineDecomposition) = ad.aff_dec

get_findnz_idx(ad::AffineDecomposition) = ad.findnz_idx

function AffineDecomposition(info::RBInfo,op::RBVariable,args...)
  if info.load_offline
    empty_affine_decomposition(op)
  else
    assemble_affine_decomposition(info,op,args...)
  end
end

function empty_affine_decomposition(op::RBVariable)
  empty_snap = Snapshots(get_id(op),allocate_matrix(Matrix{Float},0,0),0)
  empty_vec = zeros(0)
  AffineDecomposition(empty_snap,empty_vec)
end

function assemble_affine_decomposition(
  ::RBInfo,
  op::RBVariable{Affine,Ttr},
  args...) where Ttr

  id = get_id(op)
  printstyled("Operator $id is Affine: computing its RB Galerkin projection \n";
    color=:blue)

  ad = assemble_affine_quantity(op)
  AffineDecomposition(Snapshots(id,ad,1),collect(axes(ad,1)))
end

function assemble_affine_decomposition(
  info::RBInfo,
  op::RBVariable{Top,Ttr},
  args...) where {Top,Ttr}

  id,nsnap = get_id(op),info.mdeim_nsnap
  printstyled("Operator $id is $Top: running the MDEIM offline phase on $nsnap snapshots \n";
    color=:blue)

  ad,findnz_idx = generate_mdeim_snapshots_on_workers(op,args...;
    fun_mdeim=info.fun_mdeim)
  AffineDecomposition(ad,findnz_idx)
end
