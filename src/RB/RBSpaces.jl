abstract type EnrichmentStyle end
struct NoEnrichment <: EnrichmentStyle end
struct SupremizingEnrichment <: EnrichmentStyle end

struct RBSpace
  basis_space::AbstractArray
  basis_time::AbstractArray

  function RBSpace(::NoEnrichment,snaps::Snapshots,args...;kwargs...)
    basis_space,basis_time = tpod(snaps;kwargs...)
    new(basis_space,basis_time)
  end

  function RBSpace(::SupremizingEnrichment,snaps::Snapshots,args...;ttol=1e-2,kwargs...)
    basis_space,basis_time = tpod(snaps;kwargs...)
    if compute_supremizers
      add_space_supremizers!(bases_space,snaps,args...)
      add_time_supremizers!(bases_time,ttol)
    end
    new(bases_space,bases_time)
  end
end

get_basis_space(rb::RBSpace) = rb.basis_space

get_basis_time(rb::RBSpace) = rb.basis_time

function compress_snapshots end

compress_snapshots(args...;enrich=NoEnrichment(),kwargs...) = RBSpace(enrich,args...;kwargs...)

function add_space_supremizers!(
  bases_space::Vector{NnzArray{T}},
  feop::ParamFEOperator,
  fesolver::ODESolver,
  snaps::MultiFieldSnapshots,
  params::Table;
  kwargs...) where T

  bsprimal,bsdual... = map(recast,bases_space)
  for (i,bsd) in enumerate(bsdual)
    printstyled("Computing supremizers in space for dual field $i\n";color=:blue)
    supr_i = space_supremizers(feop,fesolver,snaps,bsd,params,i+1)
    bsu_i = gram_schmidt(supr_i,bsprimal)
    bases_space[1].nonzero_val = hcat(bases_space[1].nonzero_val,bsu_i)
  end
  return
end

function space_supremizers(
  feop::ParamFEOperator,
  fesolver::ODESolver,
  s::MultiFieldSnapshots,
  bs::AbstractMatrix,
  params::Table,
  i::Int)

  filter = (1,i)
  snaps = get_datum(s)

  matdata = _matdata_jacobian(feop,fesolver,snaps,params,filter)
  aff = affinity_jacobian(fesolver,params,matdata)
  data = get_datum(aff,fesolver,params,matdata)
  constraint_mat = map(d->assemble_matrix(feop.assem,d),data)

  if isa(aff,ParamAffinity) || isa(aff,ParamTimeAffinity)
    supr = first(constraint_mat)*bs
  else
    supr = map(*,constraint_mat,snaps[i])
  end

  supr
end

function add_time_supremizers!(bases_time::Vector{NnzArray{T}},ttol::Real) where T
  tbu,tbdual... = map(get_nonzero_val,bases_time)
  for (i,tb) in enumerate(tbdual)
    printstyled("Computing supremizers in time for dual field $i\n";color=:blue)
    tbu_i = add_time_supremizers([tbu,tb],ttol)
    bases_time[1].nonzero_val = hcat(bases_time[1].nonzero_val,tbu_i)
  end
  return
end

function add_time_supremizers(bases_time::Vector{<:AbstractMatrix},ttol::Real)
  basis_u,basis_p = bases_time
  basis_up = basis_u'*basis_p

  function enrich(
    basis_u::AbstractMatrix{Float},
    basis_up::AbstractMatrix{Float},
    v::AbstractArray{Float})

    vnew = orth_complement(v,basis_u)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),vcat(basis_up,vnew'*basis_p)
  end

  count = 0
  ntp_minus_ntu = size(basis_p,2) - size(basis_u,2)
  if ntp_minus_ntu > 0
    for ntp = 1:ntp_minus_ntu
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
    end
  end

  ntp = 1
  while ntp ≤ size(basis_up,2)
    proj = ntp == 1 ? zeros(size(basis_up[:,1])) : orth_projection(basis_up[:,ntp],basis_up[:,1:ntp-1])
    dist = norm(basis_up[:,1]-proj)
    if dist ≤ ttol
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
      ntp = 0
    else
      basis_up[:,ntp] -= proj
    end
    ntp += 1
  end

  printstyled("Added $count time supremizers\n";color=:blue)
  basis_u
end
