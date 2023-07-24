abstract type RBSpace{T} end
abstract type TransientRBSpace{T} end

struct SingleFieldRBSpace{T} <: RBSpace{T}
  basis_space::NnzArray{T}
end

struct MultiFieldRBSpace{T} <: RBSpace{T}
  basis_space::Vector{NnzArray{T}}
end

struct TransientSingleFieldRBSpace{T} <: TransientRBSpace{T}
  basis_space::NnzArray{T}
  basis_time::NnzArray{T}
end

struct TransientMultiFieldRBSpace{T} <: TransientRBSpace{T}
  basis_space::Vector{NnzArray{T}}
  basis_time::Vector{NnzArray{T}}
end

get_basis_space(rb::SingleFieldRBSpace) = recast(rb.basis_space)

get_basis_space(rb::MultiFieldRBSpace) = vcat(map(recast,rb.basis_space)...)

get_basis_space(rb::TransientSingleFieldRBSpace) = recast(rb.basis_space)

get_basis_space(rb::TransientMultiFieldRBSpace) = vcat(map(recast,rb.basis_space)...)

get_rb_space_ndofs(rb) = size(get_basis_space(rb),2)

get_basis_time(rb::TransientSingleFieldRBSpace) = rb.basis_time.nonzero_val

get_basis_time(rb::TransientMultiFieldRBSpace) = vcat(map(recast,rb.basis_time)...)

get_rb_time_ndofs(rb::TransientRBSpace) = size(get_basis_time(rb),2)

get_rb_ndofs(rb::RBSpace) = get_rb_space_ndofs(rb)

get_rb_ndofs(rb::TransientRBSpace) = get_rb_space_ndofs(rb)*get_rb_time_ndofs(rb)

for (Tsps,Tspm) in zip((:SingleFieldRBSpace,:TransientSingleFieldRBSpace),
                       (:MultiFieldRBSpace,:TransientMultiFieldRBSpace))
  @eval begin
    Base.getindex(rb::$Tsps,args...) = rb

    Base.getindex(rb::$Tspm,i::Int) = get_single_field(rb,i)

    Base.length(rb::$Tsps) = 1

    Base.length(rb::$Tspm) = length(rb.basis_space)
  end
end

function allocate_matrix(rb::SingleFieldRBSpace,sizes...)
  allocate_matrix(rb.basis_space,sizes...)
end

function get_single_field(
  rb::MultiFieldRBSpace{T},
  fieldid::Int) where T

  SingleFieldRBSpace{T}(rb.basis_space[fieldid])
end

function get_single_field(
  rb::TransientMultiFieldRBSpace{T},
  fieldid::Int) where T

  TransientSingleFieldRBSpace{T}(rb.basis_space[fieldid],rb.basis_time[fieldid])
end

function compress_solutions(
  ::ParamFEOperator,
  ::FESolver,
  s::SingleFieldSnapshots{T,A},
  args...;
  kwargs...) where {T,A}

  basis_space = tpod(s;kwargs...)
  SingleFieldRBSpace{T}(basis_space)
end

function compress_solutions(
  feop::ParamFEOperator,
  fesolver::FESolver,
  s::MultiFieldSnapshots{T,N,A},
  args...;
  compute_supremizers=false,
  kwargs...) where {T,N,A}

  bases_space = map(snap -> tpod(snap;kwargs...),s)
  if compute_supremizers
    add_space_supremizers!(bases_space,feop,fesolver,s,args...)
  end
  MultiFieldRBSpace{T}(bases_space)
end

function compress_solutions(
  ::ParamTransientFEOperator,
  fesolver::ODESolver,
  s::TransientSingleFieldSnapshots{T,A},
  args...;
  kwargs...) where {T,A}

  basis_space,basis_time = tpod(s,fesolver;kwargs...)
  TransientSingleFieldRBSpace{T}(basis_space,basis_time)
end

function compress_solutions(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  s::TransientMultiFieldSnapshots{T,N,A},
  args...;
  compute_supremizers=false,
  ttol=1e-2,
  kwargs...) where {T,N,A}

  bases = map(snap -> tpod(snap,fesolver;kwargs...),s)
  bases_space,bases_time = first.(bases),last.(bases)
  if compute_supremizers
    add_space_supremizers!(bases_space,feop,fesolver,s,args...)
    add_time_supremizers!(bases_time;ttol)
  end
  TransientMultiFieldRBSpace{T}(bases_space,bases_time)
end

for (Top,Tslv,Tsnp) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:MultiFieldSnapshots,:TransientMultiFieldSnapshots))

  @eval begin
    function add_space_supremizers!(
      bases_space::Vector{NnzArray{T}},
      feop::$Top,
      fesolver::$Tslv,
      snaps::$Tsnp,
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
      feop::$Top,
      fesolver::$Tslv,
      s::$Tsnp,
      bs::AbstractMatrix,
      params::Table,
      i::Int)

      filter = (1,i)
      snaps = get_datum(s)

      matdata = _matdata_jacobian(feop,fesolver,snaps,params,filter)
      aff = get_affinity(fesolver,params,matdata)
      data = get_datum(aff,fesolver,params,matdata)
      constraint_mat = map(d->assemble_matrix(feop.assem,d),data)

      if isa(aff,ParamAffinity) || isa(aff,ParamTimeAffinity)
        supr = first(constraint_mat)*bs
      else
        supr = map(*,constraint_mat,snaps[i])
      end

      supr
    end
  end
end

function add_time_supremizers!(
  bases_time::Vector{NnzArray{T}};
  kwargs...) where T

  tbu,tbdual... = map(get_nonzero_val,bases_time)
  for (i,tb) in enumerate(tbdual)
    printstyled("Computing supremizers in time for dual field $i\n";color=:blue)
    tbu_i = add_time_supremizers([tbu,tb];kwargs...)
    bases_time[1].nonzero_val = hcat(bases_time[1].nonzero_val,tbu_i)
  end
  return
end

function add_time_supremizers(bases_time::Vector{<:AbstractMatrix};ttol=1e-2)
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

for Trb in (:RBSpace,:TransientRBSpace)
  @eval begin
    function allocate_rb_residual(rb::$Trb)
      rb_ndofs = get_rb_ndofs(rb)
      zeros(rb_ndofs)
    end

    function allocate_rb_jacobian(rb::$Trb)
      rb_ndofs = get_rb_ndofs(rb)
      zeros(rb_ndofs,rb_ndofs)
    end
  end
end
