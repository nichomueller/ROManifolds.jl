struct OIdsToIds{T,A<:AbstractVector{<:Integer}} <: AbstractVector{T}
  indices::Vector{T}
  terms::A
end

Base.size(a::OIdsToIds) = size(a.indices)
Base.getindex(a::OIdsToIds,i::Integer) = getindex(a.indices,i)
Base.setindex!(a::OIdsToIds,v,i::Integer) = setindex!(a.indices,v,i)

struct DofsToODofs{D,P,V} <: Map
  b::LagrangianDofBasis{P,V}
  pdof_to_dof::Vector{Int32}
  node_and_comps_to_odof::Array{V,D}
  orders::NTuple{D,Int}
end

function DofsToODofs(
  b::LagrangianDofBasis,
  node_and_comps_to_odof::AbstractArray,
  orders::Tuple)

  pdof_to_dof = _local_pdof_to_dof(b,orders)
  DofsToODofs(b,pdof_to_dof,node_and_comps_to_odof,orders)
end

function DofsToODofs(fe_dof_basis::Fill{<:LagrangianDofBasis},args...)
  DofsToODofs(testitem(fe_dof_basis),args...)
end

function DofsToODofs(fe_dof_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function get_ndofs(k::DofsToODofs{D,P}) where {D,P}
  ncomps = num_components(P)
  nnodes = length(k.node_and_comps_to_odof)
  ncomps*nnodes
end

function Arrays.return_cache(k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  local_ndofs = length(k.pdof_to_dof)
  odofs = OIdsToIds(zeros(Int32,local_ndofs),k.pdof_to_dof)
  return odofs
end

function Arrays.evaluate!(cache,k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  first_new_node = k.orders .* (Tuple(cell) .- 1) .+ 1
  onodes_range = map(enumerate(first_new_node)) do (i,ni)
    ni:ni+k.orders[i]
  end
  local_comps_to_odofs = view(k.node_and_comps_to_odof,onodes_range...)
  local_nnodes = length(k.b.node_and_comp_to_dof)
  for (node,comps_to_odof) in enumerate(local_comps_to_odofs)
    for comp in k.b.dof_to_comp
      odof = comps_to_odof[comp]
      cache[node+(comp-1)*local_nnodes] = odof
    end
  end
  return cache
end

# Assembly-related functions

@inline function Algebra.add_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  add_ordered_entries!(combine,A,vs,is,js)
end

for T in (:Any,:(Algebra.ArrayCounter))
  @eval begin
    @inline function Algebra.add_entries!(combine::Function,A::$T,vs,is::OIdsToIds)
      add_ordered_entries!(combine,A,vs,is)
    end
  end
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds,js::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.indices,js.indices)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  for (lj,j) in enumerate(js)
    if j>0
      ljp = js.terms[lj]
      for (li,i) in enumerate(is)
        if i>0
          lip = is.terms[li]
          vij = vs[lip,ljp]
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.indices)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds)
  for (li,i) in enumerate(is)
    if i>0
      lip = is.terms[li]
      vi = vs[lip]
      add_entry!(A,vi,i)
    end
  end
  A
end
