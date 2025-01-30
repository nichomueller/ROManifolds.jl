"""
    struct OIdsToIds{T,A<:AbstractVector{<:Integer}} <: AbstractVector{T}
      indices::Vector{T}
      terms::A
    end
"""
struct OIdsToIds{T,A<:AbstractVector{<:Integer}} <: AbstractVector{T}
  indices::Vector{T}
  terms::A
end

Base.size(a::OIdsToIds) = size(a.indices)
Base.getindex(a::OIdsToIds,i::Integer) = getindex(a.indices,i)
Base.setindex!(a::OIdsToIds,v,i::Integer) = setindex!(a.indices,v,i)

function Base.similar(a::OIdsToIds{T},::Type{T′},s::Dims{1}) where {T,T′}
  indices′ = similar(a.indices,T′,s...)
  OIdsToIds(indices′,a.terms)
end

"""
    struct DofsToODofs{D,P,V} <: Map
      b::LagrangianDofBasis{P,V}
      odof_to_dof::Vector{Int32}
      node_and_comps_to_odof::Array{V,D}
      orders::NTuple{D,Int}
    end

Map used to convert a DOF of a standard FE space to a lexicographically-ordered DOF
"""
struct DofsToODofs{D,P,V} <: Map
  b::LagrangianDofBasis{P,V}
  odof_to_dof::Vector{Int32}
  node_and_comps_to_odof::Array{V,D}
  orders::NTuple{D,Int}
end

function DofsToODofs(
  b::LagrangianDofBasis,
  node_and_comps_to_odof::AbstractArray,
  orders::Tuple)

  odof_to_dof = _local_odof_to_dof(b,orders)
  DofsToODofs(b,odof_to_dof,node_and_comps_to_odof,orders)
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

function get_odof(k::DofsToODofs{D,P},dof::Integer) where {D,P}
  nnodes = length(k.node_and_comps_to_odof)
  comp = slow_index(dof,nnodes)
  node = fast_index(dof,nnodes)
  k.node_and_comps_to_odof[node][comp]
end

function Arrays.return_cache(k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  local_ndofs = length(k.odof_to_dof)
  odofs = OIdsToIds(zeros(Int32,local_ndofs),k.odof_to_dof)
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

function Arrays.return_value(k::Broadcasting{typeof(_sum_if_first_positive)},dofs::OIdsToIds,o::Integer)
  evaluate(k,dofs,o)
end

function Arrays.return_cache(k::Broadcasting{typeof(_sum_if_first_positive)},dofs::OIdsToIds,o::Integer)
  c = return_cache(k,dofs.indices,o)
  odofs = OIdsToIds(evaluate(k,dofs.indices,o),dofs.terms)
  c,odofs
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(_sum_if_first_positive)},dofs::OIdsToIds,o::Integer)
  c,odofs = cache
  r = evaluate!(c,k,dofs.indices,o)
  copyto!(odofs.indices,r)
  odofs
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

"""
    add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)

Adds several ordered entries only for positive input indices. Returns `A`
"""
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

"""
    struct OReindex{T<:Integer} <: Map
      indices::Vector{T}
    end

Map used to reindex according to the vector of integers `indices`
"""
struct OReindex{T<:Integer} <: Map
  indices::Vector{T}
end

function Arrays.return_value(k::OReindex,values)
  values
end

function Arrays.return_cache(k::OReindex,values::AbstractVector)
  @check length(values) == length(k.indices)
  similar(values)
end

function Arrays.evaluate!(cache,k::OReindex,values::AbstractVector)
  for (i,oi) in enumerate(k.indices)
    cache[oi] = values[i]
  end
  return cache
end
