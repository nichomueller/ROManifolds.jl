"""
    abstract type TransientIntegrationDomain <: IntegrationDomain end

Integration domain for a projection operator in a transient problem.
Subtypes:
- [`KroneckerIntegrationDomain`](@ref)
- [`LinearIntegrationDomain`](@ref)
"""
abstract type TransientIntegrationDomain <: IntegrationDomain end

get_indices_time(i::TransientIntegrationDomain) = @abstractmethod

function get_itimes(i::TransientIntegrationDomain,ids::AbstractVector)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end

"""
    struct KroneckerIntegrationDomain <: IntegrationDomain
      domain_space::IntegrationDomain
      indices_time::Vector{Int32}
    end

Integration domain for a projection operator in a transient problem where the
HR operator consists of the tensor product between a spatial HR operator and a
tempoal one
"""
struct KroneckerIntegrationDomain <: IntegrationDomain
  domain_space::IntegrationDomain
  indices_time::Vector{Int32}
end

function kron_vector_domain(
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector)

  domain_space = vector_domain(trian,test,rows)
  KroneckerIntegrationDomain(domain_space,indices_time)
end

function kron_matrix_domain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector)

  domain_space = matrix_domain(trian,trial,test,rows,cols)
  KroneckerIntegrationDomain(domain_space,indices_time)
end

RBSteady.get_integration_cells(i::KroneckerIntegrationDomain) = get_integration_cells(i.domain_space)
RBSteady.get_cellids_rows(i::KroneckerIntegrationDomain) = get_cellids_rows(i.domain_space)
RBSteady.get_cellids_cols(i::KroneckerIntegrationDomain) = get_cellids_cols(i.domain_space)
get_indices_time(i::KroneckerIntegrationDomain) = i.indices_time

function get_cells_to_idofs_spacetime(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  dofs::AbstractVector,
  times::AbstractVector)

  @assert length(dofs) == length(times) "For this integration domain to work, the
  number of spatial selected by the EIM procedure should be equal to the number of
  temporal entries selected by the EIM procedure"

  _correct_idof(is,li) = li
  _correct_idof(is::OIdsToIds,li) = is.terms[li]

  cache = array_cache(cell_dof_ids)
  itimes = Int32(1):Int32(length(times))

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    ptrs[icell+1] = length(celldofs)
  end
  length_to_ptrs!(ptrs)

  z = zero(VectorValue{2,Int32})
  data = fill(z,ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    for (idof,dof) in enumerate(dofs)
      for (_icelldof,celldof) in enumerate(celldofs)
        if dof == celldof
          icelldof = _correct_idof(celldofs,_icelldof)
          data[ptrs[icell]-1+icelldof] = VectorValue(idof,itimes[idof])
        end
      end
    end
  end

  Table(data,ptrs)
end

abstract type LinearIntegrationDomain <: TransientIntegrationDomain end

struct LinVectorIntegrationDomain <: LinearIntegrationDomain
  cells::Vector{Int32}
  indices_time::Vector{Int32}
  cell_irows::Table{VectorValue{2,Int32},Vector{VectorValue{2,Int32}},Vector{Int32}}
end

struct LinMatrixIntegrationDomain <: LinearIntegrationDomain
  cells::Vector{Int32}
  indices_time::Vector{Int32}
  cell_irows::Table{VectorValue{2,Int32},Vector{VectorValue{2,Int32}},Vector{Int32}}
  cell_icols::Table{VectorValue{2,Int32},Vector{VectorValue{2,Int32}},Vector{Int32}}
end
