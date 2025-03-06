function empirical_interpolation!(cache,A::AbstractMatrix)
  I,res = cache
  m,n = size(A)
  resize!(res,m)
  resize!(I,n)
  @views I[1] = argmax(abs.(A[:,1]))
  if n > 1
    @inbounds for i = 2:n
      @views Bi = A[:,1:i-1]
      Ci = A[I[1:i-1],1:i-1]
      Di = A[I[1:i-1],i]
      @views res = A[:,i] - Bi*(Ci \ Di)
      I[i] = argmax(abs.(res))
    end
  end
  Ai = view(A,I,:)
  return I,Ai
end

function eim_cache(A::AbstractMatrix)
  m,n = size(A)
  res = zeros(eltype(A),m)
  I = zeros(Int,n)
  return I,res
end

function empirical_interpolation(A::AbstractArray)
  cache = eim_cache(A)
  I,AI = empirical_interpolation!(cache,A)
  return I,AI
end

function empirical_interpolation(A::ParamSparseMatrix)
  I,AI = empirical_interpolation(A.data)
  R′,C′ = recast_split_indices(I,param_getindex(A,1))
  return (R′,C′),AI
end

"""
    get_dofs_to_cells(
      cell_dof_ids::AbstractArray{<:AbstractArray},
      dofs::AbstractVector
      ) -> AbstractVector

Returns the list of cells containing the dof ids `dofs`
"""
function get_dofs_to_cells(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  dofs::AbstractVector)

  cells = Int32[]
  cache = array_cache(cell_dof_ids)
  for cell = eachindex(cell_dof_ids)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    if !isempty(intersect(dofs,celldofs))
      append!(cells,cell)
    end
  end
  return unique(cells)
end

function get_cells_to_idofs(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  dofs::AbstractVector)

  cache = array_cache(cell_dof_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    ptrs[icell+1] = length(celldofs)
  end
  length_to_ptrs!(ptrs)

  data = fill(zero(Int32),ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    for (idof,dof) in enumerate(dofs)
      for (icelldof,celldof) in enumerate(celldofs)
        if dof == celldof
          data[ptrs[icell]-1+icelldof] = idof
        end
      end
    end
  end

  Table(data,ptrs)
end

function reduced_cells(
  f::FESpace,
  trian::Triangulation,
  dofs::AbstractVector)

  cell_dof_ids = get_cell_dof_ids(f,trian)
  cells = get_dofs_to_cells(cell_dof_ids,dofs)
  return cells
end

function reduced_idofs(
  f::FESpace,
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  cell_dof_ids = get_cell_dof_ids(f,trian)
  idofs = get_cells_to_idofs(cell_dof_ids,cells,dofs)
  return idofs
end

"""
    abstract type IntegrationDomain end

Type representing the set of interpolation rows of a `Projection` subjected
to a EIM approximation with `empirical_interpolation`.
Subtypes:
- [`VectorDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)
"""
abstract type IntegrationDomain end

get_integration_cells(i::IntegrationDomain) = @abstractmethod
get_cellids_rows(i::IntegrationDomain) = @abstractmethod
get_cellids_cols(i::IntegrationDomain) = @abstractmethod

"""
    struct VectorDomain <: IntegrationDomain{Int,1}
      cells::Vector{Int32}
      cell_irows::Table{Int32,Vector{Int32},Vector{Int32}}
    end

Integration domain for a projection vector operator in a steady problem
"""
struct VectorDomain <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::Table{Int32,Vector{Int32},Vector{Int32}}
end

get_integration_cells(i::VectorDomain) = i.cells
get_cellids_rows(i::VectorDomain) = i.cell_irows

function vector_domain(args...)
  @abstractmethod
end

function vector_domain(
  trian::Triangulation,
  test::FESpace,
  rows::Vector{<:Number})

  cells = reduced_cells(test,trian,rows)
  irows = reduced_idofs(test,trian,cells,rows)
  VectorDomain(cells,irows)
end

"""
    struct MatrixDomain <: IntegrationDomain{Int,1}
      cells::Vector{Int32}
      cell_irows::Table{Int32,Vector{Int32},Vector{Int32}}
      cell_icols::Table{Int32,Vector{Int32},Vector{Int32}}
    end

Integration domain for a projection vector operator in a steady problem
"""
struct MatrixDomain <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::Table{Int32,Vector{Int32},Vector{Int32}}
  cell_icols::Table{Int32,Vector{Int32},Vector{Int32}}
end

function matrix_domain(args...)
  @abstractmethod
end

function matrix_domain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::Vector{<:Number},
  cols::Vector{<:Number})

  cells_trial = reduced_cells(trial,trian,cols)
  cells_test = reduced_cells(test,trian,rows)
  cells = unique(vcat(cells_trial,cells_test))
  icols = reduced_idofs(trial,trian,cells,cols)
  irows = reduced_idofs(test,trian,cells,rows)
  MatrixDomain(cells,irows,icols)
end

get_integration_cells(i::MatrixDomain) = i.cells
get_cellids_rows(i::MatrixDomain) = i.cell_irows
get_cellids_cols(i::MatrixDomain) = i.cell_icols
