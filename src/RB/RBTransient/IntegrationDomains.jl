"""
    struct TransientIntegrationDomain <: IntegrationDomain
      domain_space::IntegrationDomain
      indices_time::Vector{Int}
    end

Integration domain for a projection operator in a transient problem
"""
struct TransientIntegrationDomain <: IntegrationDomain
  domain_space::IntegrationDomain
  indices_time::Vector{Int}
end

function RBSteady.vector_domain(
  test::FESpace,
  trian::Triangulation,
  indices::Union{Tuple,AbstractVector})

  @check length(indices) == 2
  indices_space,indices_time = indices
  domain_space = vector_domain(test,trian,indices_space)
  TransientIntegrationDomain(domain_space,indices_time)
end

function RBSteady.matrix_domain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector)

  domain_space = matrix_domain(trian,trial,test,rows,cols)
  TransientIntegrationDomain(domain_space,indices_time)
end

RBSteady.get_integration_cells(i::TransientIntegrationDomain) = get_integration_cells(i.domain_space)
RBSteady.get_cellids_rows(i::TransientIntegrationDomain) = get_cellids_rows(i.domain_space)
RBSteady.get_cellids_cols(i::TransientIntegrationDomain) = get_cellids_cols(i.domain_space)
