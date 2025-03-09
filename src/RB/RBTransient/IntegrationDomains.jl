"""
    struct TransientIntegrationDomain <: IntegrationDomain
      domain_space::IntegrationDomain
      indices_time::Vector{Int32}
    end

Integration domain for a projection operator in a transient problem
"""
struct TransientIntegrationDomain <: IntegrationDomain
  domain_space::IntegrationDomain
  indices_time::Vector{Int32}
end

function RBSteady.vector_domain(
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector)

  domain_space = vector_domain(trian,test,rows)
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
get_integration_domain_space(i::TransientIntegrationDomain) = i.domain_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function get_itimes(i::TransientIntegrationDomain,ids::AbstractVector)::Vector{Int}
  idsi = i.indices_time
  filter(!isnothing,indexin(idsi,ids))
end
