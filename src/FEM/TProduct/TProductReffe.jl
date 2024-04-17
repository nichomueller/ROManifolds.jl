struct TensorProductRefFE{D,A,B} <: ReferenceFE{D}
  reffe::NTuple{D,A}
  dof_basis::B

  function TensorProductRefFE(
    reffe::NTuple{D,A},
    dof_basis::TensorProductDofBases{D}
    ) where {D,A<:ReferenceFE{1}}

    B = typeof(dof_basis)
    new{D,A,B}(reffe,dof_basis)
  end
end

abstract type TensorProductRefFEName <: ReferenceFEName end

ureffe(::TensorProductRefFEName) = @abstractmethod

struct TensorProdLagrangian <: TensorProductRefFEName end

ureffe(::TensorProdLagrangian) = Lagrangian()

const tplagrangian = TensorProdLagrangian()

function ReferenceFEs.ReferenceFE(p::Polytope,name::TensorProductRefFEName,order)
  TensorProductRefFE(p,name,Float64,order)
end

function ReferenceFEs.ReferenceFE(p::Polytope,name::TensorProductRefFEName,::Type{T},order) where T
  TensorProductRefFE(p,name,T,order)
end

function TensorProductRefFE(p::Polytope{D},name::TensorProductRefFEName,::Type{T},order::Int) where {D,T}
  TensorProductRefFE(p,name,T,tfill(order,Val(D)))
end

function TensorProductRefFE(p::Polytope{D},name::TensorProductRefFEName,::Type{T},orders) where {D,T}
  @check length(orders) == D
  nodes_map = compute_indices_map(p,orders)

  reffes = ntuple(i->ReferenceFE(SEGMENT,ureffe(name),T,orders[i]),D)
  prebasis = TensorProductMonomialBasis(get_prebasis.(reffes))
  nodes = TensorProductNodes(get_nodes.(reffes),nodes_map)
  dof_basis = TensorProductDofBases(get_dof_basis.(reffes),nodes)
  shapefuns = compute_shapefuns(dof_basis,prebasis)

  return TensorProductRefFE(reffes,dof_basis)
end

ReferenceFEs.num_dofs(reffe::TensorProductRefFE)      = prod(num_dofs.(reffe.reffe))
ReferenceFEs.get_polytope(reffe::TensorProductRefFE)  = Polytope(tfill(HEX_AXIS,Val(D))...)
ReferenceFEs.get_prebasis(reffe::TensorProductRefFE)  = reffe.prebasis
ReferenceFEs.get_dof_basis(reffe::TensorProductRefFE) = reffe.dof_basis
ReferenceFEs.Conformity(reffe::TensorProductRefFE)    = Conformity(first(reffe.reffe))
ReferenceFEs.get_face_dofs(reffe::TensorProductRefFE) = reffe.face_dofs
ReferenceFEs.get_shapefuns(reffe::TensorProductRefFE) = reffe.shapefuns
ReferenceFEs.get_metadata(reffe::TensorProductRefFE)  = reffe.metadata
ReferenceFEs.get_orders(reffe::TensorProductRefFE)    = ntuple(i->get_order(reffe.reffe[i]),D)
ReferenceFEs.get_order(reffe::TensorProductRefFE)     = maximum(get_orders(reffe))

ReferenceFEs.Conformity(reffe::TensorProductRefFE,sym::Symbol) = Conformity(reffe.reffe,sym)
ReferenceFEs.get_face_own_dofs(reffe::TensorProductRefFE,conf::Conformity) = get_face_own_dofs(reffe.reffe,conf)
