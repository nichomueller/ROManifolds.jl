struct TensorProductRefFE{D,A,B,C,E} <: ReferenceFE{D}
  reffe::A
  prebasis::B
  dof_basis::C
  shapefuns::E

  function TensorProductRefFE(
    reffe::ReferenceFE{D},
    prebasis::B,
    dof_basis::C,
    shapefuns::E
    ) where {D,B,C,E}

    A = typeof(reffe)
    new{D,A,B,C,E}(reffe,prebasis,dof_basis,shapefuns)
  end
end

abstract type TensorProductRefFEName <: ReferenceFEName end

product_reffe(::TensorProductRefFEName) = @abstractmethod

struct TensorProdLagrangian <: TensorProductRefFEName end

product_reffe(::TensorProdLagrangian) = Lagrangian()

const tplagrangian = TensorProdLagrangian()

function ReferenceFEs.ReferenceFE(p::Polytope,name::TensorProductRefFEName,order)
  TensorProductRefFE(p,product_reffe(name),Float64,order)
end

function ReferenceFEs.ReferenceFE(p::Polytope,name::TensorProductRefFEName,::Type{T},order) where T
  TensorProductRefFE(p,product_reffe(name),T,order)
end

function TensorProductRefFE(p::Polytope{D},name::ReferenceFEName,::Type{T},order::Int) where {D,T}
  TensorProductRefFE(p,name,T,tfill(order,Val(D)))
end

function TensorProductRefFE(p::Polytope{D},name::ReferenceFEName,::Type{T},orders) where {D,T}
  tpreffe = ReferenceFE(p,name,T,orders)
  prebasis = TensorProductMonomialBasis(T,p,orders)
  dof_basis = TensorProductDofBases(T,p,name,orders)
  shapefuns = compute_shapefuns(dof_basis,prebasis)
  return TensorProductRefFE(tpreffe,prebasis,dof_basis,shapefuns)
end

ReferenceFEs.num_dofs(r::TensorProductRefFE)      = num_dofs(r.reffe)
ReferenceFEs.get_polytope(r::TensorProductRefFE)  = get_polytope(r.reffe)
ReferenceFEs.get_prebasis(r::TensorProductRefFE)  = r.prebasis
ReferenceFEs.get_dof_basis(r::TensorProductRefFE) = r.dof_basis
ReferenceFEs.Conformity(r::TensorProductRefFE)    = Conformity(r.reffe)
ReferenceFEs.get_face_dofs(r::TensorProductRefFE) = get_face_dofs(r.reffe)
ReferenceFEs.get_shapefuns(r::TensorProductRefFE) = r.shapefuns
ReferenceFEs.get_metadata(r::TensorProductRefFE)  = get_metadata(r.reffe)
ReferenceFEs.get_orders(r::TensorProductRefFE)    = get_orders(r.reffe)
ReferenceFEs.get_order(r::TensorProductRefFE)     = get_order(r.reffe)

ReferenceFEs.Conformity(r::TensorProductRefFE,sym::Symbol) = Conformity(r.reffe,sym)
ReferenceFEs.get_face_own_dofs(r::TensorProductRefFE,conf::Conformity) = get_face_own_dofs(r.reffe,conf)
