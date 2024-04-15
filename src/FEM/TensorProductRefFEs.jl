struct TensorProductRefFE{C,D,A,B} <: ReferenceFE{D}
  reffe::A
  tpreffe::B
  function TensorProductRefFE(
    reffes::NTuple{D,ReferenceFE{1}},
    tpreffe::ReferenceFE{D}) where D
    C = typeof(Conformity(tpreffe))
    A = typeof(reffes)
    B = typeof(tpreffe)
    new{C,D,A,B}(reffes,tpreffe)
  end
end

struct TensorProdLagrangian <: ReferenceFEName end

const tplagrangian = TensorProdLagrangian()

function ReferenceFEs.ReferenceFE(
  polytope::Polytope{D},
  ::TensorProdLagrangian,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}};
  kwargs...) where {T,D}

  reffes = ntuple(i->ReferenceFE(SEGMENT,lagrangian,T,orders;kwargs...),D)
  tpp = Polytope(ntuple(i->HEX_AXIS,D)...)
  tpreffe = ReferenceFE(tpp,lagrangian,T,orders;kwargs...)
  TensorProductRefFE(reffes,tpreffe)
end

get_separate_order(r::TensorProductRefFE) = get_order.(r.reffes)
get_separate_orders(r::TensorProductRefFE) = get_orders.(r.reffes)
num_separate_dofs(r::TensorProductRefFE) = num_dofs.(r.reffes)
num_separate_polytope(r::TensorProductRefFE) = get_polytope.(r.reffes)
get_separate_prebasis(r::TensorProductRefFE) = get_prebasis.(r.reffes)
get_separate_dof_basis(r::TensorProductRefFE) = get_dof_basis.(r.reffes)
get_separate_shapefuns(r::TensorProductRefFE) = get_shapefuns.(r.reffes)
get_separate_face_dofs(r::TensorProductRefFE) = get_face_dofs.(r.reffes)
get_separate_face_nodes(r::TensorProductRefFE) = get_face_nodes.(r.reffes)
get_separate_face_own_dofs(r::TensorProductRefFE,conf::Conformity) = get_face_own_dofs.(r.reffes,conf)

ReferenceFEs.get_order(r::TensorProductRefFE) = get_order(r.tpreffe)
ReferenceFEs.get_orders(r::TensorProductRefFE) = get_orders(r.tpreffe)
ReferenceFEs.num_dofs(r::TensorProductRefFE) = num_dofs(r.tpreffe)
ReferenceFEs.get_polytope(r::TensorProductRefFE)  = get_polytope(r.tpreffe)
ReferenceFEs.get_prebasis(r::TensorProductRefFE) = get_prebasis(r.tpreffe)
ReferenceFEs.get_dof_basis(r::TensorProductRefFE) = get_dof_basis(r.tpreffe)
ReferenceFEs.Conformity(r::TensorProductRefFE) = Conformity(r.tpreffe)
ReferenceFEs.get_face_dofs(r::TensorProductRefFE) = get_face_dofs(r.tpreffe)
ReferenceFEs.get_face_nodes(r::TensorProductRefFE) = get_face_nodes(r.tpreffe)
ReferenceFEs.get_face_own_dofs(r::TensorProductRefFE,conf::Conformity) = get_face_own_dofs(r.tpreffe,conf)

function ReferenceFEs.Conformity(r::TensorProductRefFE{GradConformity},sym::Symbol)
  h1 = (:H1,:C0,:Hgrad)
  if sym == :L2
    L2Conformity()
  elseif sym in h1
    H1Conformity()
  else
    @unreachable """\n
    It is not possible to use conformity = $sym on a TensorProductRefFE with H1 conformity.

    Possible values of conformity for this reference fe are $((:L2, h1...)).
    """
  end
end

# struct TP end

# function ReferenceFEs.get_cell_dof_basis(
#   model::DiscreteModel,
#   cell_reffe::AbstractArray{<:TensorProductRefFE},
#   ::Conformity)

#   lazy_map(get_dof_basis,cell_reffe)
# end

# function ReferenceFEs.get_cell_shapefuns(
#   model::DiscreteModel,
#   cell_reffe::AbstractArray{<:TensorProductRefFE},
#   ::Conformity)

#   lazy_map(get_shapefuns,cell_reffe)
# end
