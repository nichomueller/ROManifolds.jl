abstract type RBFunction <: FEFunction end

abstract type RBSpace <: FESpace end

abstract type RBBasis <: FEBasis end

struct SingleFieldRBBasis{BS<:BasisStyle} <: RBBasis
  cell_basis::AbstractArray
  trian::Triangulation
  basis_style::BS

  function SingleFieldRBBasis(
    cell_basis::AbstractArray,
    trian::Triangulation,
    basis_style::BasisStyle)

    BS = typeof(basis_style)
    new{BS}(Fields.MemoArray(cell_basis),trian,basis_style)
  end
end

get_data(f::SingleFieldRBBasis) = f.cell_basis

get_triangulation(f::SingleFieldRBBasis) = f.trian

BasisStyle(::Type{SingleFieldRBBasis{BS}}) where BS = BS()

DomainStyle(::Type{SingleFieldRBBasis{BS}}) where BS = PhysicalDomain()

function CellData.similar_cell_field(f::SingleFieldRBBasis,cell_data,trian,args...)
  SingleFieldRBBasis(cell_data,trian,BasisStyle(f))
end

function similar_fe_basis(::SingleFieldRBBasis,cell_data,trian,bs::BasisStyle,args...)
  SingleFieldRBBasis(cell_data,trian,bs)
end
