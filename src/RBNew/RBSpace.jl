abstract type RBSpace <: FESpace end

struct SingleFieldRBSpace{S<:SingleFieldFESpace,M<:AbstractMatrix} <: RBSpace
  space::S
  basis::M
end
