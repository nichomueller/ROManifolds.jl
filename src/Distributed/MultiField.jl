const DistributedMultiFieldParamFESpace = GridapDistributed.DistributedMultiFieldFESpace{MS,<:AbstractVector{<:MultiFieldParamFESpace},B,C,D} where {MS,B,C,D}
const DistributedMultiFieldParamFEFunction = GridapDistributed.DistributedMultiFieldFEFunction{<:AbstractVector{<:SingleFieldParamFEFunction},B,C} where {B,C}

const DistributedParamFESpace = Union{DistributedSingleFieldParamFESpace,DistributedMultiFieldParamFESpace}
