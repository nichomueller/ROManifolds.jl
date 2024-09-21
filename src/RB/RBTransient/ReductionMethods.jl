struct TransientReduction{A,B,RS<:AbstractReduction{A,B},RT<:AbstractReduction{A,EuclideanNorm}} <: AbstractReduction{A,B}
  reduction_space::RS
  reduction_time::RT
end

const TransientAffineReduction{A,B} = TransientReduction{A,B,AffineReduction{A,B},AffineReduction{A,EuclideanNorm}}
const TransientPODReduction{A,B} = TransientReduction{A,B,PODReduction{A,B},PODReduction{A,EuclideanNorm}}

# generic constructor

function TransientReduction(style_space::ReductionStyle,style_time::ReductionStyle,args...;kwargs...)
  reduction_space = RBSteady.Reduction(style_space,args...;kwargs...)
  reduction_time = RBSteady.Reduction(style_time;kwargs...)
  TransientReduction(reduction_space,reduction_time)
end

function TransientReduction(red_style::ReductionStyle,args...;kwargs...)
  TransientReduction(red_style,red_style,args...;kwargs...)
end

function TransientReduction(
  tolrank_space::Union{Int,Float64},
  tolrank_time::Union{Int,Float64},
  args...;kwargs...)

  reduction_space = PODReduction(tolrank_space,args...;kwargs...)
  reduction_time = PODReduction(tolrank_time;kwargs...)
  TransientReduction(reduction_space,reduction_time)
end

function TransientReduction(tolrank::Union{Int,Float64},args...;kwargs...)
  TransientReduction(tolrank,tolrank,args...;kwargs...)
end

function TransientReduction(tolrank::Union{Vector{Int},Vector{Float64}},args...;kwargs...)
  TTSVDReduction(tolrank,args...;kwargs...)
end

get_reduction_space(r::TransientReduction) = get_reduction(r.reduction_space)
get_reduction_time(r::TransientReduction) = get_reduction(r.reduction_time)
RBSteady.ReductionStyle(r::TransientReduction) = ReductionStyle(get_reduction_space(r))
RBSteady.NormStyle(r::TransientReduction) = NormStyle(get_reduction_space(r))
ParamDataStructures.num_params(r::TransientReduction) = num_params(get_reduction_space(r))

struct TransientMDEIMReduction{A,R<:AbstractReduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  combine::Function
end

function TransientMDEIMReduction(combine::Function,args...;kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientMDEIMReduction(reduction,combine)
end

RBSteady.get_reduction(r::TransientMDEIMReduction) = get_reduction(r.reduction)
RBSteady.ReductionStyle(r::TransientMDEIMReduction) = ReductionStyle(get_reduction(r))
RBSteady.NormStyle(r::TransientMDEIMReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::TransientMDEIMReduction) = num_params(get_reduction(r))
get_combine(r::TransientMDEIMReduction) = r.combine
