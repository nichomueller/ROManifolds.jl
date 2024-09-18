struct TransientPODReduction{A,B} <: AbstractPODReduction{A,B}
  reduction_space::PODReduction{A,B}
  reduction_time::PODReduction{A,EuclideanNorm}
end

function TransientPODReduction(
  tolrank_space::Union{Int,Float64},
  tolrank_time::Union{Int,Float64},
  args...;kwargs...)

  reduction_space = PODReduction(tolrank_space,args...;kwargs...)
  reduction_time = PODReduction(tolrank_time;kwargs...)
  TransientPODReduction(reduction_space,reduction_time)
end

function TransientPODReduction(tolrank::Union{Int,Float64},args...;kwargs...)
  TransientPODReduction(tolrank,tolrank,args...;kwargs...)
end

get_reduction_space(r::TransientPODReduction) = RBSteady.get_reduction(r.reduction_space)
get_reduction_time(r::TransientPODReduction) = RBSteady.get_reduction(r.reduction_time)
RBSteady.ReductionStyle(r::TransientPODReduction) = ReductionStyle(get_reduction_space(r))
RBSteady.NormStyle(r::TransientPODReduction) = NormStyle(get_reduction_space(r))
ParamDataStructures.num_params(r::TransientPODReduction) = num_params(get_reduction_space(r))

# generic constructor

function TransientReduction(red_style::Union{SearchSVDRank,FixedSVDRank},args...;kwargs...)
  reduction_space = PODReduction(red_style,args...;kwargs...)
  reduction_time = PODReduction(red_style;kwargs...)
  TransientPODReduction(reduction_space,reduction_time)
end

function TransientReduction(red_style::TTSVDRanks,args...;kwargs...)
  TTSVDReduction(red_style,args...;kwargs...)
end

struct TransientMDEIMReduction{A,R<:AbstractReduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  combine::Function
  nparams_test::Int
end

function TransientMDEIMReduction(combine::Function,args...;nparams_test=10,kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientMDEIMReduction(reduction,combine,nparams_test)
end

RBSteady.get_reduction(r::TransientMDEIMReduction) = RBSteady.get_reduction(r.reduction)
RBSteady.ReductionStyle(r::TransientMDEIMReduction) = ReductionStyle(RBSteady.get_reduction(r))
RBSteady.NormStyle(r::TransientMDEIMReduction) = NormStyle(RBSteady.get_reduction(r))
ParamDataStructures.num_params(r::TransientMDEIMReduction) = num_params(RBSteady.get_reduction(r))
RBSteady.num_online_params(r::TransientMDEIMReduction) = r.nparams_test
