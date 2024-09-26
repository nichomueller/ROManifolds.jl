abstract type ReductionStyle end

ReductionStyle(args...;kwargs...) = @abstractmethod

struct SearchSVDRank <: ReductionStyle
  tol::Float64
end

struct FixedSVDRank <: ReductionStyle
  rank::Int
end

struct LRApproxRank <: ReductionStyle
  opts::LRAOptions
end

function LRApproxRank(tol::Float64;maxdet_tol=0.,sketch_randn_niter=1,sketch=:sprn,kwargs...)
  opts = LRAOptions(;tol=tol,maxdet_tol,sketch_randn_niter,sketch,kwargs...)
  return LRApproxRank(opts)
end

function LRApproxRank(rank::Int;maxdet_tol=0.,sketch_randn_niter=1,sketch=:sprn,kwargs...)
  opts = LRAOptions(;rank=rank,maxdet_tol,sketch_randn_niter,sketch,kwargs...)
  return LRApproxRank(opts)
end

function ReductionStyle(tol::Float64;sketch=nothing,kwargs...)
  isa(sketch,Symbol) ? LRApproxRank(tol;sketch,kwargs...) : SearchSVDRank(tol)
end

function ReductionStyle(rank::Int;sketch=nothing,kwargs...)
  isa(sketch,Symbol) ? LRApproxRank(rank;sketch,kwargs...) : FixedSVDRank(rank)
end

struct TTSVDRanks{A<:ReductionStyle} <: ReductionStyle
  style::Vector{A}
end

function TTSVDRanks(tolranks::Vector{<:Union{Float64,Int}},args...;kwargs...)
  style = map(tolrank -> ReductionStyle(tolrank,args...;kwargs...),tolranks)
  TTSVDRanks(style)
end

function TTSVDRanks(tolrank::Union{Float64,Int},D=3,args...;kwargs...)
  TTSVDRanks(fill(tolrank,D),args...;kwargs...)
end

ReductionStyle(tolrank::Union{Float64,Int};kwargs...) = TTSVDRanks(tolrank;kwargs...)
ReductionStyle(tolrank::Vector{<:Union{Float64,Int}};kwargs...) = TTSVDRanks(tolrank;kwargs...)

Base.size(r::TTSVDRanks) = (length(r.style),)
Base.getindex(r::TTSVDRanks,i::Integer) = getindex(r.style,i)

abstract type NormStyle end

get_norm(n::NormStyle) = @abstractmethod

struct EuclideanNorm <: NormStyle end

struct EnergyNorm <: NormStyle
  norm_op::Function
end

get_norm(n::EnergyNorm) = n.norm_op

abstract type Reduction{A<:ReductionStyle,B<:NormStyle} end
abstract type DirectReduction{A,B} <: Reduction{A,B} end
abstract type GreedyReduction{A,B} <: Reduction{A,B} end

get_reduction(r::Reduction) = r
ReductionStyle(r::Reduction) = @abstractmethod
NormStyle(r::Reduction) = @abstractmethod
ParamDataStructures.num_params(r::Reduction) = @abstractmethod
get_norm(r::Reduction) = get_norm(NormStyle(r))

struct AffineReduction{A,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
end

function AffineReduction(red_style::ReductionStyle,norm_op::Function)
  norm_style = EnergyNorm(norm_op)
  AffineReduction(red_style,norm_style)
end

function AffineReduction(tol::Float64,norm_style=EuclideanNorm())
  red_style = SearchSVDRank(tol)
  AffineReduction(red_style,norm_style)
end

function AffineReduction(rank::Int,norm_style=EuclideanNorm())
  red_style = FixedSVDRank(rank)
  AffineReduction(red_style,norm_style)
end

ReductionStyle(r::AffineReduction) = r.red_style
NormStyle(r::AffineReduction) = r.norm_style
ParamDataStructures.num_params(r::AffineReduction) = 1

struct PODReduction{A,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
  nparams::Int
end

function PODReduction(red_style::ReductionStyle,norm_style::NormStyle=EuclideanNorm();nparams=50)
  iszero(nparams) && return AffineReduction(red_style,norm_style)
  PODReduction(red_style,norm_style,nparams)
end

function PODReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  PODReduction(red_style,norm_style;kwargs...)
end

function PODReduction(tol::Float64,args...;kwargs...)
  red_style = SearchSVDRank(tol)
  PODReduction(red_style,args...;kwargs...)
end

function PODReduction(rank::Int,args...;kwargs...)
  red_style = FixedSVDRank(rank)
  PODReduction(red_style,args...;kwargs...)
end

ReductionStyle(r::PODReduction) = r.red_style
NormStyle(r::PODReduction) = r.norm_style
ParamDataStructures.num_params(r::PODReduction) = r.nparams

struct TTSVDReduction{A,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
  nparams::Int
end

function TTSVDReduction(red_style::ReductionStyle,norm_style::NormStyle=EuclideanNorm();nparams=50)
  TTSVDReduction(red_style,norm_style,nparams)
end

function TTSVDReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  TTSVDReduction(red_style,norm_style;kwargs...)
end

function TTSVDReduction(tolranks::Union{Vector{Float64},Vector{Int}},args...;kwargs...)
  red_style = TTSVDRanks(tolranks)
  TTSVDReduction(red_style,args...;kwargs...)
end

function TTSVDReduction(tolrank::Union{Float64,Int},args...;D=3,kwargs...)
  red_style = TTSVDRanks(tolrank;D)
  TTSVDReduction(red_style,args...;kwargs...)
end

Base.size(r::TTSVDReduction) = (length(r.tols),)
Base.getindex(r::TTSVDReduction,i::Integer) = FixedSVDRank(r.ranks[i])

ReductionStyle(r::TTSVDReduction) = r.red_style
NormStyle(r::TTSVDReduction) = r.norm_style
ParamDataStructures.num_params(r::TTSVDReduction) = r.nparams

struct SupremizerReduction{A,R<:Reduction{A,EnergyNorm}} <: Reduction{A,EnergyNorm}
  reduction::R
  supr_op::Function
  supr_tol::Float64
end

function SupremizerReduction(supr_op::Function,args...;supr_tol=1e-2,kwargs...)
  reduction = Reduction(args...;kwargs...)
  SupremizerReduction(reduction,supr_op,supr_tol)
end

get_supr(r::SupremizerReduction) = r.supr_op
get_supr_tol(r::SupremizerReduction) = r.supr_tol

get_reduction(r::SupremizerReduction) = get_reduction(r.reduction)
ReductionStyle(r::SupremizerReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::SupremizerReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::SupremizerReduction) = num_params(get_reduction(r))

# generic constructor

function Reduction(red_style::ReductionStyle,args...;nparams=50,kwargs...)
  PODReduction(red_style,args...;nparams,kwargs...)
end

function Reduction(red_style::TTSVDRanks,args...;nparams=50,kwargs...)
  TTSVDReduction(red_style,args...;nparams,kwargs...)
end

function Reduction(tolrank,args...;kwargs...)
  red_style = ReductionStyle(tolrank;kwargs...)
  Reduction(red_style,args...;kwargs...)
end

function Reduction(supr_op::Function,args...;kwargs...)
  SupremizerReduction(supr_op,args...;kwargs...)
end

abstract type AbstractMDEIMReduction{A} <: Reduction{A,EuclideanNorm} end

struct MDEIMReduction{A,R<:Reduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
end

function MDEIMReduction(red_style::ReductionStyle,args...;kwargs...)
  reduction = Reduction(red_style,args...;kwargs...)
  MDEIMReduction(reduction)
end

get_reduction(r::MDEIMReduction) = get_reduction(r.reduction)
ReductionStyle(r::MDEIMReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::MDEIMReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::MDEIMReduction) = num_params(get_reduction(r))

struct AdaptiveReduction{A,B,R<:DirectReduction{A,B}} <: GreedyReduction{A,B}
  reduction::R
  adaptive_nparams::Int
  adaptive_tol::Float64
  adaptive_maxiter::Int
end

function AdaptiveReduction(
  red_style::ReductionStyle,
  args...;
  adaptive_nparams=10,
  adaptive_tol=1e-2,
  adaptive_maxiter=10)

  reduction = Reduction(red_style,args...;kwargs...)
  AdaptiveReduction(reduction,adaptive_nparams,adaptive_tol,adaptive_maxiter)
end
