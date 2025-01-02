TransientReductionStyle(args...;kwargs...) = @abstractmethod

function TransientReductionStyle(tolrank_space,tolrank_time;kwargs...)
  style_space = ReductionStyle(tol_space;kwargs...)
  style_time = ReductionStyle(tol_time;kwargs...)
  return style_space,style_time
end

function TransientReductionStyle(tolrank;kwargs...)
  TransientReductionStyle(tolrank,tolrank;kwargs...)
end

"""
    struct TransientReduction{A,B,RS<:Reduction{A,B},RT<:Reduction{A,EuclideanNorm}} <: Reduction{A,B}
      reduction_space::RS
      reduction_time::RT
    end

Wrapper for reduction methods in transient problems. The fields `reduction_space`
and `reduction_time` respectively represent the spatial reduction method, and
the temporal one
"""
struct TransientReduction{A,B,RS<:Reduction{A,B},RT<:Reduction{A,EuclideanNorm}} <: Reduction{A,B}
  reduction_space::RS
  reduction_time::RT
end

const TransientAffineReduction{A,B} = TransientReduction{A,B,AffineReduction{A,B},AffineReduction{A,EuclideanNorm}}
const TransientPODReduction{A,B} = TransientReduction{A,B,PODReduction{A,B},PODReduction{A,EuclideanNorm}}

# generic constructor

function TransientReduction(style_space::ReductionStyle,style_time::ReductionStyle,args...;kwargs...)
  reduction_space = Reduction(style_space,args...;kwargs...)
  reduction_time = Reduction(style_time;kwargs...)
  TransientReduction(reduction_space,reduction_time)
end

function TransientReduction(red_style::ReductionStyle,args...;kwargs...)
  TransientReduction(red_style,red_style,args...;kwargs...)
end

function TransientReduction(red_style::TTSVDRanks,args...;kwargs...)
  TTSVDReduction(red_style,args...;kwargs...)
end

function TransientReduction(
  tolrank_space::Union{Int,Float64},
  tolrank_time::Union{Int,Float64},
  args...;kwargs...)

  reduction_space = Reduction(tolrank_space,args...;kwargs...)
  reduction_time = Reduction(tolrank_time;kwargs...)
  TransientReduction(reduction_space,reduction_time)
end

function TransientReduction(tolrank::Union{Int,Float64},args...;kwargs...)
  TransientReduction(tolrank,tolrank,args...;kwargs...)
end

function TransientReduction(tolrank::Union{Vector{Int},Vector{Float64}},args...;kwargs...)
  TTSVDReduction(tolrank,args...;kwargs...)
end

function TransientReduction(supr_op::Function,args...;supr_tol=1e-2,kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  SupremizerReduction(reduction,supr_op,supr_tol)
end

get_reduction_space(r::TransientReduction) = get_reduction(r.reduction_space)
get_reduction_time(r::TransientReduction) = get_reduction(r.reduction_time)
RBSteady.ReductionStyle(r::TransientReduction) = ReductionStyle(get_reduction_space(r))
RBSteady.NormStyle(r::TransientReduction) = NormStyle(get_reduction_space(r))
ParamDataStructures.num_params(r::TransientReduction) = num_params(get_reduction_space(r))

"""
    struct TransientMDEIMReduction{A,R<:Reduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
      reduction::R
      combine::Function
    end

MDEIM struct employed in transient problems. The field `combine` is a function
used to group the reductions relative to the various jacobians (in general, more
than one in transient problems) in a smart way. We consider, for example, the ODE

du/dt - νΔu = f in Ω × [0,T]

subject to initial/boundary conditions. Upon applying a FE discretization in space,
and a θ method in time, one gets the space-time system Aθ * uθ = fθ, where

Aθ = [
A₁ + M / (θ*Δt)          0               0          ⋯          0            0
   - M / (θ*Δt)    A₂ + M / (θ*Δt)       0          ⋯          0            0
       0              - M / (θ*Δt)  A₃ + M / (θ*Δt)            0            0
       ⋱                 ⋱              ⋱          ⋯          ⋯            ⋯
       0                  0              0          ⋯      - M / (θ*Δt)  Aₙ + M / (θ*Δt)
                                                                                        ]
   = tridiag(- M/(θ*Δt), Aₖ + M/(θ*Δt), 0)
uθ = [(1-θ)u₀ + θu₁, (1-θ)u₁ + θu₂ ⋯ (1-θ)uₙ₋₁ + θuₙ]
fθ = [f₁, f₂  ⋯ fₙ]

where Aₖ = A(tₖ₋₁ + θ*Δt) and fₖ = f(tₖ₋₁ + θ*Δt).

Note: instead of multiplying Aθ by uθ, we multiply Ãθ by u, where

Ãθ = tridiag((1-θ)Aₖ₋₁ - θM/(θ*Δt), θAₖ + θM/(θ*Δt))
u = [u₁, u₂ ⋯ uₙ]

We now denote with Φ, Ψ the spatial and temporal basis obtained by reducing the
snapshots associated to the state variable u. The Galerkin projection of the
space-time system is equal to Âθ * û = f̂θ, where û is the unknown, and

Âθ = ∑ₖⁿ⁻¹ ( (1-θ)*ΦᵀAₖΦ - θ*ΦᵀMΦ / (θ*Δt) ) ⊗ Ψ[k-1,:]ᵀΨ[k,:]
   + ∑ₖⁿ   (     θ*ΦᵀAₖΦ + θ*ΦᵀMΦ / (θ*Δt) ) ⊗ Ψ[k,:]ᵀΨ[k,:]
f̂θ = ∑ₖⁿ Φᵀfₖ ⊗ Ψ[k,:]

We notice that the expression of Âθ can be written in a more general form as

Âθ = combine_A(A shift back, A) + combine_M(M shift back, M)

where combine_A and combine_M are two function specific to A and M:

combine_A(x,y) = (1-θ)*x + θ*y
combine_M(x,y) = -θ*x + θ*y

The same can be said of any time marching scheme. This is the meaning of the
function combine. Note that for a time marching with p interpolation points (e.g.
for θ method, p = 2) the combine functions will have to accept p arguments.
"""
struct TransientMDEIMReduction{A,R<:Reduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
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
