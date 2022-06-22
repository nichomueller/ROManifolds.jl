abstract type RBProblem <: Problem end
abstract type RBSteadyProblem <: RBProblem end
abstract type RBUnsteadyProblem <: RBProblem end
abstract type PoissonSteady <: RBSteadyProblem end
abstract type PoissonUnsteady <: RBUnsteadyProblem end
abstract type StokesUnsteady <: RBUnsteadyProblem end

Sᵘ = Matrix{Float64}[]
Sᵖ = Matrix{Float64}[]
Nₛᵘ = 0
Nₛᵖ = 0
Nₜ = 0
Nᵘ = 0
Nᵖ = 0
Φₛᵘ = Matrix{Float64}[]
Φₛᵖ = Matrix{Float64}[]
nₛᵘ = 0
nₛᵖ = 0
Φₜᵘ = Matrix{Float64}[]
Φₜᵖ = Matrix{Float64}[]
nₜᵘ = 0
nₜᵖ = 0
nᵘ = 0
nᵖ = 0

ũ = Matrix{Float64}[]
uₙ = Matrix{Float64}[]
û = Matrix{Float64}[]
p̃ = Matrix{Float64}[]
pₙ = Matrix{Float64}[]
p̂ = Matrix{Float64}[]

Mₙ = Matrix{Float64}[]
Aₙ = Matrix{Float64}[]
AΦᵀPᵤ⁻¹ = Matrix{Float64}[]
MΦᵀPᵤ⁻¹ = Matrix{Float64}[]
MΦ = Matrix{Float64}[]
MAₙ = Matrix{Float64}[]
Bₙ = Matrix{Float64}[]
Bᵀₙ = Matrix{Float64}[]
Cₙ = Matrix{Float64}[]
Fₙ = Matrix{Float64}[]
Hₙ = Matrix{Float64}[]
Gₙ = Matrix{Float64}[]
LHSₙ = Matrix{Float64}[]
RHSₙ = Matrix{Float64}[]
Xᵘ = sparse([], [], [])
Xᵖ = sparse([], [], [])
Xᵘ₀ = sparse([], [], [])
Xᵖ₀ = sparse([], [], [])
Pᵤ⁻¹ = sparse([], [], [])

Qᵃ = 0
Qᵐ = 0
Qᶜ = 0
Qᶠ = 0
Qʰ = 0
θᵃ = Float64[]
θᵐ = Float64[]
θᶜ = Float64[]
θᶠ = Float64[]
θʰ = Float64[]

MDEIM_mat_A = Matrix{Float64}[]
MDEIMᵢ_A = Matrix{Float64}[]
MDEIM_idx_A = Vector{Int64}[]
sparse_el_A = Vector{Int64}[]
row_idx_A = Vector{Int64}[]
MDEIM_mat_M = Matrix{Float64}[]
MDEIMᵢ_M = Matrix{Float64}[]
MDEIM_idx_M = Vector{Int64}[]
sparse_el_M = Vector{Int64}[]
row_idx_M = Vector{Int64}[]
MDEIMᵢ_C = Matrix{Float64}[]
MDEIM_idx_C = Vector{Int64}[]
sparse_el_C = Vector{Int64}[]
DEIM_mat_F = Matrix{Float64}[]
DEIMᵢ_F = Matrix{Float64}[]
DEIM_idx_F = Vector{Int64}[]
DEIM_mat_H = Matrix{Float64}[]
DEIMᵢ_H = Matrix{Float64}[]
DEIM_idx_H = Vector{Int64}[]

offline_time = 0.0
online_time = 0.0
in_adaptive_loop = false

mutable struct PoissonSGRB <: PoissonSteady
  Sᵘ::Array; Φₛᵘ::Array; ũ::Array; uₙ::Array; û::Array; Aₙ::Array; Fₙ::Array;
  Hₙ::Array; Xᵘ₀::SparseMatrixCSC; LHSₙ::Array; RHSₙ::Array; MDEIM_mat_A::Array;
  MDEIMᵢ_A::Array; MDEIM_idx_A::Array; row_idx_A::Array; sparse_el_A::Array;
  DEIM_mat_F::Array; DEIMᵢ_F::Array; DEIM_idx_F::Array; DEIM_mat_H::Array;
  DEIMᵢ_H::Array; DEIM_idx_H::Array; Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64;
  Qᶠ::Int64; Qʰ::Int64; θᵃ::Array; θᶠ::Array; θʰ::Array; offline_time::Float64;
  online_time::Float64
end

mutable struct PoissonSPGRB <: PoissonSteady
  Sᵘ::Array; Φₛᵘ::Array; ũ::Array; uₙ::Array; û::Array; Aₙ::Array; Fₙ::Array;
  Hₙ::Array; Xᵘ₀::SparseMatrixCSC; LHSₙ::Array; RHSₙ::Array; MDEIM_mat_A::Array;
  MDEIMᵢ_A::Array; MDEIM_idx_A::Array; row_idx_A::Array; sparse_el_A::Array;
  DEIM_mat_F::Array; DEIMᵢ_F::Array; DEIM_idx_F::Array; DEIM_mat_H::Array;
  DEIMᵢ_H::Array; DEIM_idx_H::Array; Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64;
  Qᶠ::Int64; Qʰ::Int64; θᵃ::Array; θᶠ::Array; θʰ::Array; offline_time::Float64;
  online_time::Float64;Pᵤ⁻¹::SparseMatrixCSC; AΦᵀPᵤ⁻¹::Array
end

function setup_PoissonSGRB(::NTuple{1,Int})::PoissonSGRB

  return PoissonSGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIM_mat_A,
  MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F, DEIMᵢ_F, DEIM_idx_F,
  DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ,
  offline_time,online_time)

end

function setup_PoissonSPGRB(::NTuple{2,Int})::PoissonSPGRB

  return PoissonSPGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIM_mat_A,
  MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F, DEIMᵢ_F, DEIM_idx_F,
  DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time,
  online_time, Pᵤ⁻¹, AΦᵀPᵤ⁻¹)

end

mutable struct PoissonSTGRB <: PoissonUnsteady
  S::PoissonSGRB; Φₜᵘ::Array; Mₙ::Array; MDEIM_mat_M::Array; MDEIMᵢ_M::Array;
  MDEIM_idx_M::Array; row_idx_M::Array; sparse_el_M::Array;
  Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64; nᵘ::Int64; Qᵐ::Int64; θᵐ::Array; in_adaptive_loop::Bool
end

mutable struct PoissonSTPGRB <: PoissonUnsteady
  S::PoissonSPGRB; Φₜᵘ::Array; Mₙ::Array; MDEIM_mat_M::Array; MDEIMᵢ_M::Array;
  MDEIM_idx_M::Array; row_idx_M::Array; sparse_el_M::Array;
  MAₙ::Array; MΦ::Array; MΦᵀPᵤ⁻¹::Array; Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64;
  nᵘ::Int64; Qᵐ::Int64; θᵐ::Array; in_adaptive_loop::Bool
end

function setup_PoissonSTGRB(::NTuple{3,Int})::PoissonSTGRB

  return PoissonSTGRB(PoissonSGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ,
  MDEIM_mat_A, MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F, DEIMᵢ_F,
  DEIM_idx_F, DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ,
  offline_time, online_time), Φₜᵘ, Mₙ, MDEIM_mat_M, MDEIMᵢ_M, MDEIM_idx_M,
  row_idx_M, sparse_el_M, Nₜ, Nᵘ, nₜᵘ, nᵘ, Qᵐ, θᵐ, in_adaptive_loop)

end

function setup_PoissonSTPGRB(::NTuple{4,Int})::PoissonSTPGRB

  return PoissonSTPGRB(PoissonSPGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ,
  RHSₙ, MDEIM_mat_A, MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F,
  DEIMᵢ_F, DEIM_idx_F, DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ,
  θᶠ, θʰ, offline_time, online_time, Pᵤ⁻¹, AΦᵀPᵤ⁻¹), Φₜᵘ, Mₙ, MDEIM_mat_M,
  MDEIMᵢ_M, MDEIM_idx_M, row_idx_M, sparse_el_M, MAₙ, MΦ, MΦᵀPᵤ⁻¹, Nₜ, Nᵘ,
  nₜᵘ, nᵘ, Qᵐ, θᵐ, in_adaptive_loop)

end

mutable struct StokesSTGRB <: StokesUnsteady
  P::PoissonSTGRB; Sᵖ::Array; Φₛᵖ::Array; Φₜᵖ::Array; p̃::Array; pₙ::Array;
  p̂::Array; Bₙ::Array; Xᵘ::SparseMatrixCSC; Xᵖ::SparseMatrixCSC;
  Xᵖ₀::SparseMatrixCSC; Nₛᵖ::Int64; Nₛˡ::Int64; Nᵖ::Int64; nₛᵖ::Int64;
  nₜᵖ::Int64; nᵖ::Int64
end

#= mutable struct StokesSTPGRB <: StokesUnsteady
  P::PoissonSTPGRB; Sᵖ::Array; Sˡ::Array; Φₛᵖ::Array; Φₛˡ::Array; Φₜᵖ::Array;
  Φₜˡ::Array; p̃::Array; pₙ::Array; p̂::Array; λ̃ ::Array; λₙ::Array; λ̂ ::Array;
  Bₙ::Array; Bᵀₙ::Array; Lₙ::Array; Lᵀₙ::Array; Xᵘ::SparseMatrixCSC;
  Xᵖ::SparseMatrixCSC; Xᵖ₀::SparseMatrixCSC; Nₛᵖ::Int64; Nₛˡ::Int64; Nᵖ::Int64;
  Nˡ::Int64; nₛᵖ::Int64; nₛˡ::Int64; nₜᵖ::Int64; nₜˡ::Int64; nᵖ::Int64; nˡ::Int64;
end =#

function setup_StokesSTGRB(::NTuple{5,Int})::StokesSTGRB

  return StokesSTGRB(PoissonSTGRB(PoissonSGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀,
  LHSₙ, RHSₙ, MDEIM_mat_A, MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A,
  DEIM_mat_F, DEIMᵢ_F, DEIM_idx_F, DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, Nₛᵘ, nₛᵘ, Qᵃ,
  Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time, online_time), Φₜᵘ, Mₙ, MDEIM_mat_M, MDEIMᵢ_M,
  MDEIM_idx_M, row_idx_M, sparse_el_M, Nₜ, Nᵘ, nₜᵘ, nᵘ, Qᵐ, θᵐ, in_adaptive_loop), Sᵖ, Φₛᵖ, Φₜᵖ, p̃,
  pₙ, p̂, Bₙ, Xᵘ, Xᵖ, Xᵖ₀, Nₛᵖ, Nᵖ, nₛᵖ, nₛˡ, nₜᵖ, nᵖ)

end

#= function setup_StokesSTPGRB(::NTuple{6,Int})::StokesSTPGRB

  return StokesSTPGRB(PoissonSTPGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ,
  RHSₙ, MDEIMᵢ_A, MDEIM_idx_A, DEIMᵢ_F, DEIM_idx_F, DEIMᵢ_H, DEIM_idx_H,
  sparse_el_A, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time, Pᵤ⁻¹, AΦᵀPᵤ⁻¹),
  Φₜᵘ, Mₙ, MDEIMᵢ_M, MDEIM_idx_M, sparse_el_M, MAₙ, MΦ, MΦᵀPᵤ⁻¹, Nₜ, Nᵘ, nₜᵘ,
  nᵘ)

end =#

setup(NT::NTuple{1,Int}) = setup_PoissonSGRB(NT)
setup(NT::NTuple{2,Int}) = setup_PoissonSPGRB(NT)
setup(NT::NTuple{3,Int}) = setup_PoissonSTGRB(NT)
setup(NT::NTuple{4,Int}) = setup_PoissonSTPGRB(NT)
setup(NT::NTuple{5,Int}) = setup_StokesSTGRB(NT)
#setup(NT::NTuple{6,Int}) = setup_StokesSTPGRB(NT)

struct ROMInfoSteady <: SteadyInfo
  probl_nl::Dict
  case::Int
  paths::Function
  RB_method::String
  nₛ::Int64
  ϵₛ::Float64
  use_norm_X::Bool
  build_Parametric_RHS::Bool
  nₛ_MDEIM::Int64
  nₛ_DEIM::Int64
  postprocess::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end

mutable struct ROMInfoUnsteady <: UnsteadyInfo
  probl_nl::Dict
  case::Int
  paths::Function
  RB_method::String
  time_reduction_technique::String
  perform_nested_POD::Bool
  nₛ::Int64
  t₀::Float64
  T::Float64
  δt::Float64
  θ::Float64
  ϵₛ::Float64
  ϵₜ::Float64
  use_norm_X::Bool
  build_Parametric_RHS::Bool
  nₛ_MDEIM::Int64
  nₛ_DEIM::Int64
  space_time_M_DEIM::Bool
  functional_M_DEIM::Bool
  sampling_M_DEIM::Bool
  sampling_percentage::Float64
  postprocess::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end
