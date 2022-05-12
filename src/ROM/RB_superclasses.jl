include("../FEM/FEM_superclasses.jl")
abstract type RBProblem <: Problem end
abstract type RBProblemSteady <: RBProblem end
abstract type SGRB <: RBProblemSteady end
abstract type SPGRB <: RBProblemSteady end
abstract type RBProblemUnsteady <: RBProblem end
abstract type STGRB <: RBProblemUnsteady end
abstract type STPGRB <: RBProblemUnsteady end

Sᵘ = Array{Float64}(undef, 0, 0)
Sᵖ = Array{Float64}(undef, 0, 0)
Nₛᵘ = 0
Nₛᵖ = 0
Nₜ = 0
Nᵘ = 0
Φₛᵘ = Array{Float64}(undef, 0, 0)
Φₛᵖ = Array{Float64}(undef, 0, 0)
nₛᵘ = 0
nₛᵖ = 0
Φₜᵘ = Array{Float64}(undef, 0, 0)
Φₜᵖ = Array{Float64}(undef, 0, 0)
nₜᵘ = 0
nₜᵖ = 0
nᵘ = 0
nᵖ = 0

ũ = Float64[]
uₙ = Float64[]
û = Float64[]
p̃ = Float64[]
pₙ = Float64[]
p̂ = Float64[]

Mₙ = Matrix{Float64}[]
Aₙ = Matrix{Float64}[]
AΦᵀPᵤ⁻¹ = Matrix{Float64}[]
MΦᵀPᵤ⁻¹ = Matrix{Float64}[]
MΦ = Matrix{Float64}[]
MAₙ = Matrix{Float64}[]
Bₙ = Matrix{Float64}[]
Bₙ_affine = Matrix{Float64}[]
Bₙ_idx = Float64[]
Cₙ = Matrix{Float64}[]
Cₙ_affine = Matrix{Float64}[]
Cₙ_idx = Float64[]
Fₙ = Matrix{Float64}[]
Hₙ = Matrix{Float64}[]
LHSₙ = Matrix{Float64}[]
RHSₙ = Matrix{Float64}[]
Xᵘ = sparse([], [], [])
Xᵖ = sparse([], [], [])
Xᵘ₀ = sparse([], [], [])
Xᵖ₀ = sparse([], [], [])
Pᵤ⁻¹ = sparse([], [], [])

Qᵃ = 0
Qᵐ = 0
Qᶠ = 0
Qʰ = 0
θᵃ = Float64[]
θᵐ = Float64[]
θᶠ = Float64[]
θʰ = Float64[]
MDEIMᵢ_A = Matrix{Float64}[]
MDEIM_idx_A = Float64[]
sparse_el_A = Float64[]
MDEIMᵢ_M = Matrix{Float64}[]
MDEIM_idx_M = Float64[]
sparse_el_M = Float64[]
DEIMᵢ_mat_F = Float64[]
DEIM_idx_F = Float64[]
DEIMᵢ_mat_H = Float64[]
DEIM_idx_H = Float64[]

offline_time = 0.0

mutable struct PoissonSGRB <: SGRB
  Sᵘ::Array; Φₛᵘ::Array; ũ::Array; uₙ::Array; û::Array; Aₙ::Array; Fₙ::Array;
  Hₙ::Array; Xᵘ₀::SparseMatrixCSC; LHSₙ::Array; RHSₙ::Array; MDEIMᵢ_A::Array;
  MDEIM_idx_A::Array; DEIMᵢ_mat_F::Array; DEIM_idx_F::Array; DEIMᵢ_mat_H::Array;
  DEIM_idx_H::Array; sparse_el_A::Array; Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64;
  Qᶠ::Int64; Qʰ::Int64; θᵃ::Array; θᶠ::Array; θʰ::Array; offline_time::Float64
end

mutable struct PoissonSPGRB <: SPGRB
  Sᵘ::Array; Φₛᵘ::Array; ũ::Array; uₙ::Array; û::Array; Aₙ::Array; Fₙ::Array;
  Hₙ::Array; Xᵘ₀::SparseMatrixCSC; LHSₙ::Array; RHSₙ::Array; MDEIMᵢ_A::Array;
  MDEIM_idx_A::Array; DEIMᵢ_mat_F::Array; DEIM_idx_F::Array; DEIMᵢ_mat_H::Array;
  DEIM_idx_H::Array; sparse_el_A::Array; Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64;
  Qᶠ::Int64; Qʰ::Int64; θᵃ::Array; θᶠ::Array; θʰ::Array; offline_time::Float64;
  Pᵤ⁻¹::SparseMatrixCSC; AΦᵀPᵤ⁻¹::Array
end

function setup_PoissonSGRB(::NTuple{1,Int})::PoissonSGRB

  return PoissonSGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIMᵢ_A,
  MDEIM_idx_A, DEIMᵢ_mat_F, DEIM_idx_F, DEIMᵢ_mat_H, DEIM_idx_H, sparse_el_A,
  Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time)

end

function setup_PoissonSPGRB(::NTuple{2,Int})::PoissonSPGRB

  return PoissonSPGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIMᵢ_A,
  MDEIM_idx_A, DEIMᵢ_mat_F, DEIM_idx_F, DEIMᵢ_mat_H, DEIM_idx_H, sparse_el_A,
  Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time, Pᵤ⁻¹, AΦᵀPᵤ⁻¹)

end

mutable struct PoissonSTGRB <: STGRB
  steady_info::PoissonSGRB; Φₜᵘ::Array; Mₙ::Array; MDEIMᵢ_M::Array;
  MDEIM_idx_M::Array; sparse_el_M::Array; Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64;
  nᵘ::Int64; Qᵐ::Int64; θᵐ::Array
end

mutable struct PoissonSTPGRB <: STPGRB
  steady_info::PoissonSPGRB; Φₜᵘ::Array; Mₙ::Array; MDEIMᵢ_M::Array;
  MDEIM_idx_M::Array; sparse_el_M::Array; MAₙ::Array; MΦ::Array; MΦᵀPᵤ⁻¹::Array;
  Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64; nᵘ::Int64; Qᵐ::Int64; θᵐ::Array
end

function setup_PoissonSTGRB(::NTuple{3,Int})::PoissonSTGRB

  return PoissonSTGRB(PoissonSGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ,
  MDEIMᵢ_A, MDEIM_idx_A, DEIMᵢ_mat_F, DEIM_idx_F, DEIMᵢ_mat_H, DEIM_idx_H,
  sparse_el_A, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time), Φₜᵘ, Mₙ,
  MDEIMᵢ_M, MDEIM_idx_M, sparse_el_M, Nₜ, Nᵘ, nₜᵘ, nᵘ, Qᵐ, θᵐ)

end

function setup_PoissonSTPGRB(::NTuple{4,Int})::PoissonSTPGRB

  return PoissonSTPGRB(PoissonSPGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ,
  RHSₙ, MDEIMᵢ_A, MDEIM_idx_A, DEIMᵢ_mat_F, DEIM_idx_F, DEIMᵢ_mat_H, DEIM_idx_H,
  sparse_el_A, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time, Pᵤ⁻¹, AΦᵀPᵤ⁻¹),
  Φₜᵘ, Mₙ, MDEIMᵢ_M, MDEIM_idx_M, sparse_el_M, MAₙ, MΦ, MΦᵀPᵤ⁻¹, Nₜ, Nᵘ, nₜᵘ,
  nᵘ, Qᵐ, θᵐ)

end

setup(NT::NTuple{1,Int}) = setup_PoissonSGRB(NT)
setup(NT::NTuple{2,Int}) = setup_PoissonSPGRB(NT)
setup(NT::NTuple{3,Int}) = setup_PoissonSTGRB(NT)
setup(NT::NTuple{4,Int}) = setup_PoissonSTPGRB(NT)

struct ROMSpecificsSteady <: SteadyProblem
  probl_nl::Dict
  paths::Function
  RB_method::String
  nₛ::Int64
  ϵₛ::Float64
  use_norm_X::Bool
  build_parametric_RHS::Bool
  nₛ_MDEIM::Int64
  nₛ_DEIM::Int64
  postprocess::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end

struct ROMSpecificsUnsteady <: UnsteadyProblem
  probl_nl::Dict
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
  build_parametric_RHS::Bool
  nₛ_MDEIM::Int64
  nₛ_DEIM::Int64
  space_time_M_DEIM::Bool
  postprocess::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end
