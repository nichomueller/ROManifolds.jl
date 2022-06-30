abstract type RBProblem <: Problem end
abstract type RBSteadyProblem{T} <: RBProblem end
abstract type RBUnsteadyProblem{T} <: RBProblem end
abstract type PoissonSteady{T} <: RBSteadyProblem{T} end
abstract type PoissonUnsteady{T} <: RBUnsteadyProblem{T} end
abstract type StokesUnsteady{T} <: RBUnsteadyProblem{T} end

function init_PoissonSGRB_variables(::Type{T}) where T
  Sᵘ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  ũ = Matrix{T}(undef,0,0)
  uₙ = Matrix{T}(undef,0,0)
  û = Matrix{T}(undef,0,0)
  Aₙ = Array{T}(undef,0,0,0)
  Fₙ = Matrix{T}(undef,0,0)
  Hₙ = Matrix{T}(undef,0,0)
  Xᵘ₀ = sparse([], [], T[])
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]
  MDEIM_mat_A = Matrix{T}(undef,0,0)
  MDEIMᵢ_A = Matrix{T}(undef,0,0)
  MDEIM_idx_A = Vector{Int64}(undef,0)
  row_idx_A = Vector{Int64}(undef,0)
  sparse_el_A = Vector{Int64}(undef,0)
  DEIM_mat_F = Matrix{T}(undef,0,0)
  DEIMᵢ_F = Matrix{T}(undef,0,0)
  DEIM_idx_F = Vector{Int64}(undef,0)
  DEIM_mat_H = Matrix{T}(undef,0,0)
  DEIMᵢ_H = Matrix{T}(undef,0,0)
  DEIM_idx_H = Vector{Int64}(undef,0)
  θᵃ = Matrix{T}(undef,0,0)
  θᶠ = Matrix{T}(undef,0,0)
  θʰ = Matrix{T}(undef,0,0)
  Nₛᵘ = 0
  nₛᵘ = 0
  Qᵃ = 0
  Qᶠ = 0
  Qʰ = 0
  offline_time = 0.0
  online_time = 0.0

  (Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIM_mat_A,
  MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F, DEIMᵢ_F, DEIM_idx_F,
  DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, θᵃ, θᶠ, θʰ, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ,
  offline_time, online_time)
end

function init_PoissonSPGRB_variables(::Type{T}) where T
  Pᵤ⁻¹ = sparse([], [], T[])
  AΦᵀPᵤ⁻¹ = Array{T}(undef,0,0,0)
  Pᵤ⁻¹, AΦᵀPᵤ⁻¹
end

function init_PoissonSTGRB_variables(::Type{T}) where T
  Φₜᵘ = Matrix{T}(undef,0,0)
  Mₙ = Array{T}(undef,0,0,0)
  MDEIM_mat_M = Matrix{T}(undef,0,0)
  MDEIMᵢ_M = Matrix{T}(undef,0,0)
  MDEIM_idx_M = Vector{Int64}(undef,0)
  row_idx_M = Vector{Int64}(undef,0)
  sparse_el_M = Vector{Int64}(undef,0)
  θᵐ = Matrix{T}(undef,0,0)
  Nₜ = 0
  Nᵘ = 0
  nₜᵘ = 0
  nᵘ = 0
  Qᵐ = 0
  Φₜᵘ,Mₙ,MDEIM_mat_M,MDEIMᵢ_M,MDEIM_idx_M,row_idx_M,sparse_el_M,θᵐ,Nₜ,Nᵘ,nₜᵘ,nᵘ,Qᵐ
end

function init_PoissonSTPGRB_variables(::Type{T}) where T
  MAₙ = Array{T}(undef,0,0,0)
  MΦ = Array{T}(undef,0,0,0)
  MΦᵀPᵤ⁻¹ = Array{T}(undef,0,0,0)
  MAₙ,MΦ,MΦᵀPᵤ⁻¹
end

function init_StokesSTGRB_variables(::Type{T}) where T
  Sᵖ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  Φₜᵖ = Matrix{T}(undef,0,0)
  p̃ = Matrix{T}(undef,0,0)
  pₙ = Matrix{T}(undef,0,0)
  p̂ = Matrix{T}(undef,0,0)
  Bₙ = Array{T}(undef,0,0,0)
  Xᵘ = sparse([], [], T[])
  Xᵖ = sparse([], [], T[])
  Xᵖ₀ = sparse([], [], T[])
  Nₛᵖ = 0
  Nᵖ = 0
  nₛᵖ = 0
  nₜᵖ = 0
  nᵖ = 0
  Sᵖ,Φₛᵘ,Φₜᵖ,p̃,pₙ,p̂,Bₙ,Xᵘ,Xᵖ,Xᵖ₀,Nₛᵖ,Nᵖ,nₛᵖ,nₜᵖ,nᵖ
end

mutable struct PoissonSGRB{T} <: PoissonSteady{T}
  Sᵘ::Matrix{T}; Φₛᵘ::Matrix{T}; ũ::Matrix{T}; uₙ::Matrix{T}; û::Matrix{T}; Aₙ::Array{T}; Fₙ::Matrix{T};
  Hₙ::Matrix{T}; Xᵘ₀::SparseMatrixCSC{T}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}}; MDEIM_mat_A::Matrix{T};
  MDEIMᵢ_A::Matrix{T}; MDEIM_idx_A::Vector{Int64}; row_idx_A::Vector{Int64}; sparse_el_A::Vector{Int64};
  DEIM_mat_F::Matrix{T}; DEIMᵢ_F::Matrix{T}; DEIM_idx_F::Vector{Int64}; DEIM_mat_H::Matrix{T};
  DEIMᵢ_H::Matrix{T}; DEIM_idx_H::Vector{Int64}; θᵃ::Matrix{T}; θᶠ::Matrix{T}; θʰ::Matrix{T};
  Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64; Qᶠ::Int64; Qʰ::Int64; offline_time::Float64;
  online_time::Float64
end

mutable struct PoissonSPGRB{T} <: PoissonSteady{T}
  Sᵘ::Matrix{T}; Φₛᵘ::Matrix{T}; ũ::Matrix{T}; uₙ::Matrix{T}; û::Matrix{T}; Aₙ::Array{T}; Fₙ::Matrix{T};
  Hₙ::Matrix{T}; Xᵘ₀::SparseMatrixCSC{T}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}}; MDEIM_mat_A::Matrix{T};
  MDEIMᵢ_A::Matrix{T}; MDEIM_idx_A::Vector{Int64}; row_idx_A::Vector{Int64}; sparse_el_A::Vector{Int64};
  DEIM_mat_F::Matrix{T}; DEIMᵢ_F::Matrix{T}; DEIM_idx_F::Vector{Int64}; DEIM_mat_H::Matrix{T};
  DEIMᵢ_H::Matrix{T}; DEIM_idx_H::Vector{Int64}; θᵃ::Matrix{T}; θᶠ::Matrix{T}; θʰ::Matrix{T};
  Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64; Qᶠ::Int64; Qʰ::Int64; offline_time::Float64;
  online_time::Float64;Pᵤ⁻¹::SparseMatrixCSC{T}; AΦᵀPᵤ⁻¹::Array{T}
end

function setup(::NTuple{1,Int}, ::Type{T}) where T

  PoissonSGRB{T}(init_PoissonSGRB_variables(T)...)::PoissonSGRB

end

function setup(::NTuple{2,Int}, ::Type{T}) where T

  PoissonSPGRB{T}(init_PoissonSGRB_variables(T)...,
    init_PoissonSPGRB_variables(T)...)

end

mutable struct PoissonSTGRB{T} <: PoissonUnsteady{T}
  S::PoissonSGRB{T}; Φₜᵘ::Matrix{T}; Mₙ::Array{T}; MDEIM_mat_M::Matrix{T}; MDEIMᵢ_M::Matrix{T};
  MDEIM_idx_M::Vector{Int64}; row_idx_M::Vector{Int64}; sparse_el_M::Vector{Int64}; θᵐ::Matrix{T};
  Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64; nᵘ::Int64; Qᵐ::Int64;
end

mutable struct PoissonSTPGRB{T} <: PoissonUnsteady{T}
  S::PoissonSGRB{T}; Φₜᵘ::Matrix{T}; Mₙ::Array{T}; MDEIM_mat_M::Matrix{T}; MDEIMᵢ_M::Matrix{T};
  MDEIM_idx_M::Vector{Int64}; row_idx_M::Vector{Int64}; sparse_el_M::Vector{Int64}; θᵐ::Matrix{T};
  Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64; nᵘ::Int64; Qᵐ::Int64;MAₙ::Array{T};MΦ::Array{T};MΦᵀPᵤ⁻¹::Array{T}
end

function setup(::NTuple{3,Int}, ::Type{T}) where T

  PoissonSTGRB{T}(PoissonSGRB{T}(init_PoissonSGRB_variables(T)...),
    init_PoissonSTGRB_variables(T)...)

end

function setup(::NTuple{4,Int}, ::Type{T}) where T

  PoissonSTPGRB{T}(PoissonSPGRB{T}(init_PoissonSGRB_variables(T)...,
    init_PoissonSPGRB_variables(T)...), init_PoissonSTPGRB_variables(T)...)

end

mutable struct StokesSTGRB{T} <: StokesUnsteady{T}
  P::PoissonSTGRB{T}; Sᵖ::Matrix{T}; Φₛᵖ::Matrix{T}; Φₜᵖ::Matrix{T}; p̃::Matrix{T}; pₙ::Matrix{T};
  p̂::Matrix{T}; Bₙ::Array{T}; Xᵘ::SparseMatrixCSC{T}; Xᵖ::SparseMatrixCSC{T};
  Xᵖ₀::SparseMatrixCSC{T}; Nₛᵖ::Int64; Nᵖ::Int64; nₛᵖ::Int64;
  nₜᵖ::Int64; nᵖ::Int64
end

#= mutable struct StokesSTPGRB <: StokesUnsteady
  P::PoissonSTPGRB; Sᵖ::Array; Sˡ::Array; Φₛᵖ::Array; Φₛˡ::Array; Φₜᵖ::Array;
  Φₜˡ::Array; p̃::Array; pₙ::Array; p̂::Array; λ̃ ::Array; λₙ::Array; λ̂ ::Array;
  Bₙ::Array; Bᵀₙ::Array; Lₙ::Array; Lᵀₙ::Array; Xᵘ::SparseMatrixCSC;
  Xᵖ::SparseMatrixCSC; Xᵖ₀::SparseMatrixCSC; Nₛᵖ::Int64; Nₛˡ::Int64; Nᵖ::Int64;
  Nˡ::Int64; nₛᵖ::Int64; nₛˡ::Int64; nₜᵖ::Int64; nₜˡ::Int64; nᵖ::Int64; nˡ::Int64;
end =#

function setup(::NTuple{5,Int}, ::Type{T}) where T

  StokesSTGRB{T}(PoissonSTGRB{T}(PoissonSGRB{T}(init_PoissonSGRB_variables(T)...),
    init_PoissonSTGRB_variables(T)...), init_StokesSTGRB_variables(T)...)

end

#= function setup_StokesSTPGRB(::NTuple{6,Int})::StokesSTPGRB

  return StokesSTPGRB(PoissonSTPGRB(Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ,
  RHSₙ, MDEIMᵢ_A, MDEIM_idx_A, DEIMᵢ_F, DEIM_idx_F, DEIMᵢ_H, DEIM_idx_H,
  sparse_el_A, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, θᵃ, θᶠ, θʰ, offline_time, Pᵤ⁻¹, AΦᵀPᵤ⁻¹),
  Φₜᵘ, Mₙ, MDEIMᵢ_M, MDEIM_idx_M, sparse_el_M, MAₙ, MΦ, MΦᵀPᵤ⁻¹, Nₜ, Nᵘ, nₜᵘ,
  nᵘ)

end =#


struct ROMInfoSteady{T} <: Info{T}
  FEMInfo::SteadyInfo{T}
  probl_nl::Dict
  case::Int
  paths::F
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

mutable struct ROMInfoUnsteady{T} <: Info{T}
  FEMInfo::UnsteadyInfo{T}
  probl_nl::Dict
  case::Int
  paths::F
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
  time_reduction_technique::String
  perform_nested_POD::Bool
  t₀::Float64
  tₗ::Float64
  δt::Float64
  θ::Float64
  ϵₜ::Float64
  space_time_M_DEIM::Bool
  functional_M_DEIM::Bool
  sampling_M_DEIM::Bool
  sampling_percentage::Float64
  adaptivity::Bool
end
