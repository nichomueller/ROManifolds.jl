abstract type RBProblem <: Problem end
abstract type RBSteadyProblem{T} <: RBProblem end
abstract type RBUnsteadyProblem{T} <: RBProblem end
abstract type PoissonSteady{T} <: RBSteadyProblem{T} end
abstract type PoissonUnsteady{T} <: RBUnsteadyProblem{T} end
abstract type StokesSteady{T} <: RBSteadyProblem{T} end
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
  sparse_el_F = Vector{Int64}(undef,0)
  DEIM_mat_H = Matrix{T}(undef,0,0)
  DEIMᵢ_H = Matrix{T}(undef,0,0)
  DEIM_idx_H = Vector{Int64}(undef,0)
  sparse_el_H = Vector{Int64}(undef,0)
  Nₛᵘ = 0
  nₛᵘ = 0
  Qᵃ = 0
  Qᶠ = 0
  Qʰ = 0
  offline_time = 0.0
  online_time = 0.0

  (Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIM_mat_A,
  MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F, DEIMᵢ_F, DEIM_idx_F,
  sparse_el_F, DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, sparse_el_H, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ,
  offline_time, online_time)

end

function init_PoissonSPGRB_variables(::Type{T}) where T
  Pᵤ⁻¹ = sparse([], [], T[])
  AΦᵀPᵤ⁻¹ = Array{T}(undef,0,0,0)

  (init_PoissonSGRB_variables(T)..., Pᵤ⁻¹, AΦᵀPᵤ⁻¹)

end

function init_PoissonSTGRB_variables(::Type{T}) where T
  Φₜᵘ = Matrix{T}(undef,0,0)
  Mₙ = Array{T}(undef,0,0,0)
  MDEIM_mat_M = Matrix{T}(undef,0,0)
  MDEIMᵢ_M = Matrix{T}(undef,0,0)
  MDEIM_idx_M = Vector{Int64}(undef,0)
  row_idx_M = Vector{Int64}(undef,0)
  sparse_el_M = Vector{Int64}(undef,0)
  MDEIM_idx_time_A = Vector{Int64}(undef,0)
  MDEIM_idx_time_M = Vector{Int64}(undef,0)
  DEIM_idx_time_F = Vector{Int64}(undef,0)
  DEIM_idx_time_H = Vector{Int64}(undef,0)
  Nₜ = 0
  Nᵘ = 0
  nₜᵘ = 0
  nᵘ = 0
  Qᵐ = 0

  (Φₜᵘ,Mₙ,MDEIM_mat_M,MDEIMᵢ_M,MDEIM_idx_M,row_idx_M,sparse_el_M,
  MDEIM_idx_time_A, MDEIM_idx_time_M, DEIM_idx_time_F, DEIM_idx_time_H, Nₜ,Nᵘ,nₜᵘ,nᵘ,Qᵐ)

end

function init_PoissonSTPGRB_variables(::Type{T}) where T
  MAₙ = Array{T}(undef,0,0,0)
  MΦ = Array{T}(undef,0,0,0)
  MΦᵀPᵤ⁻¹ = Array{T}(undef,0,0,0)

  (init_PoissonSTGRB_variables(T)...,MAₙ,MΦ,MΦᵀPᵤ⁻¹)

end

function init_StokesSGRB_variables(::Type{T}) where T
  Sᵖ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  p̃ = Matrix{T}(undef,0,0)
  pₙ = Matrix{T}(undef,0,0)
  p̂ = Matrix{T}(undef,0,0)
  Bₙ = Array{T}(undef,0,0,0)
  Xᵘ = sparse([], [], T[])
  Xᵖ₀ = sparse([], [], T[])
  Nₛᵖ = 0
  nₛᵖ = 0

  Sᵖ,Φₛᵘ,p̃,pₙ,p̂,Bₙ,Xᵘ,Xᵖ₀,Nₛᵖ,nₛᵖ

end

function init_StokesSTGRB_variables(::Type{T}) where T
  Φₜᵖ = Matrix{T}(undef,0,0)
  Nᵖ = 0
  nₜᵖ = 0
  nᵖ = 0

  Φₜᵖ,Nᵖ,nₜᵖ,nᵖ

end

mutable struct PoissonSGRB{T} <: PoissonSteady{T}
  Sᵘ::Matrix{T}; Φₛᵘ::Matrix{T}; ũ::Matrix{T}; uₙ::Matrix{T}; û::Matrix{T}; Aₙ::Array{T}; Fₙ::Matrix{T};
  Hₙ::Matrix{T}; Xᵘ₀::SparseMatrixCSC{T}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}}; MDEIM_mat_A::Matrix{T};
  MDEIMᵢ_A::Matrix{T}; MDEIM_idx_A::Vector{Int64}; row_idx_A::Vector{Int64}; sparse_el_A::Vector{Int64};
  DEIM_mat_F::Matrix{T}; DEIMᵢ_F::Matrix{T}; DEIM_idx_F::Vector{Int64}; sparse_el_F::Vector{Int64};
  DEIM_mat_H::Matrix{T}; DEIMᵢ_H::Matrix{T}; DEIM_idx_H::Vector{Int64}; sparse_el_H::Vector{Int64};
  Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64; Qᶠ::Int64; Qʰ::Int64; offline_time::Float64;
  online_time::Float64
end

mutable struct PoissonSPGRB{T} <: PoissonSteady{T}
  Sᵘ::Matrix{T}; Φₛᵘ::Matrix{T}; ũ::Matrix{T}; uₙ::Matrix{T}; û::Matrix{T}; Aₙ::Array{T}; Fₙ::Matrix{T};
  Hₙ::Matrix{T}; Xᵘ₀::SparseMatrixCSC{T}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}}; MDEIM_mat_A::Matrix{T};
  MDEIMᵢ_A::Matrix{T}; MDEIM_idx_A::Vector{Int64}; row_idx_A::Vector{Int64}; sparse_el_A::Vector{Int64};
  DEIM_mat_F::Matrix{T}; DEIMᵢ_F::Matrix{T}; DEIM_idx_F::Vector{Int64}; sparse_el_F::Vector{Int64};
  DEIM_mat_H::Matrix{T}; DEIMᵢ_H::Matrix{T}; DEIM_idx_H::Vector{Int64}; sparse_el_H::Vector{Int64};
  Nₛᵘ::Int64; nₛᵘ::Int64; Qᵃ::Int64; Qᶠ::Int64; Qʰ::Int64; offline_time::Float64;
  online_time::Float64;Pᵤ⁻¹::SparseMatrixCSC{T}; AΦᵀPᵤ⁻¹::Array{T}
end

mutable struct PoissonSTGRB{T} <: PoissonUnsteady{T}
  S::PoissonSGRB{T}; Φₜᵘ::Matrix{T}; Mₙ::Array{T}; MDEIM_mat_M::Matrix{T}; MDEIMᵢ_M::Matrix{T};
  MDEIM_idx_M::Vector{Int64}; row_idx_M::Vector{Int64}; sparse_el_M::Vector{Int64};
  MDEIM_idx_time_A::Vector{Int64}; MDEIM_idx_time_M::Vector{Int64};
  DEIM_idx_time_F::Vector{Int64}; DEIM_idx_time_H::Vector{Int64};
  Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64; nᵘ::Int64; Qᵐ::Int64;
end

mutable struct PoissonSTPGRB{T} <: PoissonUnsteady{T}
  S::PoissonSPGRB{T}; Φₜᵘ::Matrix{T}; Mₙ::Array{T}; MDEIM_mat_M::Matrix{T}; MDEIMᵢ_M::Matrix{T};
  MDEIM_idx_M::Vector{Int64}; row_idx_M::Vector{Int64}; sparse_el_M::Vector{Int64};
  MDEIM_idx_time_A::Vector{Int64}; MDEIM_idx_time_M::Vector{Int64};
  DEIM_idx_time_F::Vector{Int64}; DEIM_idx_time_H::Vector{Int64};
  Nₜ::Int64; Nᵘ::Int64; nₜᵘ::Int64; nᵘ::Int64; Qᵐ::Int64;MAₙ::Array{T};MΦ::Array{T};MΦᵀPᵤ⁻¹::Array{T}
end

mutable struct StokesSGRB{T} <: StokesSteady{T}
  P::PoissonSGRB{T}; Sᵖ::Matrix{T}; Φₛᵖ::Matrix{T}; p̃::Matrix{T}; pₙ::Matrix{T};
  p̂::Matrix{T}; Bₙ::Array{T}; Xᵘ::SparseMatrixCSC{T};
  Xᵖ₀::SparseMatrixCSC{T}; Nₛᵖ::Int64; nₛᵖ::Int64
end

mutable struct StokesSTGRB{T} <: StokesUnsteady{T}
  P::PoissonSTGRB{T}; S::StokesSGRB{T}; Φₜᵖ::Matrix{T}; Nᵖ::Int64;
  nₜᵖ::Int64; nᵖ::Int64
end

function setup(::NTuple{1,Int}, ::Type{T}) where T

  PoissonSGRB{T}(init_PoissonSGRB_variables(T)...)

end

function setup(::NTuple{2,Int}, ::Type{T}) where T

  PoissonSPGRB{T}(init_PoissonSPGRB_variables(T)...)

end

function setup(::NTuple{3,Int}, ::Type{T}) where T

  NT1 = get_NTuple(1, Int64)
  PoissonSTGRB{T}(setup(NT1, T), init_PoissonSTGRB_variables(T)...)

end

function setup(::NTuple{4,Int}, ::Type{T}) where T

  NT2 = get_NTuple(2, Int64)
  PoissonSTPGRB{T}(setup(NT2, T), init_PoissonSTPGRB_variables(T)...)

end

function setup(::NTuple{5,Int}, ::Type{T}) where T

  NT1 = get_NTuple(1, Int64)
  StokesSGRB{T}(setup(NT1, T), init_StokesSGRB_variables(T)...)

end

function setup(::NTuple{6,Int}, ::Type{T}) where T

  NT3 = get_NTuple(3, Int64)
  NT5 = get_NTuple(5, Int64)
  StokesSTGRB{T}(setup(NT3, T), setup(NT5, T), init_StokesSTGRB_variables(T)...)

end

struct ROMInfoSteady{T} <: Info
  FEMInfo::SteadyInfo
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
  post_process::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end

mutable struct ROMInfoUnsteady{T} <: Info
  FEMInfo::UnsteadyInfo
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
  post_process::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
  time_reduction_technique::String
  t₀::Float64
  tₗ::Float64
  δt::Float64
  θ::Float64
  ϵₜ::Float64
  st_M_DEIM::Bool
  functional_M_DEIM::Bool
  adaptivity::Bool
end
