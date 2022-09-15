abstract type RBProblem <: Problem end
abstract type RBSteadyProblem{T} <: RBProblem end
abstract type RBUnsteadyProblem{T} <: RBProblem end

function init_SPoisson_variables(::Type{T}) where T
  Sᵘ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  ũ = Matrix{T}(undef,0,0)
  uₙ = Matrix{T}(undef,0,0)
  û = Matrix{T}(undef,0,0)
  Aₙ = Array{T}(undef,0,0,0)
  Fₙ = Matrix{T}(undef,0,0)
  Hₙ = Matrix{T}(undef,0,0)
  Lₙ = Matrix{T}(undef,0,0)
  Xᵘ₀ = sparse([], [], T[])
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]
  MDEIM_mat_A = Matrix{T}(undef,0,0)
  MDEIMᵢ_A = Matrix{T}(undef,0,0)
  MDEIM_idx_A = Vector{Int}(undef,0)
  row_idx_A = Vector{Int}(undef,0)
  sparse_el_A = Vector{Int}(undef,0)
  DEIM_mat_F = Matrix{T}(undef,0,0)
  DEIMᵢ_F = Matrix{T}(undef,0,0)
  DEIM_idx_F = Vector{Int}(undef,0)
  sparse_el_F = Vector{Int}(undef,0)
  DEIM_mat_H = Matrix{T}(undef,0,0)
  DEIMᵢ_H = Matrix{T}(undef,0,0)
  DEIM_idx_H = Vector{Int}(undef,0)
  sparse_el_H = Vector{Int}(undef,0)
  DEIM_mat_L = Matrix{T}(undef,0,0)
  DEIMᵢ_L = Matrix{T}(undef,0,0)
  DEIM_idx_L = Vector{Int}(undef,0)
  sparse_el_L = Vector{Int}(undef,0)
  Nₛᵘ = 0
  nₛᵘ = 0
  Qᵃ = 0
  Qᶠ = 0
  Qʰ = 0
  Qˡ = Vector{Int}(undef,0)
  offline_time = 0.0
  online_time = 0.0

  (Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Xᵘ₀, LHSₙ, RHSₙ, MDEIM_mat_A,
  MDEIMᵢ_A, MDEIM_idx_A, row_idx_A, sparse_el_A, DEIM_mat_F, DEIMᵢ_F, DEIM_idx_F,
  sparse_el_F, DEIM_mat_H, DEIMᵢ_H, DEIM_idx_H, sparse_el_H,
  DEIM_mat_L, DEIMᵢ_L, DEIM_idx_L, sparse_el_L, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, Qˡ,
  offline_time, online_time)

end

function init_STPoisson_variables(::Type{T}) where T
  Φₜᵘ = Matrix{T}(undef,0,0)
  Mₙ = Array{T}(undef,0,0,0)
  MDEIM_mat_M = Matrix{T}(undef,0,0)
  MDEIMᵢ_M = Matrix{T}(undef,0,0)
  MDEIM_idx_M = Vector{Int}(undef,0)
  row_idx_M = Vector{Int}(undef,0)
  sparse_el_M = Vector{Int}(undef,0)
  MDEIM_idx_time_A = Vector{Int}(undef,0)
  MDEIM_idx_time_M = Vector{Int}(undef,0)
  DEIM_idx_time_F = Vector{Int}(undef,0)
  DEIM_idx_time_H = Vector{Int}(undef,0)
  DEIM_idx_time_L = Vector{Int}(undef,0)
  Nₜ = 0
  Nᵘ = 0
  nₜᵘ = 0
  nᵘ = 0
  Qᵐ = 0

  (Φₜᵘ,Mₙ,MDEIM_mat_M,MDEIMᵢ_M,MDEIM_idx_M,row_idx_M,sparse_el_M,
  MDEIM_idx_time_A, MDEIM_idx_time_M, DEIM_idx_time_F, DEIM_idx_time_H, DEIM_idx_time_L, Nₜ,Nᵘ,nₜᵘ,nᵘ,Qᵐ)

end

function init_SADR_variables(::Type{T}) where T
  Bₙ = Array{T}(undef,0,0,0)
  Dₙ = Array{T}(undef,0,0,0)
  MDEIMᵢ_B = Matrix{T}(undef,0,0)
  MDEIMᵢ_D = Matrix{T}(undef,0,0)
  MDEIM_idx_B = Vector{Int}(undef,0)
  row_idx_B = Vector{Int}(undef,0)
  sparse_el_B = Vector{Int}(undef,0)
  MDEIM_idx_D = Vector{Int}(undef,0)
  row_idx_D = Vector{Int}(undef,0)
  sparse_el_D = Vector{Int}(undef,0)

  (Bₙ,Dₙ,MDEIMᵢ_B,MDEIM_idx_B,row_idx_B,sparse_el_B,
    MDEIMᵢ_D,MDEIM_idx_D,row_idx_D,sparse_el_D)

end

function init_STADR_variables(::Type{T}) where T
  MDEIM_idx_time_B = Vector{Int}(undef,0)
  MDEIM_idx_time_D = Vector{Int}(undef,0)

  MDEIM_idx_time_B, MDEIM_idx_time_D

end

function init_StokesS_variables(::Type{T}) where T
  Sᵖ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  p̃ = Matrix{T}(undef,0,0)
  pₙ = Matrix{T}(undef,0,0)
  p̂ = Matrix{T}(undef,0,0)
  Bₙ = Array{T}(undef,0,0,0)
  LCₙ = Matrix{T}(undef,0,0)
  MDEIMᵢ_B = Matrix{T}(undef,0,0)
  MDEIM_idx_B = Vector{Int}(undef,0)
  row_idx_B = Vector{Int}(undef,0)
  sparse_el_B = Vector{Int}(undef,0)
  Xᵘ = sparse([], [], T[])
  Xᵖ₀ = sparse([], [], T[])
  Nₛᵖ = 0
  nₛᵖ = 0
  Qᵇ = 0

  Sᵖ,Φₛᵘ,p̃,pₙ,p̂,Bₙ,MDEIMᵢ_B,MDEIM_idx_B,row_idx_B,sparse_el_B,Xᵘ,Xᵖ₀,Nₛᵖ,nₛᵖ,Qᵇ

end

function init_StokesST_variables(::Type{T}) where T
  Φₜᵖ = Matrix{T}(undef,0,0)
  MDEIM_idx_time_B = Vector{Int}(undef,0)
  Nᵖ = 0
  nₜᵖ = 0
  nᵖ = 0

  Φₜᵖ,MDEIM_idx_time_B,Nᵖ,nₜᵖ,nᵖ

end

function init_NavierStokesSteady_variables(::Type{T}) where T
  Cₙ = Array{T}(undef,0,0,0)
  MDEIMᵢ_C = Matrix{T}(undef,0,0)
  MDEIM_idx_C = Vector{Int}(undef,0)
  row_idx_C = Vector{Int}(undef,0)

  Cₙ, MDEIMᵢ_C, MDEIM_idx_C, row_idx_C

end

function init_NavierStokesUnsteady_variables(::Type{T}) where T
  MDEIM_idx_time_C = Vector{Int}(undef,0)

  MDEIM_idx_time_C

end

mutable struct PoissonSteady{T} <: RBSteadyProblem{T}
  Sᵘ::Matrix{T}; Φₛᵘ::Matrix{T}; ũ::Matrix{T}; uₙ::Matrix{T}; û::Matrix{T}; Aₙ::Array{T}; Fₙ::Matrix{T};
  Hₙ::Matrix{T}; Lₙ::Vector{Matrix{T}}; Xᵘ₀::SparseMatrixCSC{T}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}}; MDEIM_mat_A::Matrix{T};
  MDEIMᵢ_A::Matrix{T}; MDEIM_idx_A::Vector{Int}; row_idx_A::Vector{Int}; sparse_el_A::Vector{Int};
  DEIM_mat_F::Matrix{T}; DEIMᵢ_F::Matrix{T}; DEIM_idx_F::Vector{Int}; sparse_el_F::Vector{Int};
  DEIM_mat_H::Matrix{T}; DEIMᵢ_H::Matrix{T}; DEIM_idx_H::Vector{Int}; sparse_el_H::Vector{Int};
  DEIM_mat_L::Matrix{T}; DEIMᵢ_L::Matrix{T}; DEIM_idx_L::Vector{Int}; sparse_el_L::Vector{Int};
  Nₛᵘ::Int; nₛᵘ::Int; Qᵃ::Int; Qᶠ::Int; Qʰ::Int; Qˡ::Vector{Int}; offline_time::Float;
  online_time::Float
end

mutable struct PoissonUnsteady{T} <: RBUnsteadyProblem{T}
  Steady::PoissonSteady{T}; Φₜᵘ::Matrix{T}; Mₙ::Array{T}; MDEIM_mat_M::Matrix{T}; MDEIMᵢ_M::Matrix{T};
  MDEIM_idx_M::Vector{Int}; row_idx_M::Vector{Int}; sparse_el_M::Vector{Int};
  MDEIM_idx_time_A::Vector{Int}; MDEIM_idx_time_M::Vector{Int};
  DEIM_idx_time_F::Vector{Int}; DEIM_idx_time_H::Vector{Int};
  Nₜ::Int; Nᵘ::Int; nₜᵘ::Int; nᵘ::Int; Qᵐ::Int;
end

mutable struct ADRSteady{T} <: RBSteadyProblem{T}
  Poisson::PoissonSteady{T}; Bₙ::Array{T}; Dₙ::Array{T};
  MDEIMᵢ_B::Matrix{T}; MDEIM_idx_B::Vector{Int}; row_idx_B::Vector{Int}; sparse_el_B::Vector{Int};
  MDEIMᵢ_D::Matrix{T}; MDEIM_idx_D::Vector{Int}; row_idx_D::Vector{Int}; sparse_el_D::Vector{Int};
end

mutable struct ADRUnsteady{T} <: RBUnsteadyProblem{T}
  Poisson::PoissonUnsteady{T}; Steady::ADRSteady{T}; MDEIM_idx_time_B::Vector{Int};
  MDEIM_idx_time_D::Vector{Int};
end

mutable struct StokesSteady{T} <: RBSteadyProblem{T}
  Poisson::PoissonSteady{T}; Sᵖ::Matrix{T}; Φₛᵖ::Matrix{T}; p̃::Matrix{T}; pₙ::Matrix{T};
  p̂::Matrix{T}; Bₙ::Array{T}; MDEIMᵢ_B::Matrix{T}; MDEIM_idx_B::Vector{Int}; row_idx_B::Vector{Int}; sparse_el_B::Vector{Int};
  Xᵘ::SparseMatrixCSC{T}; Xᵖ₀::SparseMatrixCSC{T}; Nₛᵖ::Int; nₛᵖ::Int; Qᵇ::Int
end

mutable struct StokesUnsteady{T} <: RBUnsteadyProblem{T}
  Poisson::PoissonUnsteady{T}; Steady::StokesSteady{T}; MDEIM_idx_time_B::Vector{Int};
  Φₜᵖ::Matrix{T}; Nᵖ::Int; nₜᵖ::Int; nᵖ::Int
end

mutable struct NavierStokesSteady{T} <: RBSteadyProblem{T}
  Stokes::StokesSteady{T}; Sᵘ_quad::Matrix{T}; Cₙ::Array{T}; MDEIMᵢ_C::Matrix{T};
  MDEIM_idx_C::Vector{Int}; row_idx_C::Vector{Int}; sparse_el_C::Vector{Int};
end

mutable struct NavierStokesUnsteady{T} <: RBUnsteadyProblem{T}
  Stokes::StokesUnsteady{T}; Steady::NavierStokesSteady{T};
  MDEIM_idx_time_C::Vector{Int};
end

function setup(::NTuple{1,Int}, ::Type{T}) where T

  PoissonSteady{T}(init_PoissonSteady_variables(T)...)

end

function setup(::NTuple{2,Int}, ::Type{T}) where T

  PoissonUnsteady{T}(
    setup(get_NTuple(1, Int), T), init_PoissonUnsteady_variables(T)...)

end

function setup(::NTuple{3,Int}, ::Type{T}) where T

  ADRSteady{T}(
    setup(get_NTuple(1, Int), T), init_ADRSteady_variables(T)...)

end

function setup(::NTuple{4,Int}, ::Type{T}) where T

  ADRUnsteady{T}(
    setup(get_NTuple(2, Int), T), setup(get_NTuple(3, Int), T),
    init_ADRUnsteady_variables(T)...)

end

function setup(::NTuple{5,Int}, ::Type{T}) where T

  StokesSteady{T}(
    setup(get_NTuple(1, Int), T), init_StokesSteady_variables(T)...)

end

function setup(::NTuple{6,Int}, ::Type{T}) where T

  StokesUnsteady{T}(
    setup(get_NTuple(2, Int), T), setup(get_NTuple(5, Int), T),
    init_StokesUnsteady_variables(T)...)

end

function setup(::NTuple{7,Int}, ::Type{T}) where T

  NavierStokesSteady{T}(
    setup(get_NTuple(5, Int), T), init_NavierStokesSteady_variables(T)...)

end

function setup(::NTuple{8,Int}, ::Type{T}) where T

  NavierStokesUnsteady{T}(
    setup(get_NTuple(6, Int), T), setup(get_NTuple(7, Int), T),
    init_NavierStokesUnsteady_variables(T)...)

end

function Base.getproperty(RBVars::PoissonUnsteady, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :offline_time, :online_time)
    getfield(RBVars.Steady, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::PoissonUnsteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :offline_time, :online_time)
    setfield!(RBVars.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::ADRSteady, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :offline_time, :online_time)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::ADRSteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :offline_time, :online_time)
    setfield!(RBVars.Poisson, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::ADRUnsteady, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :offline_time, :online_time,
    :Φₜᵘ, :Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ)
    getfield(RBVars.Steady, sym)
  elseif sym in (:Bₙ, :Dₙ, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIMᵢ_D, :MDEIM_idx_D, :row_idx_D, :sparse_el_D)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::ADRUnsteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :offline_time, :online_time,
    :Φₜᵘ, :Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Bₙ, :Dₙ, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIMᵢ_D, :MDEIM_idx_D, :row_idx_D, :sparse_el_D)
    setfield!(RBVars.Poisson, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesSteady, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :offline_time, :online_time)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesSteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :DEIM_mat_L, :DEIMᵢ_L, :DEIM_idx_L,
    :sparse_el_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :offline_time, :online_time)
    setfield!(RBVars.Poisson, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesUnsteady, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :offline_time, :online_time,
    :Φₜᵘ, :Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Qᵇ)
    getfield(RBVars.Steady, sym)
  elseif sym in (:Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Bₙ, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesUnsteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A, :DEIM_mat_F,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H,
    :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :offline_time, :online_time,
    :Φₜᵘ, :Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Qᵇ)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Bₙ, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
    setfield!(RBVars.Poisson, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesSteady, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F,
    :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H, :sparse_el_H,
    :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :offline_time, :online_time,
    :Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Bₙ, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ, :Qᵇ)
    getfield(RBVars.Stokes, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::NavierStokesSteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F,
    :DEIM_mat_H, :DEIMᵢ_H, :DEIM_idx_H, :sparse_el_H,
    :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :offline_time, :online_time,
    :Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Bₙ, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ, :Qᵇ)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesUnsteady, sym::Symbol)
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
    getfield(RBVars.Steady, sym)
  elseif sym in (:Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_B, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Φₜᵖ, :Nᵖ, :nₜᵖ, :nᵖ)
    getfield(RBVars.Stokes, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::NavierStokesUnsteady, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_B, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Φₜᵖ, :Nᵖ, :nₜᵖ, :nᵖ)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

struct RBPathInfo <: Info
  FEMPaths::FEMPathInfo
  ROM_structures_path::String
  results_path::String
end

struct ROMInfoSteady{T} <: Info
  FEMInfo::SteadyInfo
  Paths::RBPathInfo
  RB_method::String
  nₛ::Int
  ϵₛ::Float
  use_norm_X::Bool
  assemble_parametric_RHS::Bool
  nₛ_MDEIM::Int
  nₛ_DEIM::Int
  post_process::Bool
  get_snapshots::Bool
  get_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end

mutable struct ROMInfoUnsteady{T} <: Info
  FEMInfo::UnsteadyInfo
  Paths::RBPathInfo
  RB_method::String
  nₛ::Int
  ϵₛ::Float
  use_norm_X::Bool
  assemble_parametric_RHS::Bool
  nₛ_MDEIM::Int
  nₛ_DEIM::Int
  nₛ_MDEIM_time::Int
  nₛ_DEIM_time::Int
  post_process::Bool
  get_snapshots::Bool
  get_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
  time_reduction_technique::String
  t₀::Float
  tₗ::Float
  δt::Float
  θ::Float
  ϵₜ::Float
  st_M_DEIM::Bool
  functional_M_DEIM::Bool
  adaptivity::Bool
end

function Base.getproperty(RBInfo::ROMInfoSteady, sym::Symbol)
  if sym in (:probl_nl,)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end
