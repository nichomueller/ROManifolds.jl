include("../../../src/FEM/FEM.jl")

function config_FEM()
  name = "poisson"
  issteady = false
  steadiness = issteady ? "steady" : "unsteady"
  ID = 1
  D = 3
  nₛ = 100
  order = 1
  solver = "lu"
  case = 2
  root = "/home/nicholasmueller/git_repos/Mabla.jl"

  θ = 0.5
  t₀ = 0.
  tₗ = 10/4
  δt = 0.05

  unknowns, structures = setup_FEM(name, issteady)
  affine_structures, mesh_name, bnd_info, μfun = _setup_case(case)

  Paths = FOMPath(root, steadiness, name, mesh_name, case)
  FEMInfo = FOMInfoST{ID}(D, unknowns, structures, affine_structures, bnd_info,
    order, solver, Paths, nₛ, θ, t₀, tₗ, δt)

  FEMInfo, μfun

end

function _setup_case(case::Int)
  if case == 1

    affine_structures = ["M"]
    ranges = [[1., 2.], [1., 2.], [1., 2.], [1., 2.],
              [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]
    mesh_name = "cube15x15x15.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  elseif case == 2

    affine_structures = ["M"]
    ranges = [[1., 2.], [1., 2.], [1., 2.], [1., 2.],
              [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]
    mesh_name = "cube5x5x5.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  else

    error("Not implemented")

  end

  #affine_structures = add_affine_lifts(affine_structures)
  μfun(nₛ) = generate_parameters(ranges::Vector{Vector{Float}}, nₛ)
  affine_structures, mesh_name, bnd_info, μfun

end

function get_operator(
  FEMInfo::FOMInfoST{ID},
  model::DiscreteModel{D,D},
  μ::Vector) where {ID,D}

  α, _, _ = get_fun(FEMInfo, μ, "A")
  η, _, _ = get_fun(FEMInfo, μ, "M")
  f, _, _ = get_fun(FEMInfo, μ, "F")
  h, _, _ = get_fun(FEMInfo, μ, "H")
  g, _, _ = get_fun(FEMInfo, μ, "L")
  x₀ = get_fun(FEMInfo, μ, "x₀")

  FEMSpace = FOMST(FEMInfo, model, g)::FOMST{ID,D}

  m(t, u, v) = ∫(η(t) * (u*v))FEMSpace.dΩ
  a(t, u, v) = ∫(∇(v) ⋅ (α(t) * ∇(u)))FEMSpace.dΩ
  rhs(t, v) = ∫(v * f(t))FEMSpace.dΩ + ∫(v * h(t))FEMSpace.dΓn

  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.V[1], FEMSpace.V₀[1])
  x₀_field = interpolate_everywhere(x₀, FEMSpace.V[1](FEMInfo.t₀))

  operator, x₀_field

end

function get_fun(FEMInfo::FOMInfoST, μ::Vector, var::String)
  if var == "A"
    _get_α(FEMInfo, μ)
  elseif var == "M"
    _get_m(FEMInfo, μ)
  elseif var == "F"
    _get_f(FEMInfo, μ)
  elseif var == "H"
    _get_h(FEMInfo, μ)
  elseif var == "x₀"
    _get_x₀(FEMInfo, μ)
  else
    _get_g(FEMInfo, μ)
  end
end

function _get_α(FEMInfo::FOMInfoST, μ::Vector)

  αₛ(x) = 1.
  αₜ(t) = 5. * sum(μ) * (2 + sin(t))
  function α(x, t::Real)
    if "A" ∈ FEMInfo.affine_structures
      return αₛ(x) * αₜ(t)
    else
      return 1. + μ[3] + 1 / μ[3] * exp(-sin(t) * norm(x-Point(μ[1:3]))^2 / μ[3])
    end
  end
  α(t::Real) = x -> α(x, t)

  α, αₛ, αₜ

end

function _get_f(FEMInfo::FOMInfoST, μ::Vector)

  fₛ(x) = 1.
  fₜ(t::Real) = sin(t)
  function f(x, t::Real)
    if "F" ∈ FEMInfo.affine_structures
      return fₛ(x) * fₜ(t)
    else
      return 1. + sin(norm(Point(μ[4:6]) .* x)*t)
    end
  end
  f(t::Real) = x -> f(x, t)

  f, fₛ, fₜ

end

function _get_h(FEMInfo::FOMInfoST, μ::Vector)

  hₛ(x) = 1.
  hₜ(t::Real) = sin(t)
  function h(x, t::Real)
    if "H" ∈ FEMInfo.affine_structures
      return hₛ(x) * hₜ(t)
    else
      return 1. + sin(norm(Point(μ[7:9]) .* x)*t)
    end
  end
  h(t::Real) = x -> h(x, t)

  h, hₛ, hₜ

end

function _get_g(FEMInfo::FOMInfoST, μ::Vector)

  gₛ(x) = 0.
  gₜ(t::Real) = 0.
  function g(x, t::Real)
    if "L" ∈ FEMInfo.affine_structures
      return gₛ(x) * gₜ(t)
    else
      return μ[7] * cos(x[2]*t) + μ[8] * sin(x[3]*t) .* (x[1] == 0.)
    end
  end
  g(t::Real) = x -> g(x, t)

  g, gₛ, gₜ

end

function _get_m(FEMInfo::FOMInfoST, μ::Vector)

  mₛ(x) = 1.
  mₜ(t::Real) = 1.
  function m(x, t)
    if "M" ∈ FEMInfo.affine_structures
      return mₛ(x) * mₜ(t)
    else
      return sum(μ)
    end
  end
  m(t::Real) = x -> m(x, t)

  m, mₛ, mₜ

end

function _get_x₀(::FOMInfoST, ::Vector)

  x₀(x) = 0.
  x₀

end

const FEMInfo, _ = config_FEM()
