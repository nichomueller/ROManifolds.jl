include("../../../src/Utils/Utils.jl")
include("../../../src/FEM/FEM.jl")

function config_FEM()
  name = "poisson"
  issteady = true
  steadiness = issteady ? "steady" : "unsteady"
  ID = 1
  D = 3
  nₛ = 100
  order = 1
  solver = "lu"
  case = 2
  root = "/home/nicholasmueller/git_repos/Mabla.jl"

  unknowns, structures = setup_FEM(name, issteady)
  affine_structures, mesh_name, bnd_info, μfun = _setup_case(case)

  Paths = FOMPath(root, steadiness, name, mesh_name, case)
  FEMInfo = FOMInfoS{ID}(D, unknowns, structures,
    affine_structures, bnd_info, order, solver, Paths, nₛ)

  FEMInfo, μfun
end

function _setup_case(case::Int)

  if case == 0

    affine_structures = ["A", "F", "H", "L"]
    ranges = [[0.4, 0.6], [0.4, 0.6], [0.05, 0.1]]
    mesh_name = "cube15x15x15.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  elseif case == 1

    affine_structures = ["F", "H", "L"]
    ranges = [[1., 2.], [1., 2.], [1., 2.]]
    mesh_name = "cube15x15x15.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  elseif case == 2

    affine_structures = [""]
    ranges = [[1., 2.], [1., 2.], [1., 2.], [1., 2.],
              [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]
    mesh_name = "cube15x15x15.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  else

    error("Not implemented")

  end

  μfun(nₛ) = generate_parameters(ranges::Vector{Vector{Float}}, nₛ)
  affine_structures, mesh_name, bnd_info, μfun

end

function get_operator(
  FEMInfo::FOMInfoS{ID},
  model::DiscreteModel{D,D},
  μ::Vector) where {ID,D}

  α = get_fun(FEMInfo, μ, "A")
  f = get_fun(FEMInfo, μ, "F")
  h = get_fun(FEMInfo, μ, "H")
  g = get_fun(FEMInfo, μ, "L")

  FEMSpace = FOMS(FEMInfo, model, g)::FOMS{ID,D}

  a(u, v) = ∫(∇(v) ⋅ (α * ∇(u)))FEMSpace.dΩ
  rhs(v) = ∫(v * f)FEMSpace.dΩ + ∫(v * h)FEMSpace.dΓn

  AffineFEOperator(a, rhs, FEMSpace.V[1], FEMSpace.V₀[1])

end

function get_fun(FEMInfo::FOMInfoS, μ::Vector, var::String)
  if var == "A"
    _get_α(FEMInfo, μ)
  elseif var == "F"
    _get_f(FEMInfo, μ)
  elseif var == "H"
    _get_h(FEMInfo, μ)
  else
    _get_g(FEMInfo, μ)
  end
end

function _get_α(FEMInfo::FOMInfoS, μ::Vector)

  function α(x)
    if "A" ∈ FEMInfo.affine_structures
      return sum(μ)
    else
      return 1. + μ[3] + 1. / μ[3] * exp(-norm(x-Point(μ[1:3]))^2 / μ[3])
    end
  end

  α

end

function _get_f(FEMInfo::FOMInfoS, μ::Vector)

  function f(x)
    if "F" ∈ FEMInfo.affine_structures
      return 1.
    else
      return 1. + sin(norm(Point(μ[4:6]) .* x))
    end
  end

  f

end

function _get_g(FEMInfo::FOMInfoS, μ::Vector)

  function g(x)
    if "L" ∈ FEMInfo.affine_structures
      return 0.
    else
      return norm(μ[7:9]) - 1.5
    end
  end

  g

end

function _get_h(FEMInfo::FOMInfoS, μ::Vector)

  function h(x)
    if "H" ∈ FEMInfo.affine_structures
      return 1.
    else
      return 1. + cos(norm(Point(μ[7:9]) .* x))
    end
  end

  h

end

const FEMInfo, _ = config_FEM()
