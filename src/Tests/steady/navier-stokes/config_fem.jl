include("../../../src/FEM/FEM.jl")

function config_FEM()
  name = "navier-stokes"
  issteady = true
  steadiness = issteady ? "steady" : "unsteady"
  ID = 3
  D = 3
  nₛ = 100
  order = 2
  solver = "lu"
  case = 3
  root = "/home/nicholasmueller/git_repos/Mabla.jl"

  unknowns, structures = setup_FEM(name, issteady)
  affine_structures, mesh_name, bnd_info, μfun = _setup_case(case)

  Paths = FOMPath(root, steadiness, name, mesh_name, case)
  FEMInfo = FOMInfoS{ID}(D, unknowns, structures, affine_structures, bnd_info,
    order, solver, Paths, nₛ)

  FEMInfo, μfun

end

function _setup_case(case::Int)

  if case == 1

    affine_structures = ["B"]
    ranges = [[1., 10.], [1., 10.], [1., 10.], [1., 10.],
              [1., 10.], [1., 10.], [1., 10.], [1., 10.], [1., 10.]]
    mesh_name = "cube15x15x15.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  elseif case == 2

    affine_structures = ["B", "L"]
    ranges = [[1., 10.], [1., 10.], [1., 10.], [1., 10.],
              [1., 10.], [1., 10.], [1., 10.], [1., 10.], [1., 10.]]
    mesh_name = "cube10x10x10.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  elseif case == 3

    affine_structures = ["B"]
    ranges = [[1., 10.], [1., 10.], [1., 10.], [1., 10.],
              [1., 10.], [1., 10.], [1., 10.], [1., 10.], [1., 10.]]
    mesh_name = "cube10x10x10.json"
    bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])

  else

    error("Not implemented")

  end

  affine_structures = add_affine_lifts(affine_structures)
  μfun(nₛ) = generate_parameters(ranges::Vector{Vector{Float}}, nₛ)
  affine_structures, mesh_name, bnd_info, μfun

end

function get_operator(
  FEMInfo::FOMInfoS{ID},
  model::DiscreteModel{D,D},
  μ::Vector) where {ID,D}

  α = get_fun(FEMInfo, μ, "A")
  b = get_fun(FEMInfo, μ, "B")
  f = get_fun(FEMInfo, μ, "F")
  h = get_fun(FEMInfo, μ, "H")
  g = get_fun(FEMInfo, μ, "L")

  FEMSpace = FOMS(FEMInfo, model, g)::FOMS{ID,D}
  X₀ = MultiFieldFESpace(FEMSpace.V₀)
  X = MultiFieldFESpace(FEMSpace.V)

  a((u,p),(v,q)) = ∫(∇(v) ⊙ (α * ∇(u)) - b*((∇⋅v)*p + q*(∇⋅u)))FEMSpace.dΩ
  rhs((v,q)) = ∫(v ⋅ f)FEMSpace.dΩ + ∫(v ⋅ h)FEMSpace.dΓn

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )FEMSpace.dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )FEMSpace.dΩ

  res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) - rhs((v,q))
  jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

  FEOperator(res, jac, X, X₀)

end

function get_fun(FEMInfo::FOMInfoS, μ::Vector, var::String)
  if var == "A"
    _get_α(FEMInfo, μ)
  elseif var == "B"
    _get_b(FEMInfo, μ)
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

function _get_b(FEMInfo::FOMInfoS, μ::Vector)

  function b(x)
    if "B" ∈ FEMInfo.affine_structures
      return 1.
    else
      return sum(μ[4:6])
    end
  end

  b

end

function _get_f(FEMInfo::FOMInfoS, μ::Vector)

  function f(x)
    if "F" ∈ FEMInfo.affine_structures
      return one(VectorValue(FEMInfo.D, Float))
    else
      return 1. + Point(μ[4:6]) .* x
    end
  end

  f

end

function _get_g(FEMInfo::FOMInfoS, μ::Vector)

  function g(x)
    if "L" ∈ FEMInfo.affine_structures
      return zero(VectorValue(FEMInfo.D, Float))
    else
      return 1e-4*VectorValue(μ[7] * cos(x[2]) + μ[8] * sin(x[3]), 0., 0.) .* (x[1] == 0.)
    end
  end

  g

end

function _get_h(FEMInfo::FOMInfoS, μ::Vector)

  function h(x)
    if "H" ∈ FEMInfo.affine_structures
      return one(VectorValue(FEMInfo.D, Float))
    else
      return 1. + Point(μ[7:9]) .* x
    end
  end

  h

end

const FEMInfo, _ = config_FEM()
