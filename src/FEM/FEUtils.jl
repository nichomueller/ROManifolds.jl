FESpaces.get_triangulation(meas::Measure) = meas.quad.trian

function ReferenceFEs.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  cell_basis = get_data(basis)
  try # if cartesian model
    shapefun = first(cell_basis).fields
    get_order(shapefun)
  catch
    shapefuns = first(cell_basis.values).fields
    orders = get_order(shapefuns)
    first(orders)
  end
end

function Utils.recenter(a::PArray,ah0::FEFunction;kwargs...)
  a0 = get_free_dof_values(ah0)
  recenter(a,a0;kwargs...)
end

function Utils.recenter(a::Vector{<:PArray},ah0::FEFunction;kwargs...)
  map(eachindex(a)) do i
    ai = a[i]
    ai0 = get_free_dof_values(ah0[i])
    recenter(ai,ai0;kwargs...)
  end
end

for f in (:get_L2_norm_matrix,:get_H1_norm_matrix)
  @eval begin
    function $f(op::TransientPFEOperator)
      μ,t = realization(op),0.
      test = op.test
      trial = get_trial(op)
      trial_hom = allocate_trial_space(trial,μ,t)
      $f(trial_hom,test)
    end

    function $f(
      trial::TransientMultiFieldTrialFESpace,
      test::MultiFieldFESpace)

      map($f,trial.spaces,test.spaces)
    end
  end
end

function get_L2_norm_matrix(
  trial::TrialFESpace,
  test::FESpace)

  trian = get_triangulation(test)
  order = get_order(test)
  dΩ = Measure(trian,2*order)
  L2_form(u,v) = ∫(v⋅u)dΩ
  assemble_matrix(L2_form,trial,test)
end

function get_H1_norm_matrix(
  trial::TrialFESpace,
  test::FESpace)

  trian = get_triangulation(test)
  order = get_order(test)
  dΩ = Measure(trian,2*order)
  H1_form(u,v) = ∫(∇(v)⊙∇(u))dΩ + ∫(v⋅u)dΩ
  assemble_matrix(H1_form,trial,test)
end

# Interface that allows to entirely eliminate terms from the DomainContribution
for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(::Nothing,::Nothing) = nothing
    ($op)(::Any,::Nothing) = nothing
    ($op)(::Nothing,::Any) = nothing
  end
end

Base.adjoint(::Nothing) = nothing
Base.broadcasted(f,a::Nothing,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Nothing,b::CellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::CellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
LinearAlgebra.dot(::typeof(∇),::Nothing) = nothing

Fields.gradient(::Nothing) = nothing

∂ₚt(::Nothing) = nothing

CellData.integrate(::Nothing,args...) = nothing

CellData.integrate(::Any,::Nothing) = nothing

(+)(::Nothing,b::DomainContribution) = b
(+)(a::DomainContribution,::Nothing) = a
(-)(a::DomainContribution,::Nothing) = a

function (-)(::Nothing,b::DomainContribution)
  for (trian,array) in b.dict
    b.dict[trian] = lazy_map(Broadcasting(-),array)
  end
  b
end

function FESpaces.collect_cell_vector(::FESpace,::Nothing,args...)
  nothing
end

function FESpaces.collect_cell_matrix(::FESpace,::FESpace,::Nothing,args...)
  nothing
end
