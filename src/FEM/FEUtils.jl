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

function Utils.recenter(a::PTArray,ah0::FEFunction;kwargs...)
  a0 = get_free_dof_values(ah0)
  recenter(a,a0;kwargs...)
end

function Utils.recenter(a::Vector{<:PTArray},ah0::FEFunction;kwargs...)
  map(eachindex(a)) do i
    ai = a[i]
    ai0 = get_free_dof_values(ah0[i])
    recenter(ai,ai0;kwargs...)
  end
end

for f in (:get_L2_norm_matrix,:get_H1_norm_matrix)
  @eval begin
    function $f(op::PTFEOperator)
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
