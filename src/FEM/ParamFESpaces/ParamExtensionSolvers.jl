function DofMaps.extension(u_act::BlockParamVector,u_ext::BlockParamVector)
  block = map(extension,blocks(u_act),blocks(u_ext))
  mortar(block)
end

function DofMaps.extension!(
  v::ConsecutiveParamVector,
  vact::ConsecutiveParamVector,
  vext::ConsecutiveParamVector,
  ext::Extension)

  ext_ids = DofMaps.get_extension_dof_ids(ext)
  act_ids = setdiff(1:innerlength(v),ext_ids)
  data = get_all_data(v)
  data_act = get_all_data(vact)
  data_ext = get_all_data(vext)
  for k in axes(data,2)
    for (iaid,aid) in enumerate(act_ids)
      data[aid,k] = data_act[iaid,k]
    end
    for (ieid,eid) in enumerate(ext_ids)
      data[eid,k] = data_ext[ieid,k]
    end
  end
  v
end

function zero_extension_param_values(ext::Extension,plength::Int)
  v = zero_extension_values(ext)
  global_parameterize(v,plength)
end

function DofMaps.HarmonicExtension(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  g_out::SingleFieldFESpace,
  f_out::SingleFieldFESpace,
  a::Function,
  l::Function,
  μ::AbstractRealization)

  # the laplacian should not be parameterized; the residual, on the other hand, is parameterized
  assem = SparseMatrixAssembler(g_out,f_out)
  passem = parameterize(assem,μ)
  du = get_trial_fe_basis(g_out)
  v = get_fe_basis(f_out)
  matdata = collect_cell_matrix(g_out,f_out,a(du,v))
  vecdata = collect_cell_vector(f_out,l(μ,v))
  laplacian = assemble_matrix(assem,matdata)
  residual = assemble_vector(passem,vecdata)
  HarmonicExtension(f_bg,f_ag,f_out,laplacian,residual)
end

function Algebra.solve!(vext::ConsecutiveParamVector,ext::HarmonicExtension)
  v = get_all_data(vext)
  w = get_all_data(ext.values)
  for k in axes(v,2)
    for (ildof,ldof) in enumerate(ext.ldof_ids)
      v[ildof,k] = w[ldof,k]
    end
  end
  v
end

function DofMaps.extension!(
  v::BlockParamVector,
  vact::BlockParamVector,
  vext::BlockParamVector,
  ext::BlockExtension)

  map(extension!,blocks(vact),blocks(vext),blocks(ext))
end

function DofMaps.extension!(v::BlockParamVector,ext::BlockExtension)
  map(extension!,blocks(v),blocks(ext))
end

function Algebra.solve!(vext::BlockParamVector,ext::BlockExtension)
  map(solve!,blocks(vext),blocks(ext))
end

function Algebra.solve(
  ext_solver::ExtensionSolver,
  op::NonlinearParamOperator)

  u_act = zero_initial_guess(op)
  u_ext = zero_extension_param_values(ext_solver.ext,param_length(u_act))
  u = extension(u_act,u_ext)
  u,cache = solve!(u,u_act,u_ext,ext_solver,op)
  return u
end
