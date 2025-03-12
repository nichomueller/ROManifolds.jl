abstract type Extension end

get_extension_dof_ids(ext::Extension) = @abstractmethod
num_extension_dofs(ext::Extension) = length(get_extension_dof_ids(ext))

function extend!(v::AbstractVector,vact::AbstractVector,vext::AbstractVector,ext::Extension)
  ext_ids = get_extension_dof_ids(ext)
  act_ids = setdiff(1:length(v),ext_ids)
  @views v[act_ids] = vact
  @views v[ext_ids] = vext
  v
end

function extend!(v::AbstractVector,ext::Extension)
  n = length(v) + num_extension_dofs(ext)
  resize!(v,n)
  return v
end

function zero_extended_free_values(f::FESpace,ext::Extension)
  v = zero_free_values(f)
  extend!(v,ext)
  return v
end

function zero_extension_values(ext::Extension)
  zeros(num_extension_dofs(ext))
end

struct ZeroExtension <: Extension
  dof_ids::AbstractVector
end

function ZeroExtension(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  f_out::UnconstrainedFESpace,
  args...)

  dof_ids = get_ag_out_dof_to_bg_dof(f_bg,f_ag,f_out)
  ZeroExtension(dof_ids)
end

get_extension_dof_ids(ext::ZeroExtension) = ext.dof_ids

function Algebra.solve!(u_ext::AbstractVector,ext::ZeroExtension)
  fill!(u_ext,zero(eltype(u_ext)))
end

struct FunctionalExtension <: Extension
  values::AbstractVector
  dof_ids::AbstractVector
end

function FunctionalExtension(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  f_out::UnconstrainedFESpace,
  g::Function)

  dof_ids = get_ag_out_dof_to_bg_dof(f_bg,f_ag,f_out)
  gh = interpolate_everywhere(g,f_bg)
  values = view(get_free_dof_values(gh),dof_ids)
  FunctionalExtension(values,dof_ids)
end

get_extension_dof_ids(ext::FunctionalExtension) = ext.dof_ids

function Algebra.solve!(u_ext::AbstractVector,ext::FunctionalExtension)
  copyto!(u_ext,ext.values)
end

struct HarmonicExtension <: Extension
  values::AbstractVector
  dof_ids::AbstractVector
  ldof_ids::AbstractVector
end

function HarmonicExtension(
  laplacian::AbstractMatrix,
  residual::AbstractVector,
  dof_ids::AbstractVector,
  ldof_ids::AbstractVector)

  fact = lu(laplacian)
  values = similar(residual)
  ldiv!(values,fact,residual)
  HarmonicExtension(values,dof_ids,ldof_ids)
end

function HarmonicExtension(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  f_out::UnconstrainedFESpace,
  laplacian::AbstractMatrix,
  residual::AbstractVector)

  dof_ids = get_ag_out_dof_to_bg_dof(f_bg,f_ag,f_out)
  ldof_ids = get_bg_dof_to_dof(f_bg,f_out,dof_ids)
  HarmonicExtension(laplacian,residual,dof_ids,ldof_ids)
end

function HarmonicExtension(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  g_out::UnconstrainedFESpace,
  f_out::UnconstrainedFESpace,
  a::Function,
  l::Function)

  laplacian = assemble_matrix(a,g_out,f_out)
  residual = assemble_vector(l,f_out)
  HarmonicExtension(f_bg,f_ag,f_out,laplacian,residual)
end

get_extension_dof_ids(ext::HarmonicExtension) = ext.dof_ids

function Algebra.solve!(u_ext::AbstractVector,ext::HarmonicExtension)
  values = view(ext.values,ext.ldof_ids)
  copyto!(u_ext,values)
end

for f in (:ZeroExtension,:FunctionalExtension,:HarmonicExtension)
  @eval begin
    function $f(
      bgmodel::CartesianDiscreteModel,
      f_ag::FESpaceWithLinearConstraints,
      f_out::UnconstrainedFESpace,
      reffe,
      args...;kwargs...)

      f_bg = FESpace(bgmodel,reffe;kwargs...)
      $f(f_bg,f_ag,f_out,args...)
    end
  end
end

struct ExtensionSolver <: NonlinearSolver
  solver::NonlinearSolver
  ext::Extension
end

function Algebra.solve(
  ext_solver::ExtensionSolver,
  op::NonlinearOperator)

  u_act = zero_initial_guess(op)
  u_ext = zero_extension_values(ext_solver.ext)
  u = vcat(u_act,u_ext)
  u,cache = solve!(u,u_act,u_ext,ext_solver,op)
  return u
end

function Algebra.solve!(
  u::AbstractVector,
  u_act::AbstractVector,
  u_ext::AbstractVector,
  ext_solver::ExtensionSolver,
  op::NonlinearOperator)

  slvrcache = solve!(u_act,ext_solver.solver,op)
  solve!(u_ext,ext_solver.ext)
  extend!(u,u_act,u_ext,ext_solver.ext)
  cache = (slvrcache,u_act,u_ext)
  u,cache
end

# utils

function get_ag_out_dof_to_bg_dof(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  f_out::UnconstrainedFESpace)

  ag_dof_to_bg_dof =  get_dof_to_bg_dof(f_bg,f_ag)
  act_out_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_out)
  ag_out_dof_to_bg_dof = setdiff(act_out_dof_to_bg_dof,agg_dof_to_bg_dof)
  return ag_out_dof_to_bg_dof
end

function get_bg_dof_to_dof(
  bg_f::SingleFieldFESpace,
  f::SingleFieldFESpace,
  dof_to_bg_dof::AbstractVector
  )

  bg_dof_to_all_dof = get_bg_dof_to_dof(bg_f,f)
  bg_dof_to_dof = similar(dof_to_bg_dof)
  for (i,bg_dof) in enumerate(dof_to_bg_dof)
    bg_dof_to_dof[i] = bg_dof_to_all_dof[bg_dof]
  end
  return bg_dof_to_dof
end

function get_bg_dof_to_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  bg_dof_to_dof = zeros(Int,num_free_dofs(bg_f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  cell_ids = get_cell_dof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = _get_bg_cell_to_act_cell(f)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if dof > 0
        bg_dof_to_dof[bg_dof] = dof
      end
    end
  end
  return bg_dof_to_dof
end

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  dof_to_bg_dof = zeros(Int,num_free_dofs(f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  cell_ids = get_cell_dof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = _get_bg_cell_to_act_cell(f)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if dof > 0
        dof_to_bg_dof[dof] = bg_dof
      end
    end
  end
  return dof_to_bg_dof
end

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::FESpaceWithLinearConstraints)
  agg_dof_to_act_dof = get_mdof_to_dof(f)
  act_dof_to_bg_dof = get_dof_to_bg_dof(bg_f,f.space)
  agg_dof_to_bg_dof = compose_index(agg_dof_to_act_dof,act_dof_to_bg_dof)
  return agg_dof_to_bg_dof
end

function get_mdof_to_dof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  mdof_to_dof = zeros(T,f.n_fmdofs)
  for mDOF in eachindex(mdof_to_dof)
    DOF = f.mDOF_to_DOF[mDOF]
    mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    if mdof > 0
      mdof_to_dof[mdof] = dof
    end
  end
  return mdof_to_dof
end

function compose_index(i1_to_i2,i2_to_i3)
  T_i3 = eltype(i2_to_i3)
  n_i2 = length(i1_to_i2)
  i1_to_i3 = zeros(T_i3,n_i2)
  for (i1,i2) in enumerate(i1_to_i2)
    i1_to_i3[i1] = i2_to_i3[i2]
  end
  return i1_to_i3
end

function get_dof_to_cells(cell_dofs::AbstractVector)
  inverse_table(Table(cell_dofs))
end

function inverse_table(cell_dofs::Table)
  ndofs = maximum(cell_dofs.data)
  ptrs = zeros(Int32,ndofs+1)
  for dof in cell_dofs.data
    ptrs[dof+1] += 1
  end
  length_to_ptrs!(ptrs)

  data = Vector{Int32}(undef,ptrs[end]-1)
  for cell in 1:length(cell_dofs)
    pini = cell_dofs.ptrs[cell]
    pend = cell_dofs.ptrs[cell+1]-1
    for p in pini:pend
      dof = cell_dofs.data[p]
      data[ptrs[dof]] = cell
      ptrs[dof] += 1
    end
  end
  rewind_ptrs!(ptrs)

  Table(data,ptrs)
end
