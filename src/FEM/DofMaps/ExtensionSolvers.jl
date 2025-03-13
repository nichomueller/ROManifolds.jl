function extend(u_act::AbstractVector,u_ext::AbstractVector)
  vcat(u_act,u_ext)
end

function extend(u_act::BlockVector,u_ext::BlockVector)
  block = map(extend,blocks(u_act),blocks(u_ext))
  mortar(block)
end

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

function Algebra.solve!(vext::AbstractVector,ext::ZeroExtension)
  fill!(vext,zero(eltype(vext)))
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

function Algebra.solve!(vext::AbstractVector,ext::FunctionalExtension)
  copyto!(vext,ext.values)
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
  g_out::SingleFieldFESpace,
  f_out::SingleFieldFESpace,
  a::Function,
  l::Function)

  laplacian = assemble_matrix(a,g_out,f_out)
  residual = assemble_vector(l,f_out)
  HarmonicExtension(f_bg,f_ag,f_out,laplacian,residual)
end

get_extension_dof_ids(ext::HarmonicExtension) = ext.dof_ids

function Algebra.solve!(vext::AbstractVector,ext::HarmonicExtension)
  values = view(ext.values,ext.ldof_ids)
  copyto!(vext,values)
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

# multi field interface; no constructors are provided for this type so far

struct BlockExtension{A<:AbstractVector{<:Extension}} <: Extension
  blocks::A
end

BlockArrays.blocks(ext::BlockExtension) = ext.blocks

get_extension_dof_ids(ext::BlockExtension) = mortar(map(get_extension_dof_ids,blocks(ext)))

function extend!(v::BlockVector,vact::BlockVector,vext::BlockVector,ext::BlockExtension)
  map(extend!,blocks(vact),blocks(vext),blocks(ext))
end

function extend!(v::AbstractVector,ext::BlockExtension)
  map(extend!,blocks(v),blocks(ext))
end

function Algebra.solve!(vext::BlockVector,ext::BlockExtension)
  map(solve!,blocks(vext),blocks(ext))
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
  u = extend(u_act,u_ext)
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
