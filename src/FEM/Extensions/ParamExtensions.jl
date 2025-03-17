struct ParamExtension{E<:ExtensionStyle} <: ExtensionStyle
  style::E
end

ParamExtension() = ParamExtension(HarmonicExtension())

struct UnEvalExtension{E} <: Extension{ParamExtension{E}}
  style::ParamExtension{E}
  space::SingleFieldFESpace
  cell_to_bg_cells::AbstractVector
  dof_to_bg_dofs::AbstractVector
  metadata
end

function Extension(
  style::ParamExtension,
  space::SingleFieldFESpace,
  cell_to_bg_cells::AbstractVector,
  dof_to_bg_dofs::AbstractVector,
  args...)

  metadata = args
  UnEvalExtension(style,space,cell_to_bg_cells,dof_to_bg_dofs,metadata)
end

function Arrays.evaluate(ext::UnEvalExtension{ZeroExtension},args...)
  space = ext.space(args...)
  Extension(ZeroExtension(),space,ext.cell_to_bg_cells,ext.dof_to_bg_dofs)
end

function Arrays.evaluate(ext::UnEvalExtension{FunctionExtension},args...)
  f, = ext.metadata
  space = ext.space(args...)
  fμ = f(args...)
  Extension(FunctionExtension(),space,ext.cell_to_bg_cells,ext.dof_to_bg_dofs,fμ)
end

function Arrays.evaluate(ext::UnEvalExtension{HarmonicExtension},args...)
  a,l, = ext.metdata
  space = ext.space(args...)
  aμ = a
  lμ(v) = l(args...,v)

  laplacian = assemble_matrix(aμ,space,space)
  vector = assemble_vector(lμ,space)
  factor = lu(testitem(laplacian))
  values = similar(vector)
  for i in param_eachindex(laplacian)
    x = param_getindex(values,i)
    b = param_getindex(vector,i)
    ldiv!(x,factor,b)
    if i < param_length(laplacian)
      A = param_getindex(laplacian,i+1)
      lu!(factor,A)
    end
  end

  vh = FEFunction(space,values)
  Extension(HarmonicExtension(),laplacian,vector,vh,cell_to_bg_cells,dof_to_bg_dofs)
end

(ext::ParamExtension)(::Nothing) = ext
(ext::ParamExtension)(μ) = evaluate(ext,μ)
(ext::ParamExtension)(μ,t) = evaluate(ext,μ,t)
