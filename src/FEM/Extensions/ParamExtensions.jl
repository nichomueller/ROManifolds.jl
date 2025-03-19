struct ParamExtension{E<:ExtensionStyle} <: ExtensionStyle
  style::E
end

ParamExtension() = ParamExtension(HarmonicExtension())

struct UnEvalExtension{E} <: Extension{ParamExtension{E}}
  style::ParamExtension{E}
  space::SingleFieldFESpace
  fdof_to_bg_fdofs::AbstractVector
  ddof_to_bg_ddofs::AbstractVector
  metadata
end

function Extension(
  style::ParamExtension,
  space::SingleFieldFESpace,
  fdof_to_bg_fdofs::AbstractVector,
  ddof_to_bg_ddofs::AbstractVector,
  metadata...)

  UnEvalExtension(style,space,fdof_to_bg_fdofs,ddof_to_bg_ddofs,metadata)
end

function Arrays.evaluate(ext::UnEvalExtension{ZeroExtension},args...)
  spaceμ = ext.space(args...)
  Extension(ZeroExtension(),spaceμ,ext.fdof_to_bg_fdofs,ext.ddof_to_bg_ddofs)
end

function Arrays.evaluate(ext::UnEvalExtension{FunctionExtension},args...)
  f, = ext.metadata
  spaceμ = ext.space(args...)
  fμ = f(args...)
  Extension(FunctionExtension(),spaceμ,ext.fdof_to_bg_fdofs,ext.ddof_to_bg_ddofs,fμ)
end

function Arrays.evaluate(ext::UnEvalExtension{HarmonicExtension},args...)
  a,l, = ext.metadata
  spaceμ = parameterize(ext.space,args...)
  aμ = a
  lμ(v) = l(args...,v)
  Extension(HarmonicExtension(),spaceμ,ext.fdof_to_bg_fdofs,ext.ddof_to_bg_ddofs,aμ,lμ)
end

(ext::UnEvalExtension)(::Nothing) = ext
(ext::UnEvalExtension)(μ) = evaluate(ext,μ)
(ext::UnEvalExtension)(μ,t) = evaluate(ext,μ,t)

get_out_fdof_to_bg_fdofs(ext::UnEvalExtension) = ext.fdof_to_bg_fdofs
get_out_ddof_to_bg_ddofs(ext::UnEvalExtension) = ext.ddof_to_bg_ddofs
FESpaces.get_fe_space(f::UnEvalExtension) = f.space
