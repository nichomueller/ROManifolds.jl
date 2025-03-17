function HarmonicExtension(laplacian::AbstractParamMatrix,vector::AbstractParamVector)
  @check param_length(laplacian) == param_length(vector)
  factor = lu(testitem(laplacian))
  values = similar(vector)
  for i in param_eachindex(laplacian)
    x = param_getindex(values)
    b = param_getindex(vector)
    ldiv!(x,factor,b)
    if i < param_eachindex(laplacian)
      A = param_getindex(laplacian,i+1)
      lu!(factor,A)
    end
  end
  HarmonicExtension(laplacian,vector,values)
end

function HarmonicExtension(
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function,
  μ::AbstractRealization)

  # the laplacian should not be parameterized; the residual, on the other hand, is parameterized
  assem = SparseMatrixAssembler(ext_space,ext_space)
  passem = parameterize(assem,μ)
  du = get_trial_fe_basis(ext_space)
  v = get_fe_basis(ext_space)
  matdata = collect_cell_matrix(ext_space,ext_space,a(μ,du,v))
  vecdata = collect_cell_vector(ext_space,l(μ,v))
  laplacian = assemble_matrix(passem,matdata)
  residual = assemble_vector(passem,vecdata)
  HarmonicExtension(laplacian,residual)
end
