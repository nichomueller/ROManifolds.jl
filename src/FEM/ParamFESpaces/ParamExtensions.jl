function Extensions.HarmonicExtension(
  a::Function,
  l::Function,
  ext_space::SingleFieldFESpace)

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
