struct RBParamOnlineStructure
  op::RBVariable
  assembler::Function
end

function RBParamOnlineStructure(ad::RBAffineDecomposition;kwargs...)
  op = get_op(ad)
  ad_eval = eval_affine_decomposition(ad)
  basis = get_affine_decomposition(ad)
  coeff = get_coefficient(op,basis;kwargs...)
  assembler = get_assembler(op,ad_eval)

  RBParamOnlineStructure(ad,coeff,assembler)
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition,
  coeff::Function,
  assembler::Function)

  RBParamOnlineStructure(ad,μ::Param -> assembler(coeff(μ)))
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition,
  coeff::Function,
  assembler::Function)

  RBParamOnlineStructure(ad,(μ::Param,z) -> assembler(coeff(μ,z)))
end

function get_assembler(
  ::RBSteadyVariable,
  basis::AbstractMatrix{Float},
  nr::Int,
  nc::Int)

  Qs = size(basis,2)
  online_mat = zeros(nr,nc)
  function online_mat!(coeff::AbstractMatrix{Float})
    @assert length(coeff) == Qs "Something is wrong"
    copyto!(online_mat,reshape(basis*coeff,nr,nc))
  end

  online_mat!
end

function get_assembler(
  op::RBUnsteadyVariable,
  basis::AbstractMatrix{Float})

  nsrow,nscol = get_ns(get_rbspace_row(op)),get_ns(get_rbspace_col(op))
  ntrow,ntcol = get_nt(get_rbspace_row(op)),get_nt(get_rbspace_col(op))
  nr,nc = nsrow*ntrow,nscol*ntcol
  Qs = size(basis,2)

  online_mat = zeros(nr,nc)
  mat = zeros(nr,nc)
  function online_mat!(coeff::AbstractMatrix{Float})
    @assert length(coeff) == Qs "Something is wrong"
    @inbounds for q = 1:Qs
      basis_q = reshape(basis[:,q],nsrow,nscol)
      coeff_q = reshape(coeff[:,q],ntrow,ntcol)
      mat += kron(basis_q,coeff_q)
    end
    copyto!(online_mat,mat)
  end

  online_mat!
end

get_op(param_os::RBParamOnlineStructure) = param_os.op

get_assembler(param_os::RBParamOnlineStructure) = param_os.assembler

function assemble(
  ::RBVariable,
  assembler::Function,
  args...)

  assembler(args...)
end

function assemble(
  ::RBLiftVariable,
  assembler::Function,
  args...)

  -assembler(args...)
end

function assemble(param_os::RBParamOnlineStructure,args...)::Matrix{Float}
  op = get_op(param_os)
  assembler = get_assembler(param_os)
  assemble(op,assembler,args...)
end

function assemble(
  param_os::NTuple{N,RBParamOnlineStructure},
  sym::Symbol,
  args...)::Matrix{Float} where N

  syms = get_id.(get_op.(param_os))
  idx = findall(x -> x == sym,syms)[1]
  op = get_op(param_os[idx])
  assembler = get_assembler(param_os[idx])
  assemble(op,assembler,args...)
end
