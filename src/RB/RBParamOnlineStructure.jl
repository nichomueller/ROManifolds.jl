struct RBParamOnlineStructure
  op::RBVariable
  assembler::Function
end

function RBParamOnlineStructure(adrb::RBAffineDecomposition;kwargs...)
  op = get_op(adrb)
  basis = get_rb_aff_dec(adrb)
  ad_eval = eval_affine_decomposition(adrb)
  coeff = get_coefficient(op,basis;kwargs...)
  assembler = get_assembler(op,ad_eval)

  RBParamOnlineStructure(op,coeff,assembler)
end

function RBParamOnlineStructure(
  op::RBVariable,
  coeff::Function,
  assembler::Function)

  RBParamOnlineStructure(op,μ::Param -> assembler(coeff(μ)))
end

function RBParamOnlineStructure(
  op::RBVariable{Nonlinear,Ttr},
  coeff::Function,
  assembler::Function) where Ttr

  RBParamOnlineStructure(op,(μ::Param,z) -> assembler(coeff(μ,z)))
end

function get_assembler(
  op::RBSteadyVariable,
  basis::AbstractMatrix{Float})

  nr = get_nrows(op)
  nc = get_ncols(op)
  Qs = size(basis,2)

  online_mat = allocate_matrix(Matrix{Float},nr,nc)
  function online_mat!(coeff::AbstractMatrix{Float})
    @assert size(coeff,2) == Qs "Something is wrong"
    copyto!(online_mat,reshape(basis*coeff',nr,nc))
  end

  online_mat!
end

function get_assembler(
  op::RBUnsteadyVariable,
  basis::AbstractMatrix{Float})

  nsrow,nscol = get_ns_row(op),get_ns_col(op)
  ntrow,ntcol = get_nt_row(op),get_nt_col(op)
  nr,nc = nsrow*ntrow,nscol*ntcol
  Qs = size(basis,2)

  online_mat = allocate_matrix(Matrix{Float},nr,nc)
  function online_mat!(coeff::AbstractMatrix{Float})
    @assert size(coeff,2) == Qs "Something is wrong"
    mat = @distributed (+) for q = 1:Qs
      basis_q = reshape(basis[:,q],nsrow,nscol)
      coeff_q = reshape(coeff[:,q],ntrow,ntcol)
      kron(basis_q,coeff_q)
    end
    copyto!(online_mat,sum(mat))
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
