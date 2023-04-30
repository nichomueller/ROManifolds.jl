struct RBParamOnlineStructure
  op::RBVariable
  assembler::Function
end

function RBParamOnlineStructure(ad::RBAffineDecomposition;kwargs...)
  op = get_op(ad)
  ad_eval = eval_affine_decomposition(ad)
  basis = get_affine_decomposition(ad)
  coeff = get_coefficient(op,basis;kwargs...)
  assembler = get_assembler(op,ad_eval,get_nrows(op),get_ncols(op))

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
  basis::Vector{<:AbstractMatrix{Float}},
  nr::Int,
  nc::Int)

  Qs = length(basis)
  online_mat = Elemental.zeros(EMatrix{Float},nr,nc)
  function online_mat!(coeff::Vector{<:AbstractMatrix{Float}})
    @assert length(coeff) == Qs "Something is wrong"
    copyto!(online_mat,sum([basis[k]*coeff[k] for k = 1:Qs]))
  end

  online_mat!
end

function get_assembler(
  ::RBUnsteadyVariable,
  basis::Vector{<:AbstractMatrix{Float}};
  nr::Int)

  Qs = size(basis,3)
  online_mat = Elemental.zeros(EMatrix{Float},nr,nc)
  function online_mat!(coeff::Vector{<:AbstractMatrix{Float}})
    @assert length(coeff) == Qs "Something is wrong"
    copyto!(online_mat,sum([kron(basis[k],coeff[k]) for k = 1:Qs]))
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
